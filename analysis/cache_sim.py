"""Expert-routing analysis and cache simulation.

uv run python analysis/cache_sim.py LOG.npz [--out report.md]

Cache unit = (layer, expert) pair. Access order: t asc -> l asc -> k experts of that
(t, l) "group". Within a group, lookups happen first and the group's experts never evict
each other. The cache persists across prompts. Policies are demand-fill except Belady,
which may bypass (not insert a fetched expert whose next use is farthest).
"""
import argparse
import heapq
import sys
import time
from collections import OrderedDict

import numpy as np

FRACS = (0.10, 0.20, 0.30, 0.50)
CAP_FRACS = (0.10, 0.20, 0.30, 0.40, 0.50)


# ------------------------------------------------------------------ loading
def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["num_experts"] = int(d["num_experts"])
    d["topk"] = d["topk"].astype(np.int64)
    if "pred_offsets" not in d and "pred" in d:
        d["pred_offsets"] = np.arange(d["pred"].shape[0])
    return d


def build_stream(topk, E):
    """keys [G,k] with G = T*L (group g = t*L + l), key = l*E + e."""
    T, L, k = topk.shape
    keys = (np.arange(L)[None, :, None] * E + topk).reshape(T * L, k)
    return keys


# ------------------------------------------------------------------ a. skew
def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    s = x.sum()
    if n == 0 or s == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def top_share(counts, p):
    n = max(1, int(round(p * len(counts))))
    c = np.sort(counts)[::-1]
    return c[:n].sum() / max(c.sum(), 1)


def skew(topk, E):
    T, L, k = topk.shape
    counts = np.zeros((L, E), np.int64)
    for l in range(L):
        counts[l] = np.bincount(topk[:, l, :].ravel(), minlength=E)
    per_layer = [[top_share(counts[l], p) for p in FRACS] + [gini(counts[l])] for l in range(L)]
    overall = [top_share(counts.ravel(), p) for p in FRACS]
    return counts, np.array(per_layer), np.array(overall), gini(counts.ravel())


# ------------------------------------------------------------------ b. cache policies
# Each sim returns hits per group (int array [G]).
def next_use(keys):
    G, k = keys.shape
    flat = keys.ravel()
    gid = np.repeat(np.arange(G), k)
    order = np.argsort(flat, kind="stable")
    sk, sg = flat[order], gid[order]
    nxt_sorted = np.full(len(flat), G, np.int64)  # G == never
    same = sk[1:] == sk[:-1]
    nxt_sorted[:-1] = np.where(same, sg[1:], G)
    nxt = np.empty(len(flat), np.int64)
    nxt[order] = nxt_sorted
    return nxt.reshape(G, k)


def sim_lru(keys, C):
    G = len(keys)
    hits = np.zeros(G, np.int64)
    od = OrderedDict()
    for g, grp in enumerate(keys.tolist()):
        h = 0
        miss = []
        for x in grp:
            if x in od:
                od.move_to_end(x)
                h += 1
            else:
                miss.append(x)
        gs = set(grp)
        for x in miss:
            od[x] = None
        while len(od) > C:
            first = next(iter(od))
            if first in gs:
                break
            del od[first]
        hits[g] = h
    return hits


def sim_lfu(keys, C):
    """Perfect LFU: counts persist after eviction; ties broken by least recent."""
    G = len(keys)
    hits = np.zeros(G, np.int64)
    cnt = {}
    last = {}
    cache = set()
    heap = []  # (count, last, key)
    for g, grp in enumerate(keys.tolist()):
        h = 0
        gs = set(grp)
        for x in grp:
            if x in cache:
                h += 1
            cnt[x] = cnt.get(x, 0) + 1
            last[x] = g
        for x in grp:
            cache.add(x)
            heapq.heappush(heap, (cnt[x], g, x))
        if len(cache) > C:
            held = []
            while len(cache) > C and heap:
                c, lt, x = heapq.heappop(heap)
                if x not in cache or cnt[x] != c or last[x] != lt:
                    continue  # stale
                if x in gs:
                    held.append((c, lt, x))
                    continue
                cache.discard(x)
            for e in held:
                heapq.heappush(heap, e)
        hits[g] = h
    return hits


def sim_belady(keys, C, nxt):
    G = len(keys)
    hits = np.zeros(G, np.int64)
    cache = {}  # key -> next use
    heap = []  # (-next_use, key)
    kl, nl = keys.tolist(), nxt.tolist()
    for g in range(G):
        grp, nu = kl[g], nl[g]
        h = 0
        hit_keys = set()
        for x, n in zip(grp, nu):
            if x in cache:
                h += 1
                hit_keys.add(x)
            cache[x] = n
            heapq.heappush(heap, (-n, x))
        hits[g] = h
        if len(cache) > C:
            held = []
            while len(cache) > C and heap:
                n, x = heapq.heappop(heap)
                if cache.get(x) != -n:
                    continue  # stale
                if x in hit_keys:
                    held.append((n, x))
                    continue
                del cache[x]
            for e in held:
                heapq.heappush(heap, e)
    return hits


def _first_free(od, pinned):
    """First key of OrderedDict `od` (LRU/FIFO order) not in `pinned`, else None."""
    for y in od:
        if y not in pinned:
            return y
    return None


def sim_arc(keys, C):
    """ARC (Megiddo & Modha). Group members are pinned (never chosen as victims)."""
    G = len(keys)
    hits = np.zeros(G, np.int64)
    T1, T2, B1, B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p = 0.0

    def replace(in_b2, pinned):
        first, second = (T1, T2) if (T1 and (len(T1) > p or (in_b2 and len(T1) == p))) else (T2, T1)
        for src in (first, second):
            y = _first_free(src, pinned)
            if y is not None:
                del src[y]
                (B1 if src is T1 else B2)[y] = None
                return

    for g, grp in enumerate(keys.tolist()):
        pinned = set(grp)
        miss = []
        h = 0
        for x in grp:
            if x in T1:
                del T1[x]
                T2[x] = None
                h += 1
            elif x in T2:
                T2.move_to_end(x)
                h += 1
            else:
                miss.append(x)
        for x in miss:
            full = len(T1) + len(T2) >= C
            if x in B1:
                p = min(C, p + max(len(B2) / len(B1), 1))
                del B1[x]
                if full:
                    replace(False, pinned)
                T2[x] = None
            elif x in B2:
                p = max(0.0, p - max(len(B1) / len(B2), 1))
                del B2[x]
                if full:
                    replace(True, pinned)
                T2[x] = None
            else:
                if len(T1) + len(B1) >= C:
                    if B1:
                        B1.popitem(last=False)
                        if full:
                            replace(False, pinned)
                    else:
                        y = _first_free(T1, pinned)
                        if y is not None:
                            del T1[y]
                elif full:
                    if len(T1) + len(T2) + len(B1) + len(B2) >= 2 * C and B2:
                        B2.popitem(last=False)
                    replace(False, pinned)
                T1[x] = None
        while len(B1) + len(T1) > C and B1:
            B1.popitem(last=False)
        while len(B2) > C:
            B2.popitem(last=False)
        hits[g] = h
    return hits


def sim_2q(keys, C, kin_frac=0.25, kout_frac=0.5):
    """Full 2Q: A1in FIFO (25% of C), A1out ghost (50% of C), Am LRU."""
    G = len(keys)
    hits = np.zeros(G, np.int64)
    kin, kout = max(1, int(C * kin_frac)), max(1, int(C * kout_frac))
    A1, Am, A1o = OrderedDict(), OrderedDict(), OrderedDict()
    for g, grp in enumerate(keys.tolist()):
        pinned = set(grp)
        miss = []
        h = 0
        for x in grp:
            if x in Am:
                Am.move_to_end(x)
                h += 1
            elif x in A1:
                h += 1
            else:
                miss.append(x)
        for x in miss:
            if x in A1o:
                del A1o[x]
                Am[x] = None
            else:
                A1[x] = None
        while len(A1) + len(Am) > C:
            order = (A1, Am) if len(A1) > kin else (Am, A1)
            for src in order:
                y = _first_free(src, pinned)
                if y is not None:
                    del src[y]
                    if src is A1:
                        A1o[y] = None
                        while len(A1o) > kout:
                            A1o.popitem(last=False)
                    break
            else:
                break
        hits[g] = h
    return hits


def sim_lrfu(keys, C, half_life_groups):
    """LRFU-style decayed LFU. Score = sum 2^(-(now-t_i)/H) over past accesses of cached keys
    (reset on eviction). Stored as lk = log2(score) + last/H, which orders identically at any
    time. Time unit = group index (1 group = k accesses; H given in groups, e.g. L = one token)."""
    G = len(keys)
    hits = np.zeros(G, np.int64)
    inv = 1.0 / half_life_groups
    lk = {}  # cached key -> log-score key
    heap = []  # (lk, key), lazy
    for g, grp in enumerate(keys.tolist()):
        now = g * inv
        h = 0
        gs = set(grp)
        for x in grp:
            if x in lk:
                h += 1
                v = float(np.logaddexp2(lk[x], now))
            else:
                v = now  # log2(1) + now
            lk[x] = v
            heapq.heappush(heap, (v, x))
        if len(lk) > C:
            held = []
            while len(lk) > C and heap:
                v, x = heapq.heappop(heap)
                if lk.get(x) != v:
                    continue
                if x in gs:
                    held.append((v, x))
                    continue
                del lk[x]
            for e in held:
                heapq.heappush(heap, e)
        hits[g] = h
    return hits


def sim_s3fifo(keys, C):
    """S3-FIFO: small FIFO (10% of C), main FIFO, ghost FIFO (size of main); 2-bit freq."""
    G = len(keys)
    hits = np.zeros(G, np.int64)
    ss = max(1, int(C * 0.1))
    gcap = max(1, C - ss)
    S, M, Gh = OrderedDict(), OrderedDict(), OrderedDict()

    def evict(pinned):
        tries = 0
        lim = 4 * (len(S) + len(M)) + 8
        while tries < lim:
            tries += 1
            if (len(S) >= ss or not M) and S:
                x, f = S.popitem(last=False)
                if x in pinned:
                    S[x] = f
                elif f > 1:
                    M[x] = 0
                else:
                    Gh[x] = None
                    while len(Gh) > gcap:
                        Gh.popitem(last=False)
                    return
            elif M:
                x, f = M.popitem(last=False)
                if x in pinned:
                    M[x] = f
                elif f > 0:
                    M[x] = f - 1
                else:
                    return
            else:
                return

    for g, grp in enumerate(keys.tolist()):
        pinned = set(grp)
        miss = []
        h = 0
        for x in grp:
            if x in M:
                M[x] = min(M[x] + 1, 3)
                h += 1
            elif x in S:
                S[x] = min(S[x] + 1, 3)
                h += 1
            else:
                miss.append(x)
        for x in miss:
            if x in Gh:
                del Gh[x]
                M[x] = 0
            else:
                S[x] = 0
        while len(S) + len(M) > C:
            n0 = len(S) + len(M)
            evict(pinned)
            if len(S) + len(M) >= n0:
                break
        hits[g] = h
    return hits


def sim_lru_look(keys, C, prot):
    """LRU that skips eviction candidates in prot[g] (keys predicted for the next layer,
    -1 = none). Falls back to plain LRU victim if every candidate is protected."""
    G = len(keys)
    hits = np.zeros(G, np.int64)
    od = OrderedDict()
    pl = prot.tolist()
    for g, grp in enumerate(keys.tolist()):
        h = 0
        miss = []
        for x in grp:
            if x in od:
                od.move_to_end(x)
                h += 1
            else:
                miss.append(x)
        gs = set(grp)
        for x in miss:
            od[x] = None
        if len(od) > C:
            pr = set(pl[g])
            victims = []
            need = len(od) - C
            for y in od:
                if y not in gs and y not in pr:
                    victims.append(y)
                    if len(victims) == need:
                        break
            if len(victims) < need:  # fallback: LRU among non-group
                have = set(victims)
                for y in od:
                    if y not in gs and y not in have:
                        victims.append(y)
                        if len(victims) == need:
                            break
            for y in victims:
                del od[y]
        hits[g] = h
    return hits


def sim_lru_perlayer(keys, C, L):
    """Independent LRU per layer with fixed capacity C//L (remainder to the first layers)."""
    hits = np.zeros(len(keys), np.int64)
    base, rem = divmod(C, L)
    for l in range(L):
        c = max(1, base + (1 if l < rem else 0))
        hits[l::L] = sim_lru(keys[l::L], c)
    return hits


def lookahead_protect(pred, offsets, E, L, k):
    """prot [T*L, k]: at (t,l), top-k of pred[d=1,t,l] as keys of layer l+1; -1 for last layer."""
    di = [int(d) for d in offsets].index(1)
    T = pred.shape[1]
    pk = pred[di, :, :L - 1, :k].astype(np.int64)
    pk = np.where(pk < 0, -1, np.arange(1, L)[None, :, None] * E + pk)
    prot = np.full((T, L, k), -1, np.int64)
    prot[:, :L - 1] = pk
    return prot.reshape(T * L, k)


def sim_static(keys, C, count_vec):
    """Fixed cache = top-C keys of count_vec. Returns hits per group."""
    top = np.argsort(-count_vec, kind="stable")[:C]
    inset = np.zeros(len(count_vec), bool)
    inset[top] = True
    return inset[keys].sum(axis=1)


def cache_eval(topk, E, is_gen, seq_id, fracs=CAP_FRACS, pred=None, pred_offsets=None):
    T, L, k = topk.shape
    keys = build_stream(topk, E)
    nxt = next_use(keys)
    N = L * E
    gen_g = np.repeat(is_gen, L)
    # split static: first half of sequences (by order of appearance) -> frequency, rest -> eval
    useq = np.unique(seq_id)
    first_half = np.isin(seq_id, useq[: max(1, len(useq) // 2)])
    tok_idx = np.arange(T)
    train_g = np.repeat(first_half, L)
    cnt_all = np.bincount(keys.ravel(), minlength=N)
    cnt_tr = np.bincount(keys[train_g].ravel(), minlength=N)
    prot = None
    if pred is not None and pred_offsets is not None and 1 in [int(x) for x in pred_offsets]:
        prot = lookahead_protect(pred, pred_offsets, E, L, k)
    H1 = L  # one token = L groups (L*k accesses)
    rows = []
    times = {}
    for f in fracs:
        C = max(1, int(round(f * N)))
        res = {}
        for name, fn in (("LRU", lambda: sim_lru(keys, C)),
                         ("LFU", lambda: sim_lfu(keys, C)),
                         ("Static(oracle)", lambda: sim_static(keys, C, cnt_all)),
                         ("ARC", lambda: sim_arc(keys, C)),
                         ("2Q", lambda: sim_2q(keys, C)),
                         ("LRFU(1tok)", lambda: sim_lrfu(keys, C, H1)),
                         ("LRFU(4tok)", lambda: sim_lrfu(keys, C, 4 * H1)),
                         ("LRFU(16tok)", lambda: sim_lrfu(keys, C, 16 * H1)),
                         ("S3-FIFO", lambda: sim_s3fifo(keys, C)),
                         ("LRU/layer", lambda: sim_lru_perlayer(keys, C, L)),
                         *((("LRU+look", lambda: sim_lru_look(keys, C, prot)),) if prot is not None else ()),
                         ("Belady", lambda: sim_belady(keys, C, nxt))):
            t0 = time.time()
            res[name] = fn()
            times[name] = times.get(name, 0) + time.time() - t0
        h = sim_static(keys, C, cnt_tr)
        res["Static(split)"] = h
        for name, hits in res.items():
            if name == "Static(split)":
                m = ~train_g
                allr = hits[m].sum() / (m.sum() * k)
                mg = m & gen_g
                genr = hits[mg].sum() / max(mg.sum() * k, 1)
            else:
                allr = hits.sum() / (len(keys) * k)
                genr = hits[gen_g].sum() / max(gen_g.sum() * k, 1)
            rows.append((f, C, name, allr, genr))
    distinct = len(np.unique(keys))
    return rows, times, 1 - distinct / keys.size


# ------------------------------------------------------------------ c. lookahead recall
def lookahead(topk, pred, offsets, is_gen, chunk=2048):
    D, T, L, K2 = pred.shape
    k = topk.shape[2]
    out = []
    for di, d in enumerate(offsets):
        d = int(d)
        n = L - d
        rk = np.zeros((T, max(n, 0)))
        r2k = np.zeros((T, max(n, 0)))
        for s in range(0, T, chunk):
            p = pred[di, s:s + chunk, :n].astype(np.int64)  # [c,n,2k]
            tgt = topk[s:s + chunk, d:d + n]  # [c,n,k]
            m = (tgt[..., :, None] == p[..., None, :])  # [c,n,k,2k]
            rk[s:s + chunk] = m[..., :k].any(-1).sum(-1) / k
            r2k[s:s + chunk] = m.any(-1).sum(-1) / k
        out.append((d, rk, r2k))
    return out


# ------------------------------------------------------------------ d. temporal locality
def temporal(topk, seq_id, is_gen, windows=(1, 4, 16), chunk=2048):
    T, L, k = topk.shape
    res = {}
    for w in windows:
        cov = np.zeros((T, L))
        valid = np.zeros(T, bool)
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            cur = topk[s:e]  # [c,L,k]
            inu = np.zeros(cur.shape, bool)
            anyv = np.zeros(e - s, bool)
            for j in range(1, w + 1):
                idx = np.arange(s, e) - j
                ok = idx >= 0
                idxc = np.clip(idx, 0, None)
                ok &= seq_id[idxc] == seq_id[s:e]
                prev = topk[idxc]  # [c,L,k]
                m = (cur[..., :, None] == prev[..., None, :]).any(-1)
                inu |= m & ok[:, None, None]
                anyv |= ok
            cov[s:e] = inu.sum(-1) / k
            valid[s:e] = anyv
        res[w] = (cov, valid)
    return res


# ------------------------------------------------------------------ e. co-occurrence
def cooccur(topk, E, top_pairs=3):
    T, L, k = topk.shape
    per_layer = []
    pairs = []
    for l in range(L):
        H = np.zeros((T, E), np.float32)
        np.put_along_axis(H, topk[:, l, :], 1.0, axis=1)
        C = H.T @ H
        diag = np.diag(C).copy()
        off = C.copy()
        np.fill_diagonal(off, -1)
        best = off.max(axis=1)
        m = diag > 0
        per_layer.append(float((best[m] / diag[m]).mean()))
        iu = np.triu_indices(E, 1)
        vals = C[iu]
        top = np.argsort(-vals)[:top_pairs]
        pairs.append([(int(iu[0][i]), int(iu[1][i]), int(vals[i])) for i in top])
    return np.array(per_layer), pairs


# ------------------------------------------------------------------ report
def pct(x):
    return f"{100 * x:.1f}"


def report(d):
    topk, E = d["topk"], d["num_experts"]
    seq_id, is_gen = d["seq_id"], d["is_gen"].astype(bool)
    T, L, k = topk.shape
    o = []
    o.append(f"# Routing analysis\n\nT={T} (gen {int(is_gen.sum())}), L={L}, E={E}, k={k}, "
             f"sequences={len(np.unique(seq_id))}\n")

    counts, pl, ov, g = skew(topk, E)
    o.append("## a. Usage skew (share of accesses covered by top x% experts)\n")
    o.append("| scope | top10% | top20% | top30% | top50% | gini |\n|---|---|---|---|---|---|")
    o.append("| ALL (L*E pairs) | " + " | ".join(pct(x) for x in ov) + f" | {g:.3f} |")
    for l in range(L):
        o.append(f"| layer {l} | " + " | ".join(pct(x) for x in pl[l, :4]) + f" | {pl[l, 4]:.3f} |")
    o.append(f"| layer mean | " + " | ".join(pct(x) for x in pl[:, :4].mean(0)) + f" | {pl[:, 4].mean():.3f} |\n")

    t0 = time.time()
    rows, times, first = cache_eval(topk, E, is_gen, seq_id, pred=d.get("pred"), pred_offsets=d.get("pred_offsets"))
    o.append("## b. Cache hit rate (%)\n")
    o.append(f"Capacity = fraction of L*E={L * E} pairs. Cold-miss floor: max hit rate = {pct(first)}%. "
             "Static(split) is evaluated on the 2nd half of sequences only (others: whole log).\n")
    o.append("| capacity | policy | hit (all tokens) | hit (gen only) |\n|---|---|---|---|")
    for f, C, name, a, gg in rows:
        o.append(f"| {int(f * 100)}% ({C}) | {name} | {pct(a)} | {pct(gg)} |")
    o.append("\nSim time (s): " + ", ".join(f"{n} {t:.1f}" for n, t in times.items()) + "\n")

    if "pred" in d:
        offs = [int(x) for x in d["pred_offsets"]]
        res = lookahead(topk, d["pred"], offs, is_gen)
        o.append("## c. Lookahead recall (%) of pred vs actual topk[t, l+d]\n")
        o.append("| d | recall@k all | recall@2k all | recall@k gen | recall@2k gen |\n|---|---|---|---|---|")
        for dd, rk, r2 in res:
            if rk.shape[1] == 0:
                continue
            o.append(f"| {dd} | {pct(rk.mean())} | {pct(r2.mean())} | {pct(rk[is_gen].mean())} | {pct(r2[is_gen].mean())} |")
        o.append("\nPer layer (layer = input layer l; recall@k / recall@2k, all tokens):\n")
        o.append("| layer | " + " | ".join(f"d={dd}" for dd, *_ in res) + " |\n|---|" + "---|" * len(res))
        for l in range(L):
            cells = []
            for dd, rk, r2 in res:
                cells.append(f"{pct(rk[:, l].mean())} / {pct(r2[:, l].mean())}" if l < rk.shape[1] else "-")
            o.append(f"| {l} | " + " | ".join(cells) + " |")
        o.append("")

    tr = temporal(topk, seq_id, is_gen)
    o.append("## d. Temporal locality (same layer, within a prompt)\n")
    o.append("Share of current top-k found in the previous w tokens' union (w=1: overlap rate).\n")
    o.append("| window | all tokens | gen only |\n|---|---|---|")
    for w, (cov, valid) in tr.items():
        o.append(f"| {w} | {pct(cov[valid].mean())} | {pct(cov[valid & is_gen].mean())} |")
    o.append(f"\nRandom baseline (w=1): {pct(k / E)}%\n")

    cl, pairs = cooccur(topk, E)
    o.append("## e. Co-occurrence\n")
    o.append(f"Mean over experts of P(best partner present | expert present): **{cl.mean():.3f}** "
             f"(layer range {cl.min():.3f}-{cl.max():.3f}; random baseline ~{(k - 1) / (E - 1):.3f}; "
             "small-count experts bias this upward)\n")
    o.append("| layer | cond.prob | top pairs (a,b):count |\n|---|---|---|")
    for l in range(L):
        o.append(f"| {l} | {cl[l]:.3f} | " + " ".join(f"({a},{b}):{n}" for a, b, n in pairs[l]) + " |")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    txt = report(load(a.log))
    txt += f"\n_elapsed {time.time() - t0:.1f}s_\n"
    if a.out:
        open(a.out, "w").write(txt)
        print("wrote", a.out, f"({time.time() - t0:.1f}s)")
    else:
        sys.stdout.write(txt)
