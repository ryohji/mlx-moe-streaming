"""Generate a synthetic routing log (npz) in the Phase 1 format.

uv run python analysis/make_fake_log.py OUT.npz [--T 2000 --L 16 --E 64 --k 8 --seed 0]
"""
import argparse
import numpy as np


def make_log(T=2000, L=16, E=64, k=8, offsets=(0, 1, 2), seed=0):
    rng = np.random.default_rng(seed)
    # per-layer Zipf-like popularity with a random permutation of expert ids
    zipf = 1.0 / np.arange(1, E + 1) ** 0.9
    base = np.stack([np.log(zipf[rng.permutation(E)]) for _ in range(L)])  # [L,E]
    # sequences
    seq_id = np.zeros(T, np.int32)
    is_gen = np.zeros(T, bool)
    t, s = 0, 0
    while t < T:
        n = int(rng.integers(40, 160))
        n = min(n, T - t)
        n_pre = min(int(rng.integers(5, 40)), n)
        seq_id[t:t + n] = s
        is_gen[t + n_pre:t + n] = True
        t += n
        s += 1
    # scores: popularity + slowly drifting per-sequence/layer topic + cluster coactivation + noise
    topic = np.zeros((T, L, E))
    cur = None
    for i in range(T):
        if i == 0 or seq_id[i] != seq_id[i - 1]:
            cur = rng.normal(0, 1.0, (L, E))
        else:
            cur = 0.9 * cur + np.sqrt(1 - 0.81) * rng.normal(0, 1.0, (L, E))
        topic[i] = cur
    nclu = E // 4
    z = rng.normal(0, 1.0, (T, L, nclu))
    scores = (base[None] + 0.8 * topic + 1.0 * np.repeat(z, 4, axis=2)[:, :, :E]
              + rng.normal(0, 0.7, (T, L, E)))
    topk_full = np.argsort(-scores, axis=2)[:, :, :k]
    topk = topk_full.astype(np.int16)
    D = len(offsets)
    pred = np.full((D, T, L, 2 * k), -1, np.int16)
    for di, d in enumerate(offsets):
        sigma = 0.3 + 0.6 * d
        for l in range(L - d):
            noisy = scores[:, l + d, :] + rng.normal(0, sigma, (T, E))
            pred[di, :, l, :] = np.argsort(-noisy, axis=1)[:, :2 * k].astype(np.int16)
    return dict(topk=topk, seq_id=seq_id, is_gen=is_gen, pred=pred,
                pred_offsets=np.array(offsets, np.int64), num_experts=np.int64(E))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.savez(a.out, **make_log(a.T, a.L, a.E, a.k, seed=a.seed))
    print("wrote", a.out)
