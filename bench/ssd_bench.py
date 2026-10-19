#!/usr/bin/env python3
"""SSD read benchmark (stdlib only). pread+F_NOCACHE vs mmap, random/sequential."""
import argparse, fcntl, mmap, os, random, statistics, sys, threading, time

F_NOCACHE = 48
ALIGN = 16 * 1024
KB, MB = 1024, 1024 * 1024
BLOCKS = [64 * KB, 256 * KB, 1 * MB, 4 * MB, 8 * MB]
QDS = [1, 2, 4, 8]
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, ".ssd_bench.tmp")


def make_file(path, size):
    chunk = 8 * MB
    with open(path, "wb") as f:
        left = size
        while left > 0:
            n = min(chunk, left)
            f.write(os.urandom(n))
            left -= n
        f.flush()
        os.fsync(f.fileno())


def plan(pattern, size, bs, total, rng):
    n = max(1, total // bs)
    if pattern == "random":
        slots = (size - bs) // ALIGN
        return [rng.randint(0, slots) * ALIGN for _ in range(n)]
    span = (size // bs) * bs
    return [(i * bs) % span for i in range(n)]


def run(offsets, bs, qd, readfn_factory):
    lock = threading.Lock()
    idx = [0]
    lats = [[] for _ in range(qd)]

    def worker(k):
        rd = readfn_factory()
        out = lats[k]
        while True:
            with lock:
                i = idx[0]
                if i >= len(offsets):
                    return
                idx[0] += 1
            off = offsets[i]
            t = time.perf_counter()
            rd(off)
            out.append(time.perf_counter() - t)

    ths = [threading.Thread(target=worker, args=(k,)) for k in range(qd)]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    dt = time.perf_counter() - t0
    all_l = sorted(x for l in lats for x in l)
    mbs = len(offsets) * bs / MB / dt
    p99 = all_l[min(len(all_l) - 1, int(len(all_l) * 0.99))]
    return mbs, statistics.mean(all_l) * 1000, p99 * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-gb", type=float, default=2.0)
    ap.add_argument("--total-mb", type=int, default=512)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    size = int(a.size_gb * 1024 * MB) // ALIGN * ALIGN
    total = a.total_mb * MB
    rng = random.Random(a.seed)
    rows = []
    try:
        print(f"creating {size/MB:.0f} MB test file...", file=sys.stderr)
        make_file(TMP, size)
        fd = os.open(TMP, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        mfd = os.open(TMP, os.O_RDONLY)  # no F_NOCACHE
        mm = mmap.mmap(mfd, size, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        pagesz = mmap.PAGESIZE
        for mode in ("pread+NOCACHE", "mmap"):
            for pat in ("random", "sequential"):
                for bs in BLOCKS:
                    offs = plan(pat, size, bs, total, rng)
                    for qd in QDS:
                        if mode == "mmap":
                            def fac(bs=bs):
                                def rd(off):
                                    mm.madvise(mmap.MADV_WILLNEED, off // pagesz * pagesz, bs)
                                    s = mv[off:off + bs]
                                    for p in range(0, bs, pagesz):
                                        s[p]
                                    s.release()
                                return rd
                        else:
                            def fac(bs=bs):
                                buf = bytearray(bs)  # reused per thread
                                def rd(off):
                                    n = os.preadv(fd, [buf], off)
                                    if n != bs: raise IOError(f"short read {n}")
                                return rd
                        r = run(offs, bs, qd, fac)
                        rows.append((mode, pat, bs, qd) + r)
                        print(f"{mode} {pat} {bs//KB}KB QD{qd}: {r[0]:.0f} MB/s", file=sys.stderr)
        mv.release(); mm.close(); os.close(fd); os.close(mfd)
    finally:
        try: os.unlink(TMP)
        except FileNotFoundError: pass
    lines = [f"# SSD read bench ({size/MB:.0f} MB file, {a.total_mb} MB per cell)", "",
             "- pread+NOCACHE: os.preadv with F_NOCACHE (page cache bypassed).",
             "- mmap: mmap + madvise(WILLNEED) + touch one byte per page. **Not cache-bypassing**: the file was just written, so parts may already be in the page cache, and repeated passes over a 2 GB file can hit cache; treat mmap numbers as an upper bound. Page faults also hold the GIL, so mmap QD scaling is pessimistic.",
             "- random offsets are 16KB-aligned; latency is per block read.", "",
             "| mode | pattern | block | QD | MB/s | avg ms | p99 ms |", "|---|---|---|---|---|---|---|"]
    for m, p, bs, qd, mbs, avg, p99 in rows:
        b = f"{bs//MB}MB" if bs >= MB else f"{bs//KB}KB"
        lines.append(f"| {m} | {p} | {b} | {qd} | {mbs:.0f} | {avg:.2f} | {p99:.2f} |")
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        with open(a.out, "w") as f: f.write(txt)


if __name__ == "__main__":
    main()
