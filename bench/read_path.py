"""エキスパート読み出し経路のマイクロベンチ(モデルは動かさない)。

uv run python bench/read_path.py models/qwen3-30b-a3b-4bit-smoe
1 層ぶんのミス(batch 個)をまとめて読む操作を繰り返し、方式ごとの GB/s を測る。
"""

import argparse
import fcntl
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smoe.loader import ExpertStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rounds", type=int, default=150)
    args = ap.parse_args()
    index = json.loads((Path(args.path) / "index.json").read_text())
    layers = list(index["layers"].values())
    blob = layers[0]["blob_nbytes"]
    fd = os.open(Path(args.path) / "experts.bin", os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)
    rng = random.Random(0)

    def offsets(batch):
        info = rng.choice(layers)
        return [info["offset"] + e * info["stride"] for e in rng.sample(range(info["num_experts"]), batch)]

    def run(name, batch, threads, fn):
        pool = ThreadPoolExecutor(threads) if threads > 1 else None
        bufs = [bytearray(blob) for _ in range(batch)]
        t0 = time.perf_counter()
        for _ in range(args.rounds):
            offs = offsets(batch)
            jobs = list(zip(offs, bufs))
            if pool:
                list(pool.map(fn, jobs))
            else:
                for j in jobs:
                    fn(j)
        dt = time.perf_counter() - t0
        gbps = args.rounds * batch * blob / dt / 1e9
        print(f"{name:28s} batch={batch} threads={threads}: {gbps:.2f} GB/s  {1e3 * dt / args.rounds:.2f} ms/batch", flush=True)

    new_buf = lambda j: os.pread(fd, blob, j[0])
    reuse = lambda j: os.preadv(fd, [j[1]], j[0])

    for batch in (4, 8):
        for threads in (1, 2, 4, 8):
            run("pread (new buffer)", batch, threads, new_buf)
            run("preadv (reused buffer)", batch, threads, reuse)

    # 配列化のコスト(読んだ後に numpy → mx.array へコピーして eval)
    store = ExpertStore(args.path, io_threads=4)
    prefix = next(iter(index["layers"]))
    raw = bytearray(blob)
    t0 = time.perf_counter()
    for _ in range(100):
        arrs = store._to_arrays(prefix, raw)
        mx.eval(*arrs.values())
    dt = (time.perf_counter() - t0) / 100
    print(f"to_arrays + eval: {1e3 * dt:.2f} ms/expert ({blob / dt / 1e9:.2f} GB/s)")
    # 現行の read_many を通した値
    for batch in (4, 8):
        t0 = time.perf_counter()
        for _ in range(50):
            info_prefix = rng.choice(list(index["layers"]))
            store.read_many(info_prefix, rng.sample(range(index["layers"][info_prefix]["num_experts"]), batch))
        dt = time.perf_counter() - t0
        print(f"current read_many batch={batch} (io_threads=4): {50 * batch * blob / dt / 1e9:.2f} GB/s")


if __name__ == "__main__":
    main()
