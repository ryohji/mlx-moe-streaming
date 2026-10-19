"""簡易品質比較: ワークロードのプロンプト文を、そのままテキストとして perplexity を測る。

uv run python bench/quality.py models/qwen3-30b-a3b-4bit-smoe [--smoe] [--per 4]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--smoe", action="store_true")
    ap.add_argument("--per", type=int, default=4, help="ワークロードごとに使う文数")
    ap.add_argument("--cache-gb", type=float, default=2.0)
    args = ap.parse_args()
    if args.smoe:
        from smoe import load_streaming

        model, tok, _ = load_streaming(args.model, cache_gb=args.cache_gb)
    else:
        from mlx_lm import load

        model, tok = load(args.model)
    res = {"model": args.model}
    for w in ("code", "ja_chat", "en_prose"):
        lines = Path(f"analysis/workloads/{w}.jsonl").read_text().splitlines()
        # 長い順に選ぶ(短文は perplexity が不安定)
        texts = sorted((json.loads(s)["prompt"] for s in lines if s.strip()), key=len)[-args.per:]
        nll = n = 0
        for t in texts:
            ids = mx.array(tok.encode(t))[None]
            logits = model(ids[:, :-1])
            loss = nn.losses.cross_entropy(logits.astype(mx.float32), ids[:, 1:], reduction="sum")
            mx.eval(loss)
            nll += loss.item()
            n += ids.shape[1] - 1
        res[w] = {"tokens": n, "ppl": round(math.exp(nll / n), 3)}
        print(w, res[w], file=sys.stderr, flush=True)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
