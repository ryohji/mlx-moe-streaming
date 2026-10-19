"""prefill と生成の各区間で、キャッシュのヒット率と速度を分けて測る。

uv run python bench/phase_hits.py models/<dir> --cache-gb 3.0 --prefetch-m 10 --out bench/logs/phase_<name>.jsonl

区間: prefill / 生成の最初の 10 トークン / それ以降。
プロンプト: HumanEval 4 問(コード)と会話 2 本(速度ベンチと同じもの)。最初にウォームアップを 1 本流す。
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from mlx_lm import stream_generate  # noqa: E402

from humaneval import load_problems, make_prompt, select  # noqa: E402
from smoe import load_streaming  # noqa: E402

CHAT = [
    "日本の梅雨の時期に洗濯物を早く乾かすコツを教えてください。",
    "Summarize the main causes of the Industrial Revolution in a short paragraph.",
]
HEAD = 10


def chat_ids(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True)


def run(model, tok, cache, ids, max_tokens):
    """1 本ぶん。区間ごとの (hits, misses, 秒, トークン数) を返す。"""
    if isinstance(ids, str):
        ids = tok.encode(ids, add_special_tokens=False)
    snap = lambda: (cache.hits, cache.misses, cache.pf_used, time.perf_counter(),  # noqa: E731
                    cache.pf_issued - cache.pf_cancelled)
    marks = [snap()]  # [開始, prefill 後(=1 トークン目), 2 トークン目, ...]
    for _ in stream_generate(model, tok, ids, max_tokens=max_tokens):
        marks.append(snap())

    def seg(a, b):
        h = marks[b][0] - marks[a][0]
        m = marks[b][1] - marks[a][1]
        return {"hit_rate": round(h / (h + m), 3) if h + m else None,
                "pf_cover": round((marks[b][2] - marks[a][2]) / m, 3) if m else None,
                # 発行した先読みのうち使われなかった割合(無駄読み)
                "pf_waste": round(1 - (marks[b][2] - marks[a][2]) / (marks[b][4] - marks[a][4]), 3)
                if marks[b][4] > marks[a][4] else None,
                "tokens": b - a, "s": round(marks[b][3] - marks[a][3], 2),
                "tps": round((b - a) / (marks[b][3] - marks[a][3]), 2) if b > a else None}

    n = len(marks) - 1  # 生成トークン数
    # marks[1] は 1 トークン目が出た時点 = prefill の終わり
    out = {"prompt_tokens": len(ids), "gen_tokens": n, "prefill": seg(0, 1)}
    out["prefill"]["tps"] = round(len(ids) / out["prefill"]["s"], 2)
    out["head"] = seg(1, min(1 + HEAD, n))
    out["tail"] = seg(min(1 + HEAD, n), n) if n > 1 + HEAD else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cache-gb", type=float, default=3.0)
    ap.add_argument("--prefetch-m", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prefill-admit", default="tail", choices=["all", "none", "tail"])
    ap.add_argument("--prefetch-m2", type=int, default=0, help="2 層先の予測上位 m2 個も先読み")
    ap.add_argument("--pf-threads", type=int, default=1, help="先読み専用スレッド数")
    ap.add_argument("--min-prob", type=float, default=0.0, help="ルーター確率がこれ未満の予測は先読みしない")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok, cache = load_streaming(args.path, cache_gb=args.cache_gb, prefetch_m=args.prefetch_m,
                                     prefill_admit=args.prefill_admit, prefetch_m2=args.prefetch_m2,
                                     pf_threads=args.pf_threads, prefetch_min_prob=args.min_prob)
    probs = select(load_problems(), 4, ids=["HumanEval/2", "HumanEval/12", "HumanEval/37", "HumanEval/101"])
    jobs = [("warmup", chat_ids(tok, "Hello! Introduce yourself briefly."))]
    jobs += [(p["task_id"], make_prompt(tok, p)) for p in probs]
    jobs += [(f"chat-{i}", chat_ids(tok, t)) for i, t in enumerate(CHAT)]
    with open(args.out, "w") as f:
        for name, ids in jobs:
            r = {"name": name, **run(model, tok, cache, ids, args.max_tokens)}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
