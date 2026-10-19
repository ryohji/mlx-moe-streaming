"""ストリーミング版の生成速度とヒット率を、キャッシュ容量を変えて測る。

uv run python bench/stream_bench.py models/olmoe-4bit-smoe --cache-gb 0.72 1.09 1.8
"""

import argparse
import json
import os
import subprocess
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm import stream_generate  # noqa: E402

from smoe import load_streaming  # noqa: E402

PROMPTS = [
    "Write a Python class implementing an LRU cache with get and put, and explain its complexity.",
    "日本の梅雨の時期に洗濯物を早く乾かすコツを教えてください。",
    "Summarize the main causes of the Industrial Revolution in a short paragraph.",
]


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cache-gb", type=float, nargs="+", default=[1.0])
    ap.add_argument("--io-mode", default="pread")
    ap.add_argument("--io-threads", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--prefetch-m", type=int, default=0, help="次の層の予測上位 m 個を先読み(0 で無効)")
    ap.add_argument("--prefetch-early", action="store_true", help="この層のミスより先に先読みを発行する")
    ap.add_argument("--prefetch-m2", type=int, default=0, help="2 層先の予測上位 m2 個も先読み")
    ap.add_argument("--pf-threads", type=int, default=1, help="先読み専用スレッド数")
    ap.add_argument("--min-prob", type=float, default=0.0, help="ルーター確率がこれ未満の予測は先読みしない")
    ap.add_argument("--keep-unused", action="store_true", help="外れた先読みもキャッシュに入れる")
    ap.add_argument("--per-layer", action="store_true", help="層ごとに容量を等分した LRU")
    ap.add_argument("--draft", help="投機デコードのドラフトモデル(例: mlx-community/Qwen3-0.6B-4bit)")
    ap.add_argument("--num-draft", type=int, default=3)
    args = ap.parse_args()

    model, tokenizer, cache = load_streaming(
        args.path, cache_gb=args.cache_gb[0], io_mode=args.io_mode, io_threads=args.io_threads,
        prefetch_m=args.prefetch_m, prefetch_early=args.prefetch_early, prefetch_m2=args.prefetch_m2,
        pf_threads=args.pf_threads, prefetch_min_prob=args.min_prob,
        keep_unused_prefetch=args.keep_unused, per_layer=args.per_layer,
    )
    draft = None
    if args.draft:
        from mlx_lm import load

        draft, _ = load(args.draft)
    resident = mx.get_active_memory() / 1e9
    total = sum(v["num_experts"] * v["blob_nbytes"] for v in cache.store.layers.values()) / 1e9
    print(f"# budget: resident {resident:.2f}GB, experts on disk {total:.2f}GB", file=sys.stderr)
    for gb in args.cache_gb:
        cache.capacity = int(gb * 1e9)
        while cache.used > cache.capacity and cache.items:
            cache._evict(next(iter(cache.items)))
        mx.clear_cache()
        mx.reset_peak_memory()
        swap0 = swap_used_mb()
        gen_tok = gen_s = pre_tok = pre_s = 0
        accepted = 0
        for i, prompt in enumerate(PROMPTS):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            )
            if i == 1:
                cache.reset_stats()  # 1 本目はウォームアップ扱い
            kw = {"draft_model": draft, "num_draft_tokens": args.num_draft} if draft else {}
            n_draft = 0
            for r in stream_generate(model, tokenizer, ids, max_tokens=args.max_tokens, **kw):
                n_draft += bool(getattr(r, "from_draft", False))
            if i >= 1:
                accepted += n_draft
                gen_tok += r.generation_tokens
                gen_s += r.generation_tokens / r.generation_tps
                pre_tok += r.prompt_tokens
                pre_s += r.prompt_tokens / r.prompt_tps
        s = cache.stats()
        print(json.dumps({
            "cache_gb": gb,
            "cache_pct": round(100 * gb / total),
            "io": args.io_mode,
            "gen_tps": round(gen_tok / gen_s, 1),
            "prefill_tps": round(pre_tok / pre_s, 1),
            "hit_rate": s["hit_rate"],
            "read_mb_per_tok": round(1e3 * s["read_gb"] / (gen_tok + pre_tok), 1),
            "read_s_share": round(s["read_s"] / (gen_s + pre_s), 2),
            "io_s_share": round(s["io_s"] / (gen_s + pre_s), 2),
            "prefetch_m": args.prefetch_m,
            "early": args.prefetch_early,
            "keep_unused": args.keep_unused,
            "per_layer": args.per_layer,
            "draft": args.draft and args.num_draft,
            "accept_share": round(accepted / gen_tok, 3) if draft else None,
            "pf_cover": s["pf_cover"],
            "pf_waste": round(1 - s["pf_used"] / (s["pf_issued"] - s["pf_cancelled"]), 3)
            if s["pf_issued"] > s["pf_cancelled"] else None,
            "pf_cancelled": s["pf_cancelled"],
            "io_gbps": round(s["read_gb"] / s["io_s"], 2) if s["io_s"] else None,
            "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "swap_delta_mb": round(swap_used_mb() - swap0),
        }), flush=True)


if __name__ == "__main__":
    main()
