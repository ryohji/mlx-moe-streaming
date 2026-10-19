"""素の mlx-lm でのベースライン計測。

uv run python bench/baseline.py --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit
"""

import argparse
import json
import resource
import subprocess
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

SHORT = "Explain how a hash table handles collisions."
# prefill 計測用。同じ段落を繰り返して約 1K トークンにする。
LONG_UNIT = (
    "A mixture-of-experts layer routes each token to a small subset of "
    "feed-forward experts chosen by a learned gate, so only a fraction of "
    "the parameters is touched per token. "
)


def swap_used_mb():
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
    ).stdout
    return float(out.split("used = ")[1].split("M")[0])


def run(model, tokenizer, prompt, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    last = None
    for last in stream_generate(model, tokenizer, ids, max_tokens=max_tokens):
        pass
    return {
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": round(last.prompt_tps, 2),
        "generation_tokens": last.generation_tokens,
        "generation_tps": round(last.generation_tps, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--long-repeats", type=int, default=28)
    args = ap.parse_args()

    swap0 = swap_used_mb()
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0

    res = {
        "model": args.model,
        "mlx": mx.__version__,
        "load_s": round(load_s, 2),
        "active_gb_after_load": round(mx.get_active_memory() / 1e9, 3),
    }
    # 1 回目はウォームアップ(カーネルのコンパイルとページイン)
    res["warmup"] = run(model, tokenizer, SHORT, 16)
    res["decode"] = run(model, tokenizer, SHORT, args.max_tokens)
    res["prefill"] = run(model, tokenizer, LONG_UNIT * args.long_repeats, 16)
    res["mlx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
    res["max_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 3
    )
    res["swap_used_mb_before"] = swap0
    res["swap_used_mb_after"] = swap_used_mb()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
