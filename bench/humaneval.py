"""HumanEval 評価ハーネス。

  uv run python bench/humaneval.py --self-test --n 20
  uv run python bench/humaneval.py --model models/X --smoe --n 20 --out bench/logs/humaneval_X.jsonl [--resume]

pyarrow が必要(uv add --optional bench pyarrow 済み)。--self-test ではモデルを一切ロードしない。
"""
import argparse
import json
import re
import subprocess
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import time
from pathlib import Path

REPO = "openai/openai_humaneval"
PARQUET = "openai_humaneval/test-00000-of-00001.parquet"
INSTR = ("Complete the following Python function. Return the complete function (including the "
         "signature and any needed imports) in a single ```python code block. "
         "Do not include tests or explanations.\n\n")


def load_problems():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=PARQUET)
    rows = pq.read_table(p).to_pylist()
    rows.sort(key=lambda r: int(r["task_id"].split("/")[1]))
    return rows


def select(rows, n, ids=None):
    if ids:
        by = {r["task_id"]: r for r in rows}
        return [by[i if "/" in i else f"HumanEval/{i}"] for i in ids]
    n = min(n, len(rows))
    return [rows[i * len(rows) // n] for i in range(n)]


def extract_code(text):
    m = re.search(r"```python[^\n]*\n(.*?)(```|\Z)", text, re.S)
    return m.group(1) if m else text


def build_program(prob, completion):
    code = extract_code(completion)
    if not re.search(rf"def\s+{re.escape(prob['entry_point'])}\s*\(", code):
        code = prob["prompt"] + code
    return code + "\n\n" + prob["test"] + f"\n\ncheck({prob['entry_point']})\n"


def grade(prob, completion, timeout=10):
    prog = build_program(prob, completion)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(prog)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], timeout=timeout, cwd=d, capture_output=True)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        err = r.stderr.decode(errors="replace")
        return r.returncode == 0, ("" if r.returncode == 0 else err[-300:] if err else f"rc={r.returncode}")


def self_test(probs):
    ok = bad = 0
    for p in probs:
        good = p["prompt"] + p["canonical_solution"]
        ok += grade(p, "```python\n" + good + "\n```")[0]
        broken = p["prompt"] + "    pass\n"
        bad += grade(p, "```python\n" + broken + "\n```")[0]
    n = len(probs)
    print(json.dumps({"self_test": True, "n": n, "canonical_passed": ok, "broken_passed": bad}))
    return ok == n and bad == 0


def make_prompt(tok, prob):
    msgs = [{"role": "user", "content": INSTR + prob["prompt"]}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True)


def closed(text):
    i = text.find("```python")
    return i >= 0 and "```" in text[i + 9:]


def generate(model, tok, prob, max_tokens):
    from mlx_lm import stream_generate
    ids = make_prompt(tok, prob)
    if isinstance(ids, str):
        ids = tok.encode(ids, add_special_tokens=False)
    text, n = "", 0
    t0 = time.perf_counter()
    r = None
    for r in stream_generate(model, tok, ids, max_tokens=max_tokens):
        text += r.text
        n += 1
        if closed(text):
            break
    dt = time.perf_counter() - t0
    # dt は prefill を含む全体。生成だけの速度は stream_generate の値を残す
    extra = {"prompt_tokens": r.prompt_tokens, "prompt_tps": round(r.prompt_tps, 3),
             "generation_tps": round(r.generation_tps, 3)} if r else {}
    return text, n, dt, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--smoe", action="store_true")
    ap.add_argument("--cache-gb", type=float, default=2.45)
    ap.add_argument("--prefetch-m", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    probs = select(load_problems(), args.n, args.ids)
    if args.self_test:
        print("ids:", " ".join(p["task_id"].split("/")[1] for p in probs), file=sys.stderr)
        sys.exit(0 if self_test(probs) else 1)

    if not args.model:
        ap.error("--model required")
    name = Path(args.model.rstrip("/")).name
    out = Path(args.out or f"bench/logs/humaneval_{name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        if args.resume:
            done = {json.loads(l)["task_id"]: json.loads(l) for l in out.read_text().splitlines() if l.strip()}
        else:
            out.write_text("")
    todo = [p for p in probs if p["task_id"] not in done]

    model = None
    if todo:
        if args.smoe:
            from smoe import load_streaming
            model, tok, cache = load_streaming(args.model, cache_gb=args.cache_gb, prefetch_m=args.prefetch_m)
        else:
            from mlx_lm import load
            model, tok = load(args.model)
    for p in todo:
        text, n, dt, extra = generate(model, tok, p, args.max_tokens)
        passed, err = grade(p, text)
        rec = {"task_id": p["task_id"], "passed": passed, "gen_tokens": n, "gen_s": round(dt, 3),
               "tps": round(n / dt, 3) if dt else 0.0, "error": err[:300], "completion": text, **extra}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[p["task_id"]] = rec
        print(f"{p['task_id']} {'PASS' if passed else 'FAIL'} {n}tok {rec['tps']:.2f}tps", file=sys.stderr)

    recs = [done[p["task_id"]] for p in probs]
    tps = [r["tps"] for r in recs]
    print(json.dumps({"model": name, "n": len(recs),
                      "pass_rate": round(sum(r["passed"] for r in recs) / len(recs), 4),
                      "passed": sum(r["passed"] for r in recs),
                      "total_gen_tokens": sum(r["gen_tokens"] for r in recs),
                      "mean_tps": round(sum(tps) / len(tps), 3)}))


if __name__ == "__main__":
    main()
