"""MoE のルーティングを記録する(Phase 1)。

uv run python tools/route_log.py --model MODEL \
    --prompts analysis/workloads/code.jsonl --out analysis/logs/olmoe_code.npz

出力 npz(T = 全プロンプトを連結した処理順のトークン数):
  topk   int16   [T, L, k]      実際に使われたエキスパート ID(SwitchGLU に渡った値)
  logits float16 [T, L, E]      ルーターの生ロジット
  pred   int16   [D, T, L, 2k]  層 l のブロック入力に層 l+d のルーターを仮適用した上位 2k(降順)。範囲外は -1
  seq_id int32 [T] / is_gen bool [T] / token_ids int32 [T]
  num_experts, pred_offsets
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwitchGLU

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoe.layer import StreamingSwitchGLU  # noqa: E402

PRED_OFFSETS = (0, 1, 2)


class RecordingGate(nn.Module):
    """ルーターの Linear を包み、出力ロジットを控える。"""

    def __init__(self, inner, sink):
        super().__init__()
        self.inner = inner
        self._sink = sink

    def __call__(self, x):
        y = self.inner(x)
        self._sink.append(y.reshape(-1, y.shape[-1]))
        return y


class Recorder:
    def __init__(self, model):
        self.layers = model.layers
        self.L = len(self.layers)
        # model_type ごとの差はここで吸収する。olmoe / qwen3_moe は同じ構造。
        for layer in self.layers:
            assert isinstance(layer.mlp.switch_mlp, (SwitchGLU, StreamingSwitchGLU)), type(layer.mlp)
        self.E = self.layers[0].mlp.num_experts
        self.k = self.layers[0].mlp.top_k
        self.reset_step()
        self.out = {"topk": [], "logits": [], "pred": []}
        self._install()

    def reset_step(self):
        self.step_topk = []
        self.step_logits = []
        self.step_pred = [[] for _ in PRED_OFFSETS]

    def _router(self, target, x):
        layer = self.layers[target]
        gate = layer.mlp.gate
        return gate.inner(layer.post_attention_layernorm(x))

    def _install(self):
        rec = self
        block_cls = type(self.layers[0])
        orig_block = block_cls.__call__
        glu_cls = type(self.layers[0].mlp.switch_mlp)
        orig_glu = glu_cls.__call__

        def block_call(self, x, *args, **kwargs):
            i = self._route_idx
            n2k = 2 * rec.k
            for j, d in enumerate(PRED_OFFSETS):
                if i + d < rec.L:
                    logits = rec._router(i + d, x).reshape(-1, rec.E)
                    p = mx.argsort(-logits, axis=-1)[:, :n2k]
                else:
                    p = mx.full((x.shape[0] * x.shape[1], n2k), -1)
                rec.step_pred[j].append(p)
            return orig_block(self, x, *args, **kwargs)

        def glu_call(self, x, indices):
            rec.step_topk.append(indices.reshape(-1, indices.shape[-1]))
            return orig_glu(self, x, indices)

        block_cls.__call__ = block_call
        glu_cls.__call__ = glu_call
        for i, layer in enumerate(self.layers):
            layer._route_idx = i
            layer.mlp.gate = RecordingGate(layer.mlp.gate, self.step_logits)

    def flush(self, *extra):
        """1 回の forward 分を評価して numpy に移す。"""
        topk = mx.stack(self.step_topk, axis=1)
        logits = mx.stack(self.step_logits, axis=1)
        pred = mx.stack([mx.stack(p, axis=1) for p in self.step_pred], axis=0)
        mx.eval(topk, logits, pred, *extra)
        self.out["topk"].append(np.array(topk).astype(np.int16))
        self.out["logits"].append(np.array(logits.astype(mx.float32)).astype(np.float16))
        self.out["pred"].append(np.array(pred).astype(np.int16))
        n = topk.shape[0]
        self.step_topk.clear()
        self.step_logits.clear()
        for p in self.step_pred:
            p.clear()
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoe", action="store_true", help="--model を repack.py の出力として読む")
    ap.add_argument("--cache-gb", type=float, default=2.0)
    args = ap.parse_args()

    if args.smoe:
        from smoe import load_streaming

        model, tokenizer, _ = load_streaming(args.model, cache_gb=args.cache_gb)
    else:
        model, tokenizer = load(args.model)
    rec = Recorder(model)
    prompts = [json.loads(s) for s in Path(args.prompts).read_text().splitlines() if s.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    seq_id, is_gen, token_ids = [], [], []
    eos = set(tokenizer.eos_token_ids)
    t0 = time.perf_counter()
    for si, item in enumerate(prompts):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}], add_generation_prompt=True
        )
        cache = make_prompt_cache(model)
        logits = model(mx.array(ids)[None], cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        n = rec.flush(tok)
        seq_id += [si] * n
        is_gen += [False] * n
        token_ids += list(ids)
        for _ in range(args.max_tokens):
            t = tok.item()
            if t in eos:
                break
            logits = model(tok[None], cache=cache)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            rec.flush(tok)
            seq_id.append(si)
            is_gen.append(True)
            token_ids.append(t)
        mx.clear_cache()
        print(f"{item.get('id', si)}: total {len(seq_id)} tok, {time.perf_counter() - t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        topk=np.concatenate(rec.out["topk"]),
        logits=np.concatenate(rec.out["logits"]),
        pred=np.concatenate(rec.out["pred"], axis=1),
        seq_id=np.array(seq_id, dtype=np.int32),
        is_gen=np.array(is_gen, dtype=bool),
        token_ids=np.array(token_ids, dtype=np.int32),
        num_experts=rec.E,
        pred_offsets=np.array(PRED_OFFSETS),
    )
    print(f"saved {args.out}: T={len(seq_id)} L={rec.L} E={rec.E} k={rec.k}")


if __name__ == "__main__":
    main()
