"""ストリーミング版と素の mlx-lm の出力を比べる。

ビット一致はしない。gather_qmm(sorted 経路)は、重み配列のエキスパート数によって
内部の計算順が変わり、最下位桁がずれる(1 層で 5e-6 程度)。fp16 で 16 層通すと
ロジットで 0.3 程度まで広がるので、端から端の比較は「貪欲デコードのトークン一致」で見る。
層単体は 1e-3 以内で比べる。

SMOE_REF=mlx-community/OLMoE-1B-7B-0125-Instruct-4bit SMOE_DIR=models/olmoe-4bit-smoe \
    uv run pytest tests/test_logits.py -s
"""

import gc
import os

import mlx.core as mx
import numpy as np
import pytest

REF = os.environ.get("SMOE_REF")
DIR = os.environ.get("SMOE_DIR")
PROMPT = "Write a Python function that merges two sorted lists. 日本語でも説明して。"
STEPS = 8


def run(model, tokenizer):
    """prefill 1 回 + 貪欲デコード STEPS 回。各回の最終位置のロジットを返す。"""
    from mlx_lm.models.cache import make_prompt_cache

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    cache = make_prompt_cache(model)
    x = mx.array(ids)[None]
    out = []
    for _ in range(STEPS + 1):
        logits = model(x, cache=cache)[:, -1, :]
        mx.eval(logits)
        out.append(np.array(logits.astype(mx.float32))[0])
        x = mx.argmax(logits, axis=-1)[None]
    return np.stack(out)


@pytest.mark.skipif(not (REF and DIR), reason="SMOE_REF と SMOE_DIR が必要")
@pytest.mark.parametrize("io_mode,prefetch_m,prefetch_m2", [("pread", 0, 0), ("mmap", 0, 0), ("pread", 8, 0), ("pread", 8, 8)])
def test_logits_match(io_mode, prefetch_m, prefetch_m2):
    from mlx_lm import load

    from smoe import load_streaming

    model, tokenizer = load(REF)
    ref = run(model, tokenizer)
    del model
    gc.collect()
    mx.clear_cache()

    # キャッシュを小さくして、追い出しと再読み込みも通す
    model, tokenizer, cache = load_streaming(DIR, cache_gb=0.6, io_mode=io_mode, prefetch_m=prefetch_m,
                                              prefetch_m2=prefetch_m2)
    got = run(model, tokenizer)
    stats = cache.stats()
    print(io_mode, prefetch_m, "max abs diff", np.abs(ref - got).max(), stats)

    assert (ref.argmax(-1) == got.argmax(-1)).all()
    assert np.abs(ref - got).max() < 1.0
    if prefetch_m:
        assert stats["pf_used"] > 0
    assert 0.0 < stats["hit_rate"] < 1.0 and stats["cache_gb"] <= 0.6 + 0.03


@pytest.mark.skipif(not (REF and DIR), reason="SMOE_REF と SMOE_DIR が必要")
def test_layer_matches():
    """同じ入力に対して、層単体の出力が素の SwitchGLU と一致すること(prefill 経路と 1 トークン経路)。"""
    from mlx_lm import load

    from smoe import load_streaming

    model, _ = load(REF)
    smodel, _, _ = load_streaming(DIR, cache_gb=0.6)
    rng = np.random.default_rng(0)
    for li in (0, 7, 15):
        ref_glu = model.model.layers[li].mlp.switch_mlp
        glu = smodel.model.layers[li].mlp.switch_mlp
        for n in (1, 24):
            idx = np.stack([rng.choice(64, 8, replace=False) for _ in range(n)]).astype(np.uint32)
            x = mx.array(rng.standard_normal((n, 2048)).astype(np.float16))
            a, b = ref_glu(x, mx.array(idx)), glu(x, mx.array(idx))
            assert a.shape == b.shape
            assert float(mx.abs(a - b).max()) < 1e-3, (li, n)
