"""mlx-lm の SwitchGLU を、エキスパートを SSD から引く版に差し替える。"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
from mlx_lm.utils import load_model, load_tokenizer

from .cache import ExpertCache
from .loader import ExpertStore

PROJS = ("up_proj", "gate_proj", "down_proj")


class _Next:
    """次の層のルーター(先読み用)。nn.Module の属性にすると引数扱いになるので別オブジェクトに持つ。"""

    def __init__(self, norm, gate, prefix, m, early):
        self.norm, self.gate, self.prefix, self.m, self.early = norm, gate, prefix, m, early
        self.pred = None  # この層のブロック入力から作った次の層の予測(遅延評価のまま持つ)
        self.far = None  # 2 層先の予測(同じ形の _Next。m は別に指定する)
        self.prob = None  # 予測の各エキスパートのルーター確率(pred と同じ並び、降順)
        self.min_prob = 0.0

    def preds(self):
        out = []
        for n in (self, self.far):
            if n is not None and n.pred is not None:
                out += [n.pred, n.prob]
        return out

    def chosen(self):
        """確信度の高い順に、しきい値以上のものだけ返す。"""
        ids = np.array(self.pred).tolist()
        if self.min_prob <= 0:
            return ids
        probs = np.array(self.prob.astype(mx.float32)).tolist()
        return [e for e, p in zip(ids, probs) if p >= self.min_prob]

    def clear(self):
        self.pred = self.prob = None
        if self.far is not None:
            self.far.pred = self.far.prob = None


class StreamingSwitchGLU(nn.Module):
    def __init__(self, prefix, cache, activation, input_dims, mode):
        super().__init__()
        self.prefix = prefix
        self.activation = activation
        self._cache = cache
        self._mode = mode
        parts = {(p["proj"], p["kind"]): p for p in cache.store.layers[prefix]["parts"]}
        assert ("up_proj", "bias") not in parts, "bias つきのエキスパートは未対応"
        w, s = parts[("up_proj", "weight")], parts[("up_proj", "scales")]
        self._group_size = input_dims // s["shape"][-1]
        self._bits = w["shape"][-1] * 32 // input_dims
        self._has_biases = ("up_proj", "biases") in parts
        self._per_expert = True
        self._next = None
        self._async = True
        self._chunk = 32
        self._prefill_admit = "tail"  # "all" / "none" / "tail"
        self._tail_tokens = 16

    def _qmm(self, x, arrays, proj, indices, sorted_indices):
        return mx.gather_qmm(
            x,
            arrays[(proj, "weight")],
            arrays[(proj, "scales")],
            arrays[(proj, "biases")] if self._has_biases else None,
            rhs_indices=indices,
            transpose=True,
            group_size=self._group_size,
            bits=self._bits,
            mode=self._mode,
            sorted_indices=sorted_indices,
        )

    def _single_token(self, x, idx_np, by_id):
        """1 トークンのときは積み直さず、エキスパートごとに直接掛ける(コピーなし)。"""
        x = x.reshape(1, -1)
        kw = dict(transpose=True, group_size=self._group_size, bits=self._bits, mode=self._mode)

        def qmm(v, a, proj):
            biases = a[(proj, "biases")] if self._has_biases else None
            return mx.quantized_matmul(v, a[(proj, "weight")], a[(proj, "scales")], biases, **kw)

        ys = []
        for e in idx_np.reshape(-1).tolist():
            a = by_id[e]
            h = self.activation(qmm(x, a, "up_proj"), qmm(x, a, "gate_proj"))
            ys.append(qmm(h, a, "down_proj"))
        out = mx.stack(ys, axis=-2).reshape(*idx_np.shape, -1)
        if self._async:
            # 次の層の attention を CPU で組み立てている間に、GPU にこの層の計算を進めさせる
            mx.async_eval(out)
        return out

    def _prefetch_next(self, idx_np):
        """1 トークンのとき、この層の入力に次の層のルーターを当てて上位 m 個を先に読み始める。"""
        nxt = self._next
        if nxt.pred is None or idx_np.size != idx_np.shape[-1]:
            return
        # 急ぐ 1 層先を先に、2 層先を後に。それぞれ確信度の高い順に発行する
        self._cache.prefetch(nxt.prefix, nxt.chosen())
        if nxt.far is not None and nxt.far.pred is not None:
            self._cache.prefetch(nxt.far.prefix, nxt.far.chosen())

    def _chunked(self, x, idx_np, uniq):
        """エキスパートを番号順に self._chunk 個ずつ区切り、読む → 積む → 計算 → 評価 を繰り返す。

        (トークン, エキスパート) の組を番号順に並べると、区切りごとに連続した範囲になるので、
        mlx-lm と同じ sorted 版の gather_qmm で計算できる。最後に元の並びへ戻す。
        """
        k = idx_np.shape[-1]
        flat = idx_np.reshape(-1)
        order = np.argsort(flat, kind="stable")
        inv_order = mx.array(np.argsort(order).astype(np.uint32))
        sorted_e = flat[order]
        x3 = x.reshape(-1, 1, x.shape[-1])
        rows = order // k
        # prefill で読んだ分をどこまでキャッシュに入れるか
        if self._prefill_admit == "all":
            admit = None
        elif self._prefill_admit == "none":
            admit = set()
        else:  # "tail": プロンプト末尾のトークンが使ったものだけ(生成の頭で使われやすい)
            admit = set(np.unique(idx_np.reshape(-1, k)[-self._tail_tokens:]).tolist())
        ys = []
        for c in range(0, len(uniq), self._chunk):
            chunk = uniq[c : c + self._chunk]
            lo = np.searchsorted(sorted_e, chunk[0], side="left")
            hi = np.searchsorted(sorted_e, chunk[-1], side="right")
            fetched = self._cache.get_many(self.prefix, chunk.tolist(), admit=admit)
            arrays = {key: mx.stack([f[key] for f in fetched]) for key in fetched[0]}
            local = mx.array(np.searchsorted(chunk, sorted_e[lo:hi]).astype(np.uint32))
            xs = x3[mx.array(rows[lo:hi].astype(np.uint32))]
            up = self._qmm(xs, arrays, "up_proj", local, True)
            gate = self._qmm(xs, arrays, "gate_proj", local, True)
            y = self._qmm(self.activation(up, gate), arrays, "down_proj", local, True)
            mx.eval(y)  # 積んだ配列をここで手放す
            ys.append(y)
        out = mx.concatenate(ys, axis=0)[inv_order]
        return out.reshape(*idx_np.shape, -1)

    def __call__(self, x, indices):
        # ルーティング結果を確定させないと、どのエキスパートを読むか決まらない。
        # 次の層の予測も同じ 1 回の評価で確定させる(同期を増やさない)。
        if self._next is not None and self._next.pred is not None:
            mx.eval(indices, *self._next.preds())
        idx_np = np.array(indices)
        uniq, inv = np.unique(idx_np, return_inverse=True)
        if len(uniq) > self._chunk:
            # prefill など、使うエキスパートが多いときは分けて読み・計算する(一時メモリを抑える)
            if self._next is not None:
                self._next.clear()
            return self._chunked(x, idx_np, uniq)
        nxt = self._next
        early = nxt is not None and nxt.early
        if early:
            self._prefetch_next(idx_np)
        fetched = self._cache.get_many(self.prefix, uniq.tolist())
        if nxt is not None and not early:
            self._prefetch_next(idx_np)
        if nxt is not None:
            nxt.clear()
        if idx_np.size == idx_np.shape[-1] and self._per_expert:
            return self._single_token(x, idx_np, dict(zip(uniq.tolist(), fetched)))
        # 使う分だけを (U, out, in) に積み、ID を 0..U-1 に付け替える
        arrays = {key: mx.stack([f[key] for f in fetched]) for key in fetched[0]}
        indices = mx.array(inv.reshape(idx_np.shape).astype(np.uint32))

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self._qmm(x, arrays, "up_proj", idx, do_sort)
        x_gate = self._qmm(x, arrays, "gate_proj", idx, do_sort)
        x = self._qmm(self.activation(x_up, x_gate), arrays, "down_proj", idx, do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        out = x.squeeze(-2)
        if self._async:
            mx.async_eval(out)
        return out


def _set_by_path(root, path, value):
    obj = root
    keys = path.split(".")
    for k in keys[:-1]:
        obj = obj[int(k)] if k.isdigit() else getattr(obj, k)
    setattr(obj, keys[-1], value)


def _install_prefetch(model, prefetch_m, early, prefetch_m2=0, min_prob=0.0):
    layers = model.layers
    glus = [layer.mlp.switch_mlp for layer in layers]
    for i, glu in enumerate(glus[:-1]):
        nxt = layers[i + 1]
        glu._next = _Next(nxt.post_attention_layernorm, nxt.mlp.gate, glus[i + 1].prefix, prefetch_m, early)
        glu._next.min_prob = min_prob
        if prefetch_m2 and i + 2 < len(layers):
            far = layers[i + 2]
            glu._next.far = _Next(far.post_attention_layernorm, far.mlp.gate, glus[i + 2].prefix, prefetch_m2, early)
            glu._next.far.min_prob = min_prob
    block_cls = type(layers[0])
    if getattr(block_cls, "_smoe_patched", False):
        return
    orig = block_cls.__call__

    def call(self, x, *args, **kwargs):
        glu = self.mlp.switch_mlp
        nxt = glu._next if isinstance(glu, StreamingSwitchGLU) else None
        if nxt is not None and x.shape[0] * x.shape[1] == 1:
            # 1 トークンのときだけ、次の層のルーターをこの層の入力に当てた上位 m 個を積んでおく
            for n in (nxt, nxt.far):
                if n is not None:
                    probs = mx.softmax(n.gate(n.norm(x)).reshape(-1).astype(mx.float32))
                    n.pred = mx.argsort(-probs)[: n.m]  # 確信度の高い順
                    n.prob = probs[n.pred]
        return orig(self, x, *args, **kwargs)

    block_cls.__call__ = call
    block_cls._smoe_patched = True


def load_streaming(path, cache_gb=1.0, io_mode="pread", io_threads=4, prefetch_m=0, prefetch_early=False, prefetch_m2=0,
                   pf_threads=1, prefetch_min_prob=0.0,
                   keep_unused_prefetch=False, per_layer=False, prefill_admit="tail", tail_tokens=16):
    """repack.py の出力ディレクトリからロードする。返り値は (model, tokenizer, cache)。"""
    path = Path(path)
    # 追い出したエキスパートの領域を MLX が再利用用に抱え込むと、
    # キャッシュ容量とは別にメモリが膨らむ。小さく抑える。
    mx.set_cache_limit(256 << 20)
    model, config = load_model(path, lazy=True, strict=False)
    store = ExpertStore(path, mode=io_mode, io_threads=io_threads, pf_threads=pf_threads)
    cache = ExpertCache(store, int(cache_gb * 1e9), keep_unused_prefetch, per_layer)
    mode = (config.get("quantization") or {}).get("mode", "affine")
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, SwitchGLU)]
    assert targets and {n for n, _ in targets} == set(store.layers), "index と層が対応しない"
    for name, m in targets:
        new = StreamingSwitchGLU(name, cache, m.activation, m.up_proj.input_dims, mode)
        new._prefill_admit, new._tail_tokens = prefill_admit, tail_tokens
        _set_by_path(model, name, new)
    if prefetch_m and io_mode == "pread":
        _install_prefetch(model, prefetch_m, prefetch_early, prefetch_m2, prefetch_min_prob)
    mx.eval(model.parameters())
    model.eval()
    tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
    return model, tokenizer, cache
