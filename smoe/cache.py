"""(層, エキスパート) 単位の LRU キャッシュ。容量はバイト数で指定する。"""

import time
from collections import OrderedDict


class ExpertCache:
    def __init__(self, store, capacity_bytes, keep_unused_prefetch=False, per_layer=False):
        self.store = store
        self.capacity = capacity_bytes
        self.used = 0
        self.items = OrderedDict()  # (prefix, e) -> {(proj, kind): mx.array}
        self.hits = 0
        self.misses = 0
        self.inflight = {}  # (prefix, e) -> 先読みの future
        self.pf_issued = 0
        self.pf_used = 0
        self.pf_cancelled = 0
        # 外れた先読みのうち読み終わっているものを、捨てずにキャッシュへ入れる
        self.keep_unused = keep_unused_prefetch
        self.per_layer = per_layer
        self.layer_used = {}

    def prefetch(self, prefix, experts):
        for e in experts:
            key = (prefix, e)
            if key not in self.items and key not in self.inflight:
                self.inflight[key] = self.store.submit(prefix, e)
                self.pf_issued += 1

    def _insert(self, prefix, e, arrays):
        size = self.store.blob_nbytes(prefix)
        self.items[(prefix, e)] = arrays
        self.used += size
        self.layer_used[prefix] = self.layer_used.get(prefix, 0) + size

    def _evict(self, key):
        del self.items[key]
        size = self.store.blob_nbytes(key[0])
        self.used -= size
        self.layer_used[key[0]] -= size

    def get_many(self, prefix, experts, admit=None):
        """admit: None なら読んだものを全部キャッシュに入れる。集合を渡すと、その中のものだけ入れ、
        既存の中身の順位も動かさない(prefill のような 1 回限りの利用でキャッシュを流さないため)。"""
        if admit is not None:
            return self._get_bypass(prefix, experts, admit)
        missing = [e for e in experts if (prefix, e) not in self.items]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        # この層の先読みのうち、使うものだけ受け取る。
        pending = {e: f for (p, e), f in self.inflight.items() if p == prefix}
        for e in pending:
            del self.inflight[(prefix, e)]
        need = set(missing)
        for e in list(pending):
            # 読み始める前のものは取り消す。不要なら捨て、必要なら通常の読み出しに回す(先読みの列に並ばせない)
            if pending[e][1].cancel():
                self.pf_cancelled += 1
                del pending[e]
            elif e not in need and not self.keep_unused:
                del pending[e]  # 読み始めてしまった外れ。結果は捨てる
        demand = [e for e in missing if e not in pending]
        if demand:
            for e, arrays in zip(demand, self.store.read_many(prefix, demand)):
                self._insert(prefix, e, arrays)
        for e in missing:
            if e in pending:
                self._insert(prefix, e, self.store.finish(prefix, pending[e]))
                self.pf_used += 1
        if self.keep_unused:
            for e, f in pending.items():
                if e not in missing and f[1].done() and (prefix, e) not in self.items:
                    self._insert(prefix, e, self.store.finish(prefix, f))
        out = []
        for e in experts:
            self.items.move_to_end((prefix, e))
            out.append(self.items[(prefix, e)])
        # いま使う分を取り出してから追い出す(同じ集合の中で互いを追い出さない)
        if self.per_layer:
            # 層ごとに容量を等分する。全体 LRU より、シミュレーションで 1〜6 ポイント高い
            cap = self.capacity / len(self.store.layers)
            using = set(experts)
            while self.layer_used[prefix] > cap:
                victim = next((k for k in self.items if k[0] == prefix and k[1] not in using), None)
                if victim is None:
                    break
                self._evict(victim)
        # 総量は必ず容量内に戻す(prefill で 1 層が全エキスパートを使うと、層ごとの上限まで減らせない)
        while self.used > self.capacity and len(self.items) > len(experts):
            self._evict(next(iter(self.items)))
        return out

    def _get_bypass(self, prefix, experts, admit):
        missing = [e for e in experts if (prefix, e) not in self.items]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        got = dict(zip(missing, self.store.read_many(prefix, missing))) if missing else {}
        out = []
        for e in experts:
            out.append(self.items[(prefix, e)] if e not in got else got[e])
        for e in missing:
            if e in admit:
                self._insert(prefix, e, got[e])
        while self.used > self.capacity and len(self.items) > len(experts):
            self._evict(next(iter(self.items)))
        return out

    def stats(self):
        n = self.hits + self.misses
        return {
            "hit_rate": round(self.hits / n, 4) if n else 0.0,
            "accesses": n,
            "read_gb": round((self.store.bytes_read + getattr(self.store, "pf_bytes", 0)) / 1e9, 3),
            "read_s": round(self.store.read_s, 2),
            "io_s": round(self.store.io_s, 2),
            "cache_gb": round(self.used / 1e9, 3),
            "pf_issued": self.pf_issued,
            "pf_used": self.pf_used,
            # 先読みで埋まったミスの割合(残りはその場で同期読み)
            "pf_cover": round(self.pf_used / self.misses, 3) if self.misses else 0.0,
            "pf_cancelled": self.pf_cancelled,
        }

    def reset_stats(self):
        self.hits = self.misses = 0
        self.pf_issued = self.pf_used = self.pf_cancelled = 0
        if hasattr(self.store, "pf_bytes"):
            self.store.pf_bytes = 0
        self.store.bytes_read = 0
        self.store.read_s = 0.0
        self.store.io_s = 0.0
