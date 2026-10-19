"""エキスパートのブロブを SSD から読む。"""

import fcntl
import json
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

F_NOCACHE = 48
# numpy に bfloat16 は無いので uint16 で読み、MLX 側で読み替える
DTYPES = {"U32": np.uint32, "F16": np.float16, "F32": np.float32, "BF16": np.uint16}
MX_DTYPES = {"U32": mx.uint32, "F16": mx.float16, "F32": mx.float32, "BF16": mx.bfloat16}


class ExpertStore:
    """mode="pread": F_NOCACHE + pread(HANDOVER 7(b))。mode="mmap": ページキャッシュ任せ(7(a))。"""

    def __init__(self, path, mode="pread", io_threads=4, pf_threads=1):
        path = Path(path)
        self.index = json.loads((path / "index.json").read_text())
        self.layers = self.index["layers"]
        self.mode = mode
        self.fd = os.open(path / "experts.bin", os.O_RDONLY)
        if mode == "pread":
            fcntl.fcntl(self.fd, F_NOCACHE, 1)
            self.pool = ThreadPoolExecutor(io_threads)
            # 先読みは専用の少数スレッドで読む。通常の読み出しが先読みの後ろに並ばず、
            # 先読みが同時に使う帯域もスレッド数で頭打ちになる。
            self.pf_pool = ThreadPoolExecutor(pf_threads)
            self.pf_bytes = 0  # 先読みで実際に読んだバイト数(取り消した分は含まない)
        else:
            self.mm = mmap.mmap(self.fd, 0, prot=mmap.PROT_READ)
        self.bytes_read = 0
        self.read_s = 0.0  # 配列化と転送を含む全体
        self.io_s = 0.0  # pread / ページインだけ

    def blob_nbytes(self, prefix):
        return self.layers[prefix]["blob_nbytes"]

    def _raw(self, prefix, e):
        info = self.layers[prefix]
        off = info["offset"] + e * info["stride"]
        return memoryview(self.mm)[off : off + info["blob_nbytes"]]

    def _to_arrays(self, prefix, raw):
        out = {}
        for p in self.layers[prefix]["parts"]:
            a = np.frombuffer(raw, DTYPES[p["dtype"]], offset=p["offset"],
                              count=p["nbytes"] // np.dtype(DTYPES[p["dtype"]]).itemsize)
            arr = mx.array(a.reshape(p["shape"]))
            if p["dtype"] == "BF16":
                arr = arr.view(mx.bfloat16)
            out[(p["proj"], p["kind"])] = arr
        return out

    def _alloc(self, prefix, count):
        """読み込み先。MLX の配列を CPU 側で確保し、そのメモリへ直接 preadv する(コピーなし)。"""
        n = self.layers[prefix]["blob_nbytes"]
        blobs = [mx.zeros((n,), dtype=mx.uint8, stream=mx.cpu) for _ in range(count)]
        mx.eval(*blobs)
        return blobs

    def _read_into(self, prefix, e, view):
        info = self.layers[prefix]
        os.preadv(self.fd, [view], info["offset"] + e * info["stride"])

    def _parts(self, prefix, blob):
        """ブロブを部品に切り分ける。スライスと view なので、メモリはブロブと共有する。"""
        out = {}
        for p in self.layers[prefix]["parts"]:
            arr = blob[p["offset"] : p["offset"] + p["nbytes"]].view(MX_DTYPES[p["dtype"]])
            out[(p["proj"], p["kind"])] = arr.reshape(p["shape"])
        return out

    def _pf_read(self, prefix, e, view):
        self._read_into(prefix, e, view)
        self.pf_bytes += len(view)

    def submit(self, prefix, e):
        """先読み用。先読み専用スレッドで読む。(ブロブ, future) を返す。
        読み始める前なら future.cancel() で取り消せる(帯域を使わない)。"""
        (blob,) = self._alloc(prefix, 1)
        return blob, self.pf_pool.submit(self._pf_read, prefix, e, memoryview(blob))

    def finish(self, prefix, pending):
        """submit の結果を待って部品に切り分ける。"""
        blob, fut = pending
        t0 = time.perf_counter()
        fut.result()
        self.io_s += time.perf_counter() - t0
        arrays = self._parts(prefix, blob)
        mx.eval(*arrays.values())
        return arrays

    def read_many(self, prefix, experts):
        """複数エキスパートをまとめて読む。返り値は {(proj, kind): mx.array} のリスト。"""
        t0 = time.perf_counter()
        out = []
        if self.mode == "pread":
            blobs = self._alloc(prefix, len(experts))
            views = [memoryview(b) for b in blobs]
            t1 = time.perf_counter()
            list(self.pool.map(lambda job: self._read_into(prefix, *job), zip(experts, views)))
            self.io_s += time.perf_counter() - t1
            out = [self._parts(prefix, b) for b in blobs]
            mx.eval(*[a for d in out for a in d.values()])
        else:
            raws = [self._raw(prefix, e) for e in experts]
            self.io_s += time.perf_counter() - t0
            out = [self._to_arrays(prefix, r) for r in raws]
            mx.eval(*[a for d in out for a in d.values()])
        self.read_s += time.perf_counter() - t0
        self.bytes_read += len(experts) * self.layers[prefix]["blob_nbytes"]
        return out
