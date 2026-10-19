"""safetensors をエキスパート単位のブロブへ再配置する(HANDOVER アイデア 1)。

uv run python tools/repack.py SRC_DIR OUT_DIR [--delete-source]
uv run python tools/repack.py --repo mlx-community/Qwen3-30B-A3B-4bit OUT_DIR

出力:
  experts.bin           (層, エキスパート) ごとの連続ブロブ。各ブロブは 16KB 境界で始まる
  index.json            ブロブの配置
  model.safetensors     非エキスパート部だけ(mlx-lm が strict=False でそのまま読める)
  config.json ほか      トークナイザ等は SRC_DIR からコピー

配置は全シャードのヘッダだけで決まるので、シャードは 1 本ずつ処理できる。
--delete-source を付けると、処理し終えたシャードをその場で消す(ディスクが足りないとき用)。
--repo は HF からヘッダだけを Range 取得して配置を決め、シャードを 1 本ずつ
ダウンロード → 写す → 消す。ディスクのピークは「出力 + シャード 1 本」で済む。
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import struct
from pathlib import Path

ALIGN = 16384
CHUNK = 8 << 20
PROJ_ORDER = ("gate_proj", "up_proj", "down_proj")
KIND_ORDER = ("weight", "scales", "biases", "bias")

# エキスパートが 1 個ずつ別テンソルの形式(OLMoE の変換など)
PER_EXPERT = re.compile(r"^(.*)\.experts\.(\d+)\.(\w+)\.(weight|scales|biases|bias)$")
# 画像入力など、テキスト生成に使わない部分(mlx-lm も読み込み時に捨てる)
SKIP_PREFIXES = ("vision_tower.", "model.visual.")
# (E, out, in) に積まれた形式
STACKED = re.compile(r"^(.*\.switch_mlp)\.(\w+)\.(weight|scales|biases|bias)$")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, 8 + n, meta


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def fetch_header(repo, filename):
    """HF 上の safetensors のヘッダだけを HTTP Range で取る。"""
    from huggingface_hub import get_session, hf_hub_url

    url = hf_hub_url(repo, filename)
    session = get_session()
    head = session.get(url, headers={"Range": "bytes=0-7"}, follow_redirects=True)
    n = struct.unpack("<Q", head.content[:8])[0]
    body = session.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}, follow_redirects=True)
    header = json.loads(body.content)
    header.pop("__metadata__", None)
    return header, 8 + n


def plan(shards, headers=None):
    """全シャードのヘッダから、ブロブの配置と各テンソルの写し先を決める。"""
    groups = {}  # prefix -> {(proj, kind): {"dtype", "shape", "nbytes", "src": {e: (shard, off)}}}
    resident = {}  # name -> (shard, off, nbytes, dtype, shape)
    for path in shards:
        header, base = headers[path] if headers else read_header(path)[:2]
        for name, t in header.items():
            if name.startswith(SKIP_PREFIXES):
                continue
            off, end = t["data_offsets"]
            off += base
            nbytes = end - t["data_offsets"][0]
            m = PER_EXPERT.match(name)
            if m:
                prefix, e, proj, kind = m[1] + ".switch_mlp", int(m[2]), m[3], m[4]
                part = groups.setdefault(prefix, {}).setdefault(
                    (proj, kind),
                    {"dtype": t["dtype"], "shape": t["shape"], "nbytes": nbytes, "src": {}},
                )
                assert part["shape"] == t["shape"], name
                part["src"][e] = (path, off)
                continue
            m = STACKED.match(name)
            if m and len(t["shape"]) == 3:
                E = t["shape"][0]
                per = nbytes // E
                groups.setdefault(m[1], {})[(m[2], m[3])] = {
                    "dtype": t["dtype"],
                    "shape": t["shape"][1:],
                    "nbytes": per,
                    "src": {e: (path, off + e * per) for e in range(E)},
                }
                continue
            resident[name] = (path, off, nbytes, t["dtype"], t["shape"])

    def part_key(pk):
        return (PROJ_ORDER.index(pk[0]), KIND_ORDER.index(pk[1]))

    def layer_key(prefix):
        return [int(s) for s in re.findall(r"\d+", prefix)]

    index = {"align": ALIGN, "layers": {}}
    copies = {}  # shard -> [(src_off, nbytes, dst_off)]
    cursor = 0
    for prefix in sorted(groups, key=layer_key):
        g = groups[prefix]
        keys = sorted(g, key=part_key)
        E = len(g[keys[0]]["src"])
        parts, rel = [], 0
        for pk in keys:
            p = g[pk]
            assert len(p["src"]) == E, (prefix, pk)
            parts.append(
                {"proj": pk[0], "kind": pk[1], "dtype": p["dtype"], "shape": p["shape"],
                 "offset": rel, "nbytes": p["nbytes"]}
            )
            rel += p["nbytes"]
        stride = align_up(rel)
        for part, pk in zip(parts, keys):
            for e, (path, off) in g[pk]["src"].items():
                copies.setdefault(path, []).append(
                    (off, part["nbytes"], cursor + e * stride + part["offset"])
                )
        index["layers"][prefix] = {
            "num_experts": E, "offset": cursor, "stride": stride, "blob_nbytes": rel, "parts": parts,
        }
        cursor += E * stride
    index["total_nbytes"] = cursor
    return index, copies, resident


def preallocate(fd, nbytes):
    """書き込み順に関係なく、ファイルを連続した領域に置く。

    pwrite で飛び飛びに埋めると APFS 上でブロブが細かく断片化し、
    1 ブロブの読み出しが小さなランダム読みの連続になる(実測で約半分の速度)。
    """
    F_PREALLOCATE, F_ALLOCATECONTIG, F_ALLOCATEALL, F_PEOFPOSMODE = 42, 2, 4, 3
    for flags in (F_ALLOCATECONTIG | F_ALLOCATEALL, F_ALLOCATEALL):
        fstore = struct.pack("IiqqQ", flags, F_PEOFPOSMODE, 0, nbytes, 0)
        try:
            fcntl.fcntl(fd, F_PREALLOCATE, fstore)
            return flags & F_ALLOCATECONTIG != 0
        except OSError:
            continue
    return False


def copy_range(src_fd, src_off, nbytes, dst_fd, dst_off):
    done = 0
    while done < nbytes:
        buf = os.pread(src_fd, min(CHUNK, nbytes - done), src_off + done)
        os.pwrite(dst_fd, buf, dst_off + done)
        done += len(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--repo")
    ap.add_argument("--delete-source", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.repo:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(args.repo)
        src = out / "_download"
        for f in files:
            if not f.endswith(".safetensors") and not f.startswith("."):
                hf_hub_download(args.repo, f, local_dir=src)
        shards = sorted(f for f in files if f.endswith(".safetensors"))
        headers = {f: fetch_header(args.repo, f) for f in shards}
        index, copies, resident = plan(shards, headers)
        args.delete_source = True
    else:
        src = Path(args.src)
        shards = sorted(str(p) for p in src.glob("*.safetensors"))
        index, copies, resident = plan(shards)

    # 非エキスパート部のヘッダを先に確定する
    rheader, roff = {"__metadata__": {"format": "mlx"}}, 0
    rcopies = {}
    for name in sorted(resident):
        path, off, nbytes, dtype, shape = resident[name]
        rheader[name] = {"dtype": dtype, "shape": shape, "data_offsets": [roff, roff + nbytes]}
        rcopies.setdefault(path, []).append((off, nbytes, roff))
        roff += nbytes
    hjson = json.dumps(rheader, separators=(",", ":")).encode()
    hjson += b" " * (align_up(8 + len(hjson), 8) - 8 - len(hjson))
    rbase = 8 + len(hjson)

    efd = os.open(out / "experts.bin", os.O_RDWR | os.O_CREAT, 0o644)
    contig = preallocate(efd, index["total_nbytes"])
    print(f"preallocate contiguous={contig}", flush=True)
    os.ftruncate(efd, index["total_nbytes"])
    rfd = os.open(out / "model.safetensors", os.O_RDWR | os.O_CREAT, 0o644)
    os.pwrite(rfd, struct.pack("<Q", len(hjson)) + hjson, 0)
    for path in shards:
        if args.repo:
            print(f"download {path}", flush=True)
            local = hf_hub_download(args.repo, path, local_dir=src)
        else:
            local = path
        sfd = os.open(local, os.O_RDONLY)
        for off, nbytes, dst in copies.get(path, []):
            copy_range(sfd, off, nbytes, efd, dst)
        for off, nbytes, dst in rcopies.get(path, []):
            copy_range(sfd, off, nbytes, rfd, rbase + dst)
        os.close(sfd)
        print(f"done {Path(path).name}", flush=True)
        if args.delete_source:
            os.remove(os.path.realpath(local))
    os.close(efd)
    os.close(rfd)

    (out / "index.json").write_text(json.dumps(index))
    for p in src.iterdir():
        if p.suffix != ".safetensors" and p.is_file() and p.name != "model.safetensors.index.json":
            shutil.copy(p, out / p.name)
    if args.repo:
        shutil.rmtree(src)
    n = sum(v["num_experts"] for v in index["layers"].values())
    print(
        f"experts.bin {index['total_nbytes'] / 1e9:.2f}GB ({n} blobs), "
        f"resident {(rbase + roff) / 1e9:.2f}GB"
    )


if __name__ == "__main__":
    main()
