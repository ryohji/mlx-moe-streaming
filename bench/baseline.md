# Phase 0 ベースライン

計測日 2026-10-14。Apple M1 / 8GB / macOS 27.0.1 / 内蔵 SSD 228GB(空き 14〜18GB)。
mlx 0.32.3 / mlx-lm 0.32.0 / Python 3.13。

**計測条件の注意:** VSCode ほかが起動したままで、計測前からスワップを 7.5GB 使っていた。
数値は「混んだ状態」での値。他アプリを閉じた状態での取り直しは未実施。

## ハードウェア(実測)

| 項目 | 実測 | HANDOVER の想定 |
|---|---|---|
| GPU 作業領域の上限 (`max_recommended_working_set_size`) | 5.73GB (5.33GiB) | 約 5.3GB |
| 1 バッファの上限 (`max_buffer_length`) | 4GiB | 記載なし。キャッシュを 1 配列にする場合の上限になる |
| ページサイズ | 16KB | 16KB |
| SSD 連続読み (pread + F_NOCACHE) | 最大 約 3.2GB/s(1MB×QD2〜4、4MB×QD4) | 2.5〜3GB/s |

## SSD 読み出し

再現: `python3 bench/ssd_bench.py --out bench/ssd_results.md`(全表は [ssd_results.md](ssd_results.md)、1 回だけの実行)

16KB 境界のランダム読み、pread + F_NOCACHE、MB/s:

| ブロック | QD1 | QD2 | QD4 | QD8 |
|---|---|---|---|---|
| 64KB | 394 | 789 | 925 | 1193 |
| 256KB | 1035 | 1842 | 1918 | 1899 |
| 1MB | 1938 | 2859 | 2849 | 2730 |
| 4MB | 2633 | 3264 | 3227 | 2208 |
| 8MB | 2558 | 2904 | 2367 | 1586 |

- 1MB 以上は QD2 で飽和し、4MB 以上で QD を 8 まで上げると逆に落ちる。**エキスパート読みは 1〜4MB 単位・QD2〜4 が適正。**
- 256KB 以下は 1 回あたりの固定コスト(約 0.16〜0.25ms)が支配的。小さなエキスパート(Ling-mini の約 1.5MB 未満など)は不利。
- mmap は QD1 のランダム読みで 1.0〜1.2GB/s(pread の約半分)。それ以外の mmap の値はページキャッシュに当たっており SSD の速度ではない。mmap と pread の正式な比較は Phase 2 で取り直す。

## OLMoE-1B-7B-0125-Instruct-4bit(素の mlx-lm)

再現: `uv run python bench/baseline.py --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit`

| 指標 | 値 |
|---|---|
| safetensors サイズ | 3.89GB |
| 構成 | 16 層 / 64 エキスパート / top-8 / hidden 2048 / MoE 中間 1024 / 4bit gs64 |
| ロード時間 | 12.6s |
| ロード後の MLX active メモリ | 3.89GB |
| 生成速度(22 トークンのプロンプト、128 トークン生成) | 66.4 tok/s |
| prefill 速度(1078 トークン) | 350.6 tok/s |
| MLX ピークメモリ | 4.51GB |
| スワップ使用量(前 → 後) | 7.5GB → 10.3GB |

- 初回の forward は 22 トークンに 13 秒かかった(1.65 tok/s)。カーネルのコンパイルと重みのページインによる。
- スワップが 2.8GB 増えた。4GB のモデルでも、この混み具合では他プロセスを押し出している。
- `ru_maxrss` は 0.40GB で、GPU 側のメモリを含まない。**ピークメモリの指標には使えない**ので、以降は MLX の値と `phys_footprint` を使う。

## モデル候補(HF で確認)

| モデル | repo | サイズ | 層 / エキスパート / top-k | 1 エキスパートの概算 |
|---|---|---|---|---|
| OLMoE-1B-7B | mlx-community/OLMoE-1B-7B-0125-Instruct-4bit | 3.89GB | 16 / 64 / 8 | 約 3.3MB |
| Moonlight-16B-A3B | mlx-community/Moonlight-16B-A3B-Instruct-4-bit | 8.98GB | 27 / 64+共有 2 / 6 | 約 4.6MB |
| DeepSeek-V2-Lite | mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx | 8.84GB | 27 / 64+共有 2 / 6 | 約 4.6MB |
| Ling-mini-2.0 | mlx-community/Ling-mini-2.0-4bit | 9.14GB | 20 / 256+共有 1 / 8 | 約 1.7MB |
| Qwen3-30B-A3B | mlx-community/Qwen3-30B-A3B-4bit | 17.17GB | 48 / 128 / 8 | 約 2.5MB |
| 同 3bit | mlx-community/Qwen3-30B-A3B-3bit | 13.36GB | 同上 | 約 1.9MB |

- すべて mlx-lm の `SwitchGLU`(`mx.gather_qmm`)実装で、エキスパートは層ごとに `(E, out, in)` の 1 配列。差し替えは 1 実装で共通化できる。
- 1 エキスパートの概算は「3 行列 × 4bit + scales / biases」からの机上計算。実サイズは再配置ツールで確定する。
- 16B 級の第一候補は Moonlight-16B-A3B(エキスパートが 4.6MB で SSD の得意な読み出しサイズに合う)。

## 次の判断

- Phase 1 へ進む。OLMoE のルーティング記録は動作確認済み(`tools/route_log.py`)。
- Qwen3-30B-A3B 4bit(17.17GB)は現在の空き容量に入らない。3bit 版(13.36GB)なら入るが、再配置後に元ファイルを消す逐次処理が前提。
