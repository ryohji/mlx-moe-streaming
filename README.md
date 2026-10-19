# mlx-moe-streaming

メモリ 8GB の M1 MacBook Air で、メモリに載らない大きな MoE(Mixture of Experts)モデルを MLX で動かす実験です。

- エキスパート以外の部分(attention、埋め込み、共有エキスパートなど)はメモリに常駐させます。
- エキスパートは SSD から必要な分だけ読み込み、小さなキャッシュで使い回します。
- この方式で、44.9GB の **Qwen3-Next-80B-A3B(4bit)が 8GB 機で動きました**(外付け SSD から 1.5〜2.3 tok/s)。同じ機体では、素の mlx-lm も llama.cpp の GPU 実行も、メモリ不足で動きません。

## 主な結果

すべて M1 / 8GB で実測した値です。速度は生成(デコード)の tok/s です。

| モデル | 置き場所 | 生成速度 | 備考 |
|---|---|---|---|
| Qwen3-30B-A3B 4bit(17.2GB) | 内蔵 SSD | 3.5〜3.8 | キャッシュ 2.45〜3.26GB |
| Qwen3-30B-A3B 4bit | 外付け SSD | 1.2〜1.5 | USB 10Gb/s(約 0.7GB/s) |
| Qwen3.6-35B-A3B 4bit(19.5GB) | 外付け SSD | コード 1.9〜2.1 / 会話 3.7〜3.9 | 先読みの自動調整後 |
| Qwen3-Next-80B-A3B 4bit(44.9GB) | 外付け SSD | コード 1.2〜1.5 / 会話 1.9〜2.3 | 先読みの自動調整後 |

既存の実行方法との比較(Qwen3-30B-A3B):

| 方法 | 結果 |
|---|---|
| 素の mlx-lm | 生成開始後、約 2 分で GPU のメモリ不足 |
| llama.cpp(GPU) | GPU のメモリ不足 |
| llama.cpp(CPU のみ、mmap) | 0.7〜1.3 tok/s |
| **このリポジトリの方式** | **3.5〜3.8 tok/s**(内蔵 SSD) |

コーディング課題(HumanEval 20 問、貪欲デコード、最大 512 トークン)の合格数は次のとおりです。

| モデル | 合格 |
|---|---|
| Qwen3-Next-80B-A3B | 19 / 20 |
| Qwen3.6-35B-A3B | 18 / 20 |

ただし、メモリに素直に載る dense の Qwen3-8B(4bit)は 13.4 tok/s で、Qwen3-30B-A3B より約 4 倍速く出ます。品質(簡易 perplexity)は Qwen3-30B-A3B とほぼ同等でした。この方式が本当に意味を持つのは、80B 級のような「8GB 機では他に動かす手段がないモデル」です。

詳細は [bench/compare.md](bench/compare.md)、[bench/external.md](bench/external.md)、[bench/coding_compare.md](bench/coding_compare.md) にあります。

## 仕組み

1. **再配置**(`tools/repack.py`):safetensors から、(層, エキスパート) ごとに gate / up / down の重み・scales・biases を 1 つの連続したブロブにまとめます。
   - ブロブは 16KB 境界に揃え、1 エキスパートを 1 回の読み出しで取れるようにします。
   - 書き込み前に連続した領域を確保します(`F_PREALLOCATE`)。APFS 上で断片化すると、読み出しが約半分の速度に落ちるためです。
   - HF からはヘッダだけ取って配置を決め、シャードを 1 本ずつダウンロード・変換・削除できます。ディスクが元モデル分だけあれば足ります。
2. **ストリーミング層**(`smoe/layer.py`):mlx-lm の `SwitchGLU` を差し替えます。
   - 層ごとにルーティング結果を確定させ、必要なエキスパートをキャッシュから取り、無ければ SSD から読みます。
   - 1 トークンのときは、エキスパートごとに `quantized_matmul` で直接計算し、コピーを避けます。
   - prefill のように多くのエキスパートを使うときは、32 個ずつ区切って計算し、一時メモリを抑えます。
3. **読み出し**(`smoe/loader.py`):`F_NOCACHE` を付けた `preadv` で、MLX の配列のメモリへ直接読み込みます(ゼロコピー)。
4. **キャッシュ**(`smoe/cache.py`):(層, エキスパート) 単位の LRU で、容量はバイト数で指定します。prefill で読んだものは、プロンプト末尾のトークンが使った分だけをキャッシュに入れます。
5. **先読み**:層 l のブロック入力に層 l+1 のルーターを当てて、使われそうなエキスパートを予測し、裏で読み始めます。
   - 予測は、ルーティングの確定と同じ 1 回の評価で確定させます。同期の回数は増えません。
   - 先読みは専用の 1 スレッドで確信度の高い順に読み、使われないと分かったものは読み始める前に取り消します。帯域の細い外付け SSD でも、外れた読み出しで遅くなりません。
6. **同期のコスト削減**:層ごとの同期で GPU が止まるのが最大のオーバーヘッドでした。MoE の出力を組み立てた直後に `mx.async_eval` で GPU に投げ、次の層の組み立てと重ねます。

`SwitchGLU` を使う mlx-lm のモデルなら、同じ差し替えで動く見込みです。動作を確かめたのは、OLMoE-1B-7B、Qwen3-30B-A3B、Qwen3-Next-80B-A3B、Qwen3.6-35B-A3B です。

## 使い方

```sh
uv sync                  # mlx 0.32.3 / mlx-lm 0.32.0 に固定
uv sync --extra bench    # HumanEval の評価を使う場合(pyarrow)
```

再配置(HF からシャードを 1 本ずつ取得して変換します。出力先は外付けでも構いません):

```sh
uv run python tools/repack.py --repo mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit models/qwen3-next-80b
# ローカルの safetensors から作る場合
uv run python tools/repack.py <snapshot_dir> models/<name> [--delete-source]
```

生成:

```python
from mlx_lm import stream_generate
from smoe import load_streaming

model, tok, cache = load_streaming("models/qwen3-next-80b", cache_gb=3.0, prefetch_m=10)
ids = tok.apply_chat_template([{"role": "user", "content": "日本の首都は?"}], add_generation_prompt=True)
for r in stream_generate(model, tok, ids, max_tokens=64):
    print(r.text, end="", flush=True)
print(cache.stats())
```

- `cache_gb`:エキスパートキャッシュの容量です。常駐部(1〜1.4GB)と合わせて、GPU 作業領域(M1 8GB で約 5.4GB)に収まるようにします。2.5〜3GB が目安です。
- `prefetch_m`:先読みする個数です。そのモデルの top-k(Qwen3-Next は 10、Qwen3.6 / Qwen3-30B は 8)が目安です。

ベンチマークとテスト:

```sh
uv run python bench/stream_bench.py models/<name> --cache-gb 2.45 3.0 --prefetch-m 8   # 速度とヒット率
uv run python bench/phase_hits.py models/<name> --prefetch-m 8 --out out.jsonl         # prefill / 生成の区間別
uv run python bench/humaneval.py --model models/<name> --smoe --prefetch-m 8 --n 20    # コーディング課題
uv run python bench/quality.py models/<name> --smoe                                     # 簡易 perplexity
SMOE_REF=mlx-community/OLMoE-1B-7B-0125-Instruct-4bit SMOE_DIR=models/olmoe-4bit-smoe \
    uv run pytest tests                                                                  # 素の mlx-lm との一致
```

## 構成

```
smoe/            ストリーミング MoE の本体(layer / loader / cache)
tools/           再配置ツール(repack.py)、ルーティング記録(route_log.py)
analysis/        ルーティング解析(キャッシュのシミュレータ、ワークロード、レポート)
bench/           ベンチマークのスクリプト、結果の文書、ログ
tests/           素の mlx-lm との出力一致テスト、シミュレータのテスト
HANDOVER.md      実験開始時の引き継ぎ文書(方針とアイデア)
PLAN.md          実行計画と、フェーズごとの進捗メモ
```

## 実験の経緯

実験は [HANDOVER.md](HANDOVER.md) の方針(エキスパートのストリーミング、ルーティング統計にもとづくキャッシュと先読み)から始めました。各段階の結果は、リンク先の文書にまとめています。

1. **ベースラインと SSD の性能**([bench/baseline.md](bench/baseline.md)):内蔵 SSD は、1〜4MB のランダム読み出しで 2.5〜3.2GB/s 出ました。
2. **ルーティング解析**([analysis/routing_report.md](analysis/routing_report.md)):
   - LRU のヒット率は、キャッシュ容量が全体の 20% のとき、OLMoE で 38〜55%、Qwen3-30B で 71〜74% でした。
   - 1 層先のルーティング予測の recall@2k は 92〜97% と高く出ました。
   - ARC や S3-FIFO などの追い出し方針は、LRU との差が ±1 ポイントにとどまりました。
3. **ストリーミングの最小実装**([bench/phase2_olmoe.md](bench/phase2_olmoe.md)、[bench/phase2_qwen3.md](bench/phase2_qwen3.md)):
   - OLMoE で素の mlx-lm と出力が一致することを確認し、そのうえで Qwen3-30B-A3B を動かしました。
   - 3bit 版は速い一方で、perplexity が 55〜72% 悪化しました。
4. **読み出しと計算の重ね合わせ**([bench/phase3_io.md](bench/phase3_io.md)):
   - 断片化の解消(+20〜28%)、ゼロコピー読み込み(約 +20%)、同期の削減(+8%)で、内蔵 SSD の 4bit が 2.0 → 3.5 tok/s になりました。
   - 投機デコードは、mlx-lm 標準の実装のままでは逆効果でした。
5. **既存手段との比較**([bench/compare.md](bench/compare.md)):素の mlx-lm と llama.cpp(GPU)では動かず、llama.cpp(CPU)より 3〜5 倍速いことを確かめました。
6. **外付け SSD と 80B**([bench/external.md](bench/external.md)):
   - Qwen3-Next-80B が動きました。エキスパートが細かい(512 個)ため、キャッシュが全体の 6% しかなくてもヒット率 60% が出ました。
   - 外付けでは帯域が律速になるため、先読みの最適な個数が内蔵と逆になりました。そこで、取り消しと専用スレッドで自動的に絞る仕組みを入れました。
7. **Qwen3.6-35B と 80B の比較**([bench/coding_compare.md](bench/coding_compare.md)):
   - HumanEval では 18 / 20 と 19 / 20 でした。80B の回答は約半分の長さで、課題全体の所要時間は 80B のほうが短くなりました。
   - コード生成は、会話よりキャッシュのヒット率が 15〜25 ポイント低く、生成速度もほぼ半分になります。

比較用のモデル候補は [analysis/model_candidates.md](analysis/model_candidates.md) にまとめています。

## 制限と注意

- 計測した機体は M1 MacBook Air 8GB の 1 台だけです。同じ設定でも、実行ごとに ±1 割程度ぶれます。
- 品質評価は、簡易 perplexity(数百トークン)と HumanEval 20 問だけです。HumanEval は学習データに含まれている可能性があります。
- 内蔵 SSD に 80B / Qwen3.6 を置いた計測はしていません(空き容量が足りないため)。先読みの既定値(専用 1 スレッド)は外付けで決めたもので、内蔵での最適値は未確認です。
- `smoe/layer.py` は mlx-lm の内部(`switch_layers`、モデルクラスの `__call__`)に依存しています。mlx-lm のバージョンを上げると動かなくなる可能性があります。
- Qwen3.8-Flash-Next(qwen4_exp)と gpt-oss-120b は未対応です。調査結果は [analysis/model_candidates.md](analysis/model_candidates.md) にあります。

## ライセンス

[MIT](LICENSE)
