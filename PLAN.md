# 実行計画

HANDOVER.md の Phase 0〜4 を、この機体の実測値に合わせて具体化したもの。各フェーズ末に「結果」を追記する。

## 着手時の実測(2026-10-14)

| 項目 | 値 | HANDOVER の想定との差 |
|---|---|---|
| 機体 | Apple M1 / 8GB / macOS 27.0.1 / ページ 16KB | 一致 |
| 空きディスク | **18GB**(228GB 中) | 想定外。最大の制約 |
| スワップ | 着手時点で 7.2GB 使用中(VSCode ほか起動中) | ベンチは他アプリを閉じて取り直す必要あり |
| mlx / mlx-lm | 0.32.3 / 0.32.0(pyproject で固定、Python 3.13) | — |

## 確認済みの設計前提(mlx-lm 0.32.0 のソースより)

- OLMoE / Qwen3-MoE は `SwitchGLU` → `QuantizedSwitchLinear` → `mx.gather_qmm(rhs_indices=...)`。
- エキスパート重みは `(num_experts, out, in)` に積んだ 1 本の配列(weight / scales / biases)。
- よって HANDOVER §9 の第 1 問は **インデックスの付け替えだけで済む**(カスタムカーネル不要)。
  方式は 2 つあり、Phase 2 で実測して選ぶ:
  - A. キャッシュスロット配列 `(S, out, in)` + ID→スロット変換。スロット書き換えが配列全体のコピーになる懸念あり。
  - B. エキスパートごとに独立した配列を持ち、top-k 個を個別に `quantized_matmul`(コピーなし、op 数は k 倍)。
- デコード時(トークン数 < 64)は `_gather_sort` を通らないので、差し替えは `SwitchGLU.__call__` 相当だけでよい。

## ディスク予算(18GB)

| 時点 | 置くもの | 概算 |
|---|---|---|
| Phase 0–1 | OLMoE 4bit | 約 4GB |
| Phase 1–2 | 16B 級 4bit(元の safetensors) | 約 9GB |
| Phase 2 | 16B 級の再配置ブロブ | +約 9GB → **元と同時には置けない** |
| Phase 3 | Qwen3-30B-A3B 4bit | 約 17GB → **単体でも入らない** |

対策:
- 再配置ツールは「シャード 1 本をダウンロード → ブロブへ追記 → シャード削除」の逐次方式にする。
- 非エキスパート部は別の小さな safetensors に切り出し、元ファイルは残さない。
- Phase 3 の前に、空き容量を増やすか外付け SSD を使うかの判断が要る(下の「相談事項」)。

## フェーズと担当

担当: **O** = Opus(設計・判断・正しさに関わる実装)、**S** = Sonnet、**H** = Haiku(定型作業)。

### Phase 0: 環境とベースライン
- [x] O: uv プロジェクト作成、mlx / mlx-lm のバージョン固定
- [x] S: `bench/ssd_bench.py`(pread + F_NOCACHE、64KB〜8MB × QD1〜8、mmap 比較)と実行
- [x] S: モデル調査(mlx-community の 4bit 変換の有無・サイズ・model_type・MoE 実装)
- [x] O: OLMoE 4bit を素の mlx-lm で実行し、tok/s・prefill・ピークメモリを記録
- [x] H: `bench/baseline.md` に数値と再現コマンドを整形
- 完了条件: `bench/baseline.md` → **完了**(H への整形依頼はせず O が直接書いた)

### Phase 1: ルーティング解析
- [x] O: `tools/route_log.py`(MoE ブロックをラップし、層・トークンごとの top-k ID と重み、
      および先読み評価用に「第 L 層入力へ第 L+1 / L+2 層のルーターを当てた top-k / top-2k」を記録。npz 出力)
- [x] S: ワークロード 3 種(コード / 日本語会話 / 英文)のプロンプトセット作成
- [x] S: `analysis/cache_sim.py`(LRU / LFU / Belady のヒット率、容量 10〜50%)
- [x] S: 使用頻度分布・共起・先読み recall の集計
- [x] O: 結果を読んで Phase 2 のキャッシュ方針を決める → `analysis/routing_report.md`(OLMoE のみ。16B 級は未着手)
- 対象は OLMoE と 16B 級。16B 級は全載せできないので、ルーターだけ先に解析する手も検討
  (ルーティングの記録にはエキスパート計算が必要なため、遅くても mmap 任せで一度流す)。
- 完了条件: キャッシュ 20% / 30% のヒット率と先読み recall がレポートにある

### Phase 2: ストリーミング最小実装
- [x] O: `tools/repack.py`(エキスパート単位ブロブ、16KB 境界、JSON インデックス、逐次方式)
- [x] O: `smoe/layer.py`(SwitchGLU 差し替え)、`smoe/loader.py`(7(a) mmap → 7(b) pread)、`smoe/cache.py`
- [x] O: `tests/test_logits.py`(OLMoE で素の mlx-lm とロジット一致)
- [ ] S: メモリ予算の起動時ログ、tok/s・ヒット率・読み出し量のログ出力、ベンチスクリプト
- 完了条件: 16B 級がスワップなしで生成でき、tok/s とヒット率が出る

### Phase 3: I/O と計算の重ね合わせ
- [ ] O: 先読み(pre-gating)+ 非同期 I/O スレッド、層単位 prefill
- [ ] O: MTLIOCommandQueue は Phase 2 の結果で要否を判断
- 完了条件: Qwen3-30B-A3B で生成、Phase 2 比の改善率

### Phase 4: 発展
- 投機デコード統合 / 剪定 / ビットプレーン / ANE。Phase 3 の数値を見て優先順位を決め直す。

## 相談事項(いまは作業を止めない)

1. **ディスク**: Qwen3-30B-A3B(約 17GB)は現状の空き 18GB では扱えない。Phase 3 までに
   (a) 内蔵を 25GB 以上空ける、(b) 外付け SSD を使う(その場合 I/O 特性が変わる)、(c) 本命を 16B 級に下げる、のどれか。
2. **ベンチ条件**: 現状スワップ 7GB 使用中。正式な数値は VSCode 以外を閉じた状態で取り直したい。

## 進捗(2026-10-14)

- Phase 0: 完了。[bench/baseline.md](bench/baseline.md)
- Phase 1: OLMoE は完了。[analysis/routing_report.md](analysis/routing_report.md)。16B 級は未着手。
- Phase 2: OLMoE でロジット一致と速度計測まで完了。[bench/phase2_olmoe.md](bench/phase2_olmoe.md)。
  16B 級での実行と、メモリ予算ログ・メモリ圧への追従は未着手。
- 設計前提の更新: エキスパートの持ち方は方式 B(個別配列)を採用。1 トークン時は積み直さずに個別に掛け、複数トークン時だけ使う分を積む。

## 進捗(2026-10-15)

- Qwen3-30B-A3B の 4bit / 3bit をストリーミングで生成できた。[bench/phase2_qwen3.md](bench/phase2_qwen3.md)。
  - 4bit はキャッシュ 2.45GB で 1.9 tok/s。3bit は 2.4 tok/s だが、perplexity が 55〜72% 悪化する。
- Bonsai は dense のみで 3bit 版も無く、ストリーミングの対象外。比較候補と合わせて [analysis/model_candidates.md](analysis/model_candidates.md)。
- 次:
  1. スワップを空にして取り直す(`zsh bench/run_qwen3_compare.sh`)。
  2. Phase 1 を Qwen3 で取る(`tools/route_log.py --smoe`)。
  3. Phase 3 で、読み出しの並列度(1 層のミスが 3〜4 個しかない)を上げ、先読みで計算と重ねる。

## 進捗(2026-10-16)

- スワップを空にして取り直した結果、読み出しが遅い主因はスワップではなく、`experts.bin` の**断片化**だった。
  `tools/repack.py` に連続確保を入れて解消し、+20〜28%。
- 1 層先の先読み(m = k)で、3bit +20%、4bit +4〜17%。
- 現状のベスト(キャッシュ 2.45GB): **4bit 2.5〜2.8 tok/s、3bit 3.8〜3.9 tok/s**。
- 詳細は [bench/phase3_io.md](bench/phase3_io.md)。
- ゼロコピー読み込み + 同期削減で、**4bit 3.5 tok/s(キャッシュ 2.45GB)/ 3.8 tok/s(3.26GB)、3bit 4.6〜4.8 tok/s**。
- Qwen3 のルーティング解析を追加([analysis/routing_report.md](analysis/routing_report.md))。キャッシュ方針は LRU で十分(汎用方針は ±1 ポイント)。
- 層ごとの LRU は実機で効果なし。投機デコードは、標準実装のままでは逆効果(1.7〜2.2 tok/s)。
- 既存の実行方法と比較した([bench/compare.md](bench/compare.md))。
  - 素の mlx-lm と llama.cpp の GPU 実行は、メモリ不足で動かない。
  - 動く既存手段は llama.cpp の CPU 実行(0.7〜1.3 tok/s)だけで、本方式はその 3〜5 倍速い。
  - ただし dense の Qwen3-8B 4bit は 13.4 tok/s で約 4 倍速く、簡易 perplexity の差は 7〜9% 以内。
- 3bit の再配置版と、比較用モデルは削除した。残りは OLMoE と Qwen3 4bit の再配置版のみ。
- 外付け SSD(USB 10Gb/s、約 0.75GB/s)で計測した([bench/external.md](bench/external.md))。
  - 30B 4bit は 1.2〜1.5 tok/s(内蔵の約 4 割)。
  - **Qwen3-Next-80B-A3B 4bit が 8GB 機で動いた。** 外付けから 1.5〜1.7 tok/s、MLX ピーク 4.2〜4.7GB。
  - 簡易 perplexity は 30B・8B より大幅によい(日本語 7.64 vs 11.85 / 12.90)。
- Qwen3 30B 4bit の再配置版は外付けへ移した(`models/` にはシンボリックリンク)。
- Qwen3.6-35B-A3B を外付けから動かした(画像側の重みを捨てる以外は変更なし)。生成 1.9〜2.1 tok/s。
- HumanEval 20 問: Qwen3.6-35B 18/20、Qwen3-Next-80B 19/20([bench/coding_compare.md](bench/coding_compare.md))。80B は回答が半分の長さで、課題全体では速い。
- prefill の一時メモリでメモリ不足になったため、エキスパートを 32 個ずつ処理するチャンク経路を追加した。
- 区間別のヒット率を計測した。コード生成が遅い主因は、prefill によるキャッシュ荒れではなく、**コード生成のヒット率自体が会話より 15〜25 ポイント低い**こと。
  prefill の扱いは「末尾 16 トークン分だけキャッシュに入れる」を既定にした(prefill が 10〜15% 速くなる。生成の頭で +0〜3 ポイント)。
- 先読みの個数を外付けで再調整した。2 層先は逆効果。1 層先は外付けでは 4 個が最良(top-k と同数だと無駄読みで 20〜25% 遅い)。内蔵では top-k と同数が最良なので、帯域で最適値が変わる([bench/external.md](bench/external.md))。
- 先読みの自動調整を実装した(不要な先読みの取り消し、先読み専用スレッド、確信度順の発行)。外付けで、先読みスレッド 1 本・個数 = top-k が最良になり、改修前の最良より Qwen3.6 で 8〜10%、80B で 2〜9% 速い。既定を pf_threads=1 にした。
