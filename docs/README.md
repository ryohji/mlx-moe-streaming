# 実験記録

実験ごとに 1 ファイルを `NN-短い名前.md` の形で追加し、この目次に 1 行足す。
各ファイルには、目的、結論、手順(再現コマンド)、結果、分かったこと、次の実験を書く。

## 目次

| # | 文書 | 期間 | 要旨 |
|---|---|---|---|
| 01 | [GitHub Copilot Chat にローカルモデルを登録する](01-copilot-local-model.md) | 2026-10-19〜21 | 登録はできたが、約 2 万トークンのプロンプトで prefill が 40 分前後かかり、メモリ不足で失敗。自走型エージェントは別方式で作る |

## これまでの記録(docs/ を作る前のもの)

実験の主要部分は、以下の文書にまとめてある。経緯の概要はリポジトリの [README](../README.md) を参照。

| 段階 | 文書 |
|---|---|
| 実行計画と進捗メモ | [PLAN.md](../PLAN.md) |
| ベースラインと SSD の性能 | [bench/baseline.md](../bench/baseline.md)、[bench/ssd_results.md](../bench/ssd_results.md) |
| ルーティング解析 | [analysis/routing_report.md](../analysis/routing_report.md) |
| ストリーミングの最小実装(OLMoE / Qwen3-30B) | [bench/phase2_olmoe.md](../bench/phase2_olmoe.md)、[bench/phase2_qwen3.md](../bench/phase2_qwen3.md) |
| 読み出しと計算の重ね合わせ | [bench/phase3_io.md](../bench/phase3_io.md) |
| 既存の実行方法との比較 | [bench/compare.md](../bench/compare.md) |
| 外付け SSD、80B、先読みの自動調整 | [bench/external.md](../bench/external.md) |
| Qwen3.6-35B と 80B のコーディング比較 | [bench/coding_compare.md](../bench/coding_compare.md) |
| 比較用モデルの候補 | [analysis/model_candidates.md](../analysis/model_candidates.md) |
