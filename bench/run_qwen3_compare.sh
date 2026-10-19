#!/bin/zsh
# Qwen3-30B-A3B の 3bit / 4bit を同条件で比べる(速度 → 品質の順に 1 本ずつ実行)。
# 再起動直後など、スワップが空の状態で実行する。
#   zsh bench/run_qwen3_compare.sh [ラベル]
set -eu
cd "$(dirname "$0")/.."
tag=${1:-$(date +%Y%m%d-%H%M)}
out=bench/logs/$tag
mkdir -p $out
{ sysctl -n vm.swapusage; memory_pressure | tail -1; } > $out/conditions.txt
for v in 3bit 4bit; do
  uv run python -u bench/stream_bench.py models/qwen3-30b-a3b-$v-smoe \
    --cache-gb 1.63 2.45 3.26 --max-tokens 64 > $out/speed_$v.jsonl 2> $out/speed_$v.err
  uv run python -u bench/quality.py models/qwen3-30b-a3b-$v-smoe --smoe --cache-gb 3 \
    > $out/quality_$v.json 2> $out/quality_$v.err
done
sysctl -n vm.swapusage >> $out/conditions.txt
cat $out/speed_*.jsonl $out/quality_*.json
