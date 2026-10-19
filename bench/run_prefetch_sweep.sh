#!/bin/zsh
# 先読み設定のスイープ。キャッシュ 2.45GB 固定、4bit と 3bit。
#   zsh bench/run_prefetch_sweep.sh [ラベル]
set -eu
cd "$(dirname "$0")/.."
tag=${1:-prefetch-$(date +%Y%m%d-%H%M)}
out=bench/logs/$tag
mkdir -p $out
{ sysctl -n vm.swapusage; memory_pressure | tail -1; } > $out/conditions.txt
for v in 4bit 3bit; do
  for opt in "--prefetch-m 0" "--prefetch-m 8" "--prefetch-m 12" "--prefetch-m 12 --prefetch-early" "--prefetch-m 16 --prefetch-early"; do
    uv run python -u bench/stream_bench.py models/qwen3-30b-a3b-$v-smoe --cache-gb 2.45 --max-tokens 64 ${=opt} \
      | sed "s/^{/{\"model\": \"$v\", /" >> $out/sweep.jsonl 2>> $out/sweep.err
  done
done
sysctl -n vm.swapusage >> $out/conditions.txt
cat $out/sweep.jsonl
