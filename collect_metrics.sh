#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/logs"
OUT="$DIR/logs/metrics_$(date +%Y%m%d_%H%M%S).log"
echo "# timestamp kv_cache_free_pages_ratio kv_cache_active_streams kv_cache_preemptions_total" > "$OUT"
trap 'exit 0' INT TERM HUP
while :; do
  TS=$(date +%s.%3N)
  M=$(curl -s --connect-timeout 1 http://localhost:8000/metrics)
  R=$(echo "$M" | awk '/kv_cache_free_pages_ratio/{print $2}')
  A=$(echo "$M" | awk '/kv_cache_active_streams/{print $2}')
  P=$(echo "$M" | awk '/kv_cache_preemptions_total/{print $2}')
  echo "$TS ${R:-0} ${A:-0} ${P:-0}" >> "$OUT"
  sleep 0.1
done
