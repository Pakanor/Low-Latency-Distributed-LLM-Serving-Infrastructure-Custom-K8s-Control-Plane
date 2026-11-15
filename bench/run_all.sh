#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/venv/bin/python}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT="$($PY - <<'PY'
import subprocess
try:
    print(subprocess.check_output(['git','rev-parse','--short','HEAD'], text=True).strip())
except Exception:
    print('nogit')
PY
)"
OUT_DIR="${1:-$ROOT/bench/results/${DATE_TAG}_${COMMIT}}"
mkdir -p "$OUT_DIR/plots"

echo "[bench] output: $OUT_DIR"
"$PY" bench/env_info.py "$OUT_DIR/env.json"
"$PY" -c "from bench.common import log_cpu_freqs; from pathlib import Path; log_cpu_freqs(Path('$OUT_DIR/cpu_freq.csv'),'before')"

# Krok 1: poprawność (pełny zestaw domyślnie; użyj CORRECTNESS_QUICK=1 dla szybkiego przebiegu)
if [[ "${SKIP_CORRECTNESS:-0}" != "1" ]]; then
  if [[ "${CORRECTNESS_QUICK:-0}" == "1" ]]; then
    "$PY" bench/correctness.py --output "$OUT_DIR/correctness.json" --quick
  else
    "$PY" bench/correctness.py --output "$OUT_DIR/correctness.json"
  fi
else
  echo "[bench] skipping correctness (SKIP_CORRECTNESS=1)"
fi

"$PY" -c "from bench.common import log_cpu_freqs; from pathlib import Path; log_cpu_freqs(Path('$OUT_DIR/cpu_freq.csv'),'after')"
echo "[bench] done -> $OUT_DIR"
