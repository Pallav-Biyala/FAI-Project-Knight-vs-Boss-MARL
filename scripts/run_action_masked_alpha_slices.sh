#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT_ROOT="${1:-experiments_action_masked_alpha_slices_v1}"

COMMON_ARGS=(
  --out "$OUTPUT_ROOT"
  --seeds 0 1 2 3 4
  --diagonal 0.5 0.75 1.0 1.25 1.5
  --durability 0.5 0.75 1.25 1.5
  --lethality 0.5 0.75 1.25 1.5
  # Skip fresh best-response Boss training for this reduced-cost grid.
  --br-steps 0
  --eval-episodes 500
  --reward-mode tactical
  --workers 1
)

# Fixed and simultaneous: 40k environment steps per run.
"$PYTHON" scripts/run_proposal_experiments.py \
  "${COMMON_ARGS[@]}" \
  --regimes fixed_opponent \
  --steps 40000

"$PYTHON" scripts/run_proposal_experiments.py \
  "${COMMON_ARGS[@]}" \
  --regimes simultaneous_learning \
  --steps 40000

# Alternating: 80k total steps, split into 40k Knight and 40k Boss phases.
"$PYTHON" scripts/run_proposal_experiments.py \
  "${COMMON_ARGS[@]}" \
  --regimes alternating_learning \
  --steps 80000 \
  --phase-steps 40000

echo "Alpha-slice results are in: $OUTPUT_ROOT/results.jsonl"
