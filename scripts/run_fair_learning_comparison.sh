#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT_ROOT="${1:-experiments_confirmatory_v1}"

COMMON_ARGS=(
  --out "$OUTPUT_ROOT"
  --seeds 0 1 2 3 4
  --diagonal 1.0
  --br-steps 20000
  --eval-episodes 500
  --reward-mode tactical
  --workers 1
)

# Fixed and simultaneous learning each give the Knight about 40 PPO updates.
"$PYTHON" scripts/run_proposal_experiments.py \
  "${COMMON_ARGS[@]}" \
  --regimes fixed_opponent \
  --steps 40000

"$PYTHON" scripts/run_proposal_experiments.py \
  "${COMMON_ARGS[@]}" \
  --regimes simultaneous_learning \
  --steps 40000

# Alternating learning needs 80,000 total steps so each agent gets a 40,000-step
# phase. With only 40,000 total steps, the Boss phase never starts.
"$PYTHON" scripts/run_proposal_experiments.py \
  "${COMMON_ARGS[@]}" \
  --regimes alternating_learning \
  --steps 80000 \
  --phase-steps 40000

# Population self-play alternates updates between Knight and Boss, so use about
# twice the environment steps to match the Knight's active update budget.
"$PYTHON" scripts/run_proposal_experiments.py \
  "${COMMON_ARGS[@]}" \
  --regimes population_self_play \
  --steps 80000 \
  --population-size 5 \
  --phase-updates 2 \
  --snapshot-interval-updates 2

echo "Experiment results are in: $OUTPUT_ROOT/results.jsonl"
