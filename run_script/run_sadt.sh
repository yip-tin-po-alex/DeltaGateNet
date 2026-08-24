#!/usr/bin/env bash
set -euo pipefail

# Train DeltaGateNet on SADT (30 channels, 2 classes).
# Run from the repository root:  bash run_script/run_sadt.sh
#
# Switch DATA_DIR to ./datasets/SADT-2952 for the unbalanced release.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET="sadt"
DATA_DIR="./datasets/SADT-2022"   # or ./datasets/SADT-2952
NUM_CHANNELS=30
NUM_CLASSES=2
MODE="intra"          # "intra" or "inter"
N_FOLDS=5
OUTPUT_DIR="./logs"

python -m train.train \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --num_channels "$NUM_CHANNELS" \
    --num_classes "$NUM_CLASSES" \
    --mode "$MODE" \
    --n_folds "$N_FOLDS" \
    --output_dir "$OUTPUT_DIR"
