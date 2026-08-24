#!/usr/bin/env bash
set -euo pipefail

# Train DeltaGateNet on SEED-VIG (17 channels, 3 classes).
# Run from the repository root:  bash run_script/run_seedvig.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET="seed-vig"
DATA_DIR="./datasets/SEED-VIG"
NUM_CHANNELS=17
NUM_CLASSES=3
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
