#!/usr/bin/env bash
set -euo pipefail

# Colab launcher. Override paths with environment variables, e.g.:
#   DATASET=seed-vig DATA_DIR="/content/drive/My Drive/.../datasets/SEED-VIG" \
#   NUM_CHANNELS=17 NUM_CLASSES=3 bash run_script/run_colab.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ============================================================================
# CONFIGURATION (override via env vars on Colab)
# ============================================================================
DATASET="${DATASET:-seed-vig}"
DATA_DIR="${DATA_DIR:-/content/drive/My Drive/Driving Fatigue Project/Data/SEED-VIG}"
NUM_CHANNELS="${NUM_CHANNELS:-17}"
NUM_CLASSES="${NUM_CLASSES:-3}"
MODE="${MODE:-intra}"
N_FOLDS="${N_FOLDS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs}"

python -m train.train \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --num_channels "$NUM_CHANNELS" \
    --num_classes "$NUM_CLASSES" \
    --mode "$MODE" \
    --n_folds "$N_FOLDS" \
    --output_dir "$OUTPUT_DIR"
