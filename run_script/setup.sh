#!/usr/bin/env bash
set -euo pipefail

# Create and activate the DeltaGateNet conda environment.
# Run from the repository root:  bash run_script/setup.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

conda env create -f environment.yml
echo "Environment created. Activate with: conda activate deltagatenet"
