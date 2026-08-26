#!/bin/bash
set -euo pipefail

# ============================================================
# Humboldt University Berlin
# Master's Thesis
#
# Stage: Run final SLEAP V2 inference on experimental recordings
# Script: 04_run_inference.sh
# Author: Olivia Francis
# ============================================================

# Usage:
# bash scripts/04_run_inference.sh VIDEO_PATH OUTPUT_PATH MAX_INSTANCES
#
# Example:
# bash scripts/04_run_inference.sh \
#   videos/TESTsleapvideos/4074408TE1at1040test_1.avi \
#   outputs/experimental_v2/4074408TE1at1040test_v2.predictions.slp \
#   2

VIDEO_PATH="$1"
OUTPUT_PATH="$2"
MAX_INSTANCES="$3"

CENTROID_MODEL="models/topdown_v2/centroid/topdown_v2_centroid"
CENTERED_INSTANCE_MODEL="models/topdown_v2/centered_instance/topdown_v2_centered_instance"

mkdir -p "$(dirname "$OUTPUT_PATH")"

env -u SLURM_NTASKS \
sleap-nn track \
  --data_path "$VIDEO_PATH" \
  --model_paths "$CENTROID_MODEL" \
  --model_paths "$CENTERED_INSTANCE_MODEL" \
  --output_path "$OUTPUT_PATH" \
  --device cuda \
  --batch_size 8 \
  --max_instances "$MAX_INSTANCES" \
  --tracking

echo "Final V2 inference completed successfully."
echo "Predictions saved to: $OUTPUT_PATH"
