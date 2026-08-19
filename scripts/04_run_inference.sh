#!/bin/bash
set -euo pipefail

# ============================================================
# Humboldt University Berlin
# Master's Thesis
#
# Stage: Run SLEAP inference on unseen validation footage
# Script: 04_run_inference.sh
# Author: Olivia Francis
# ============================================================

PROJECT_DIR="/lustre/biologie/franciso/Humboldt_Thesis"

cd "$PROJECT_DIR"
source environments/sleap_env/bin/activate

mkdir -p outputs/validation_v1

env -u SLURM_NTASKS \
SLURM_JOB_NAME=interactive \
sleap-nn track \
  --data_path incoming/405406validation15julyat1232_1.avi \
  --model_paths models/topdown_v1/centroid/topdown_v1_centroid \
  --model_paths models/topdown_v1/centered_instance/topdown_v1_centered_instance \
  --output_path outputs/validation_v1/405406validation_first30s.predictions.slp \
  --device cuda \
  --batch_size 8 \
  --frames 0-899 \
  --max_instances 2 \
  --tracking

echo "Validation inference completed successfully."
