#!/bin/bash
set -euo pipefail

# ============================================================
# Humboldt University Berlin
# Master's Thesis
#
# Stage: Train SLEAP top-down models
# Script: 03_train_topdown_models.sh
# Author: Olivia Francis
#
# Purpose:
# Trains the centroid and centered-instance models using the
# saved SLEAP-NN configurations and explicit reproducibility
# overrides.
# ============================================================

PROJECT_DIR="/lustre/biologie/franciso/Humboldt_Thesis"

cd "$PROJECT_DIR"
source environments/sleap_env/bin/activate

echo "Checking GPU..."
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

mkdir -p models/topdown_v1/centroid
mkdir -p models/topdown_v1/centered_instance

echo "Training centroid model..."

env -u SLURM_NTASKS \
SLURM_JOB_NAME=interactive \
sleap-nn train \
  training_profiles/topdown_v1/training_centroid.yaml \
  trainer_config.seed=42 \
  trainer_config.run_name=topdown_v1_centroid \
  trainer_config.ckpt_dir="$PROJECT_DIR/models/topdown_v1/centroid"

echo "Centroid training completed."
echo "Training centered-instance model..."

env -u SLURM_NTASKS \
SLURM_JOB_NAME=interactive \
sleap-nn train \
  training_profiles/topdown_v1/training_centered_instance.yaml \
  trainer_config.seed=42 \
  trainer_config.run_name=topdown_v1_centered_instance \
  trainer_config.ckpt_dir="$PROJECT_DIR/models/topdown_v1/centered_instance"
echo "Centered-instance training completed."
echo "Top-down model training completed successfully."
