#!/bin/bash

#!/bin/bash

# ============================================================
# Humboldt University Berlin
# Bachelor's Thesis
#
# Stage: Generate SLEAP Training Configurations
# Script: 02_generate_training_configs.sh
# Author: Olivia Francis
# Date Created: 2026-07-13
#
# Purpose:
# Generates reproducible top-down SLEAP-NN training
# configuration files from the validated annotation dataset.
# ============================================================

cd /lustre/biologie/franciso/Humboldt_Thesis

# Activate environment
source environments/sleap_env/bin/activate

echo "Checking environment..."

python -c "import sleap; sleap.versions()"

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

echo ""
echo "Generating Top-Down Training Configurations..."

sleap-nn config \
    labels/SLEAP_social_training_v1_124frames_HPC.slp \
    --auto \
    --pipeline topdown \
    --output training_profiles/topdown_v1/training.yaml

echo ""
echo "Done."

