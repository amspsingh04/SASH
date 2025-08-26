#!/bin/bash

export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

CKPT_PATH="/root/hack/data/buddy/experiments/VCTK_16k_4s_time-10350.pt"

INPUT_AUDIO_DIR="/root/hack/data/audio_16k/valid/reverbs"


EXPERIMENT_NAME="my_first_test_run"

# ---
# --- DO NOT EDIT BELOW THIS LINE ---
# ---

# Set up the output directory path
PATH_EXPERIMENT="experiments/$EXPERIMENT_NAME"
mkdir -p $PATH_EXPERIMENT

echo "--- Running Test ---"
echo "Checkpoint: $CKPT_PATH"
echo "Input Audio: $INPUT_AUDIO_DIR"
echo "Output will be saved to: $PATH_EXPERIMENT"
echo "--------------------"

# This is the main command that runs the testing process.
python test.py --config-name=conf_VCTK.yaml \
    tester=blind_dereverberation_BUDDy \
    tester.checkpoint=$CKPT_PATH \
    model_dir=$PATH_EXPERIMENT \
    +gpu=0 \
    dset=mydata \
    dset.test.path=$INPUT_AUDIO_DIR \
    dset.test.path_reverb=$INPUT_AUDIO_DIR

