#!/bin/bash

#!/bin/bash

# These lines help with debugging if something goes wrong.
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

# --- Training Command ---
# We use a '+' to add the reverb paths to the configuration,
# since they don't exist in the original 'mydata.yaml' file.

echo "Starting training..."

python train.py --config-name=conf_VCTK.yaml \
    dset.train.path=/root/hack/data/audio_16k/train/cleans \
    dset.train.path_reverb=/root/hack/data/audio_16k/train/reverbs \
    dset.test.path=/root/hack/data/audio_16k/valid/cleans \
    dset.test.path_reverb=/root/hack/data/audio_16k/valid/reverbs \
    exp.batch_size=2
