# BUDDy - All Changes and improvements made:

This fork modifies the original BUDDy codebase to train and perform blind dereverberation on a custom dataset of clean and reverberant audio pairs.

RESULTS: [DRIVE LINK](https://drive.google.com/drive/folders/1UcCFWdqq8ra_6JKhabMavG_N_UU9Sn0d?usp=sharing)

## Summary of Changes

The key modifications made to adapt the code to a custom dataset are:

1.  **Dataset Configuration:** Swapped the VCTK dataset for a new `mydata` configuration that uses a generic `AudioDataset` class.
2.  **Custom Data Loader:** The new dataset expects pairs of audio files in `cleans` and `reverbs` directories.
3.  **Testing Script:** Updated `test_blind_dereverberation.sh` to be user-friendly, allowing easy specification of input and output paths.
4.  **Training Script:** Updated `train.sh` with the correct paths for the custom dataset.
5.  **Tester Logic:** Refactored `tester.py` to handle the new data format and focus solely on blind dereverberation.
6.  **Checkpoint Loading:** Added `weights_only=False` flag to `torch.load` for compatibility with older checkpoints.
7.  **Logging Frequency:** Reduced logging intervals (`log_interval`, `heavy_log_interval`, `save_interval`) to be more appropriate for smaller datasets.


## Improvements made

Simplified Data Loading (The Biggest Win for Usability)
Streamlined Testing Pipeline
Focused Functionality
Adjusted Logging for Practical Training

## Project Structure 

Your data directory should be organized as follows. The scripts are pre-configured to look for these paths:

```
/root/hack/data/
├── audio_16k/
│   ├── train/
│   │   ├── cleans/          # Place your clean training audio files here
│   │   └── reverbs/         # Place your reverberant training audio files here
│   └── valid/
│       ├── cleans/          # Place your clean validation audio files here
│       └── reverbs/         # Place your reverberant validation audio files here
├── buddy/                   # This project's code
└── buddy/experiments/       # Training outputs and checkpoints will be saved here
```

*   **File Correspondence:** For the training loop to work correctly, each clean file (e.g., `sample1.wav`) in `cleans` must have a corresponding reverberant file with the **same filename** (e.g., `sample1.wav`) in the `reverbs` directory.
*   **Audio Format:** All audio files are expected to be **16 kHz sampling rate**.

## Quick Start Guide

### 1. Installation & Setup

Ensure you are in the project's Python environment (`myenv`) and have all dependencies installed.

```bash
conda activate myenv
# Install requirements if not already done
# pip install -r requirements.txt
```

### 2. Training

To train a model on your custom dataset, simply run the training script. The paths are already configured in `train.sh`.

```bash
cd /root/hack/data/buddy
bash train.sh
```

**Key Training Parameters (you can modify these in `train.sh`):**
*   `exp.batch_size`: Set to `2` in the diff. Increase based on your GPU memory.

### 3. Running Blind Dereverberation (Testing)

To perform blind dereverberation on a set of reverberant audio files, use the provided script.

First, edit `test_blind_dereverberation.sh` to set your desired parameters:
```bash
# Edit these variables
CKPT_PATH="/root/hack/data/buddy/experiments/VCTK_16k_4s_time-10350.pt" # Path to your trained model checkpoint
INPUT_AUDIO_DIR="/root/hack/data/audio_16k/valid/reverbs" # Directory containing input reverberant files
EXPERIMENT_NAME="my_test_run_1" # A name for this test run
```

Then, execute the script:
```bash
bash test_blind_dereverberation.sh
```

**Outputs:**
The results will be saved in `experiments/$EXPERIMENT_NAME/`. For each input file, you will find:
*   `degraded/`: A copy of the input reverberant audio.
*   `reconstructed/`: The dereverberated (clean) output audio.
*   `estimated_rir/`: (If applicable) The estimated Room Impulse Response.

## Configuration Files

The changes are primarily reflected in these configuration files:
*   `conf/conf_VCTK.yaml`: Now points to the `mydata` dataset.
*   `conf/dset/mydata.yaml` (not shown in diff, but implied): This file should be created to define the `AudioDataset` parameters. The diff shows the replacement for `vctk_16k_4s.yaml`, which effectively serves as the new `mydata` config.

## Notes

*   **Checkpoint Compatibility:** The change to `torch.load(..., weights_only=False)` allows loading checkpoints from previous versions of PyTorch but may have security implications. Ensure you trust the source of your checkpoint files.
*   **Focus on Blind Dereverberation:** The `tester.py` file has been simplified to only support the `blind_dereverberation` mode, as this was the primary goal of the modification.
*   **W&B Logging:** The WandB configuration entity is commented out in `train.sh`. You will need to uncomment and set it to your own username/team to log metrics to Weights & Biases.

# BUDDy: Single-channel Blind Unsupervised Dereverberation with Diffusion Models #

This is the official code for our paper [BUDDy: Single-channel Blind Unsupervised Dereverberation with Diffusion Models](https://arxiv.org/abs/2405.04272)

We invite you to check our [companion website](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/iwaenc2024-buddy.html) for listening samples and insights into the paper

## 1 - Requirements

Install required Python packages with `python -m pip install -r requirements.txt`

## 2 - Checkpoints

You can access our pretrained checkpoint, trained on VCTK anechoic speech, at [the following link](https://drive.google.com/drive/u/2/folders/1fEvzbiIy77A1i5aiOwPf78OKQjCemOmQ)

## 3 - Testing

You can launch blind dereverberation with `bash test_blind_dereverberation.sh`.
You can launch informed dereverberation with `bash test_informed_dereverberation.sh`.
In both cases, do not forget to add the path to the pretrained model checkpoint in the bash file (i.e. replace `ckpt=<pretrained-vctk-checkpoint.pt>` with your path).
The directory tree in `audio_examples/` contains an example test set to reproduce the results.  

## 4 - Training

You can retrain an unsupervised diffusion model on your own dataset with `bash train.sh`.
Do not forget to fill in the path to your training and testing dataset (i.e. replace `dset.train.path=/your/path/to/anechoic/training/set` and so on)

## 5 - Citing

If you used this repo for your own work, do not forget to cite us:

@bibtex
```
@article{moliner2024buddy,
    title={{BUDD}y: Single-channel Blind Unsupervised Dereverberation with Diffusion Models},
    author={Moliner, Eloi and Lemercier, Jean-Marie and Welker, Simon and Gerkmann, Timo and V\"alim\"aki, Vesa},
    year={2024},
    journal={arXiv 2405.04272}
}
```
