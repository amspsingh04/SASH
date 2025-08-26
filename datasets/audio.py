import os
import numpy as np
import torch
import random
import glob
import soundfile as sf

class AudioTrainDataset(torch.utils.data.IterableDataset):
    """
    Generic IterableDataset for paired clean and reverberant audio.
    Finds all .wav files in the clean path, and assumes a corresponding
    file with the same name exists in the reverb path.
    """
    def __init__(self,
                 fs=16000,
                 segment_length=65536,
                 path="",             # Path to the clean audio directory
                 path_reverb="",      # Path to the reverberant audio directory
                 normalize=False,
                 seed=0):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        # Find all clean audio files
        self.clean_samples = glob.glob(os.path.join(path, "**/*.wav"), recursive=True)
        assert len(self.clean_samples) > 0, f"No .wav files found in {path}"
        
        self.path_reverb = path_reverb
        self.segment_length = int(segment_length)
        self.fs = fs
        self.normalize = normalize

        if self.normalize:
            raise NotImplementedError("Normalization not implemented yet")

    def __iter__(self):
        while True:
            # 1. Select a random clean audio file
            clean_file_path = random.choice(self.clean_samples)
            
            # 2. Construct the corresponding reverberant audio file path
            #    This assumes the directory structure is the only difference.
            #    e.g., /path/to/train/cleans/file1.wav -> /path/to/train/reverbs/file1.wav
            relative_path = os.path.relpath(clean_file_path, start=os.path.dirname(os.path.dirname(clean_file_path)))
            reverb_file_path = os.path.join(self.path_reverb, relative_path.split(os.sep, 1)[1])

            if not os.path.exists(reverb_file_path):
                # Skip if the paired file doesn't exist
                continue

            # 3. Load both audio files
            clean_data, samplerate_clean = sf.read(clean_file_path)
            reverb_data, samplerate_reverb = sf.read(reverb_file_path)

            assert samplerate_clean == self.fs, f"Wrong sampling rate for {clean_file_path}"
            assert samplerate_reverb == self.fs, f"Wrong sampling rate for {reverb_file_path}"

            # 4. Process and return a segment
            # Combine into a stereo signal [clean, reverb]
            # This assumes both files are mono and have the same length
            if len(clean_data.shape) > 1: clean_data = np.mean(clean_data, axis=1)
            if len(reverb_data.shape) > 1: reverb_data = np.mean(reverb_data, axis=1)

            min_len = min(len(clean_data), len(reverb_data))
            
            # Create a 2-channel array
            paired_data = np.vstack([clean_data[:min_len], reverb_data[:min_len]]).T

            L = len(paired_data)

            # Crop or pad to get to the right length
            if L > self.segment_length:
                idx = np.random.randint(0, L - self.segment_length)
                segment = paired_data[idx:idx + self.segment_length]
            else:
                # Pad with zeros (or wrap)
                segment = np.pad(paired_data, ((0, self.segment_length - L), (0, 0)), 'constant')

            # The model likely expects a dictionary with specific keys
            yield {"reference": segment[:, 0], "degraded": segment[:, 1]}


class AudioTestDataset(torch.utils.data.Dataset):
    """
    Generic Dataset for paired clean and reverberant audio for validation/testing.
    """
    def __init__(self,
                 fs=16000,
                 segment_length=65536,
                 path="",             # Path to the clean audio directory
                 path_reverb="",      # Path to the reverberant audio directory
                 normalize=False,
                 num_examples=-1,
                 seed=0):
        super().__init__()
        random.seed(seed)
        
        self.clean_samples = sorted(glob.glob(os.path.join(path, "**/*.wav"), recursive=True))
        assert len(self.clean_samples) > 0, f"No .wav files found in {path}"

        if num_examples > 0:
            self.clean_samples = random.sample(self.clean_samples, num_examples)

        self.path_reverb = path_reverb
        self.segment_length = int(segment_length)
        self.fs = fs
        self.normalize = normalize

    def __len__(self):
        return len(self.clean_samples)

    def __getitem__(self, idx):
        clean_file_path = self.clean_samples[idx]
        
        relative_path = os.path.relpath(clean_file_path, start=os.path.dirname(os.path.dirname(clean_file_path)))
        reverb_file_path = os.path.join(self.path_reverb, relative_path.split(os.sep, 1)[1])

        clean_data, _ = sf.read(clean_file_path)
        reverb_data, _ = sf.read(reverb_file_path)

        if len(clean_data.shape) > 1: clean_data = np.mean(clean_data, axis=1)
        if len(reverb_data.shape) > 1: reverb_data = np.mean(reverb_data, axis=1)

        min_len = min(len(clean_data), len(reverb_data))
        paired_data = np.vstack([clean_data[:min_len], reverb_data[:min_len]]).T
        
        L = len(paired_data)
        if self.segment_length > 0:
            if L > self.segment_length:
                segment = paired_data[:self.segment_length]
            else:
                segment = np.pad(paired_data, ((0, self.segment_length - L), (0, 0)), 'constant')
        else:
            segment = paired_data

        filename = os.path.basename(clean_file_path)
        return {"reference": segment[:, 0], "degraded": segment[:, 1], "filename": filename}

