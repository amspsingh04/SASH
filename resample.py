import os
import glob
import librosa
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Set the target sample rate the model expects.
TARGET_SR = 16000

# 2. Set the paths to your original data folders.
INPUT_DIRS = [
    "/root/hack/data/audio/train/cleans",
    "/root/hack/data/audio/train/reverbs",
    "/root/hack/data/audio/valid/cleans",
    "/root/hack/data/audio/valid/reverbs",
]

# 3. Set a path to a NEW folder where the converted files will be saved.
#    The script will recreate the original folder structure inside this directory.
OUTPUT_BASE_DIR = "/root/hack/data/audio_16k"

# --- SCRIPT ---
def resample_directory(input_dir, output_dir, target_sr):
    """
    Finds all .wav files in input_dir, resamples them, and saves them to output_dir,
    preserving the subdirectory structure.
    """
    print(f"--- Processing directory: {input_dir} ---")
    
    # Find all .wav files, including those in subdirectories
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    
    if not wav_files:
        print(f"Warning: No .wav files found in {input_dir}")
        return

    for original_path in tqdm(wav_files, desc=f"Resampling {os.path.basename(input_dir)}"):
        try:
            # Load audio file and resample it to the target rate
            audio, sr = librosa.load(original_path, sr=target_sr, mono=True)
            
            # Create the corresponding output path
            relative_path = os.path.relpath(original_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the resampled audio
            sf.write(output_path, audio, target_sr)

        except Exception as e:
            print(f"Error processing {original_path}: {e}")

if __name__ == "__main__":
    print(f"Starting resampling process. Target sample rate: {TARGET_SR} Hz.")
    
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)
        print(f"Created output base directory: {OUTPUT_BASE_DIR}")

    for in_dir in INPUT_DIRS:
        # Create a corresponding output directory inside the base output folder
        # e.g., /root/hack/data/audio/train/cleans -> /root/hack/data/audio_16k/train/cleans
        base_name = os.path.basename(os.path.dirname(in_dir)) # 'train' or 'valid'
        type_name = os.path.basename(in_dir) # 'cleans' or 'reverbs'
        out_dir = os.path.join(OUTPUT_BASE_DIR, base_name, type_name)
        
        if not os.path.exists(in_dir):
            print(f"Warning: Input directory does not exist, skipping: {in_dir}")
            continue

        resample_directory(in_dir, out_dir, TARGET_SR)

    print("\nResampling complete!")
    print(f"All converted files are saved in: {OUTPUT_BASE_DIR}")
