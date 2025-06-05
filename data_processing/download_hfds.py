from huggingface_hub import list_repo_files, hf_hub_download
import os
from tqdm import tqdm

# Replace with your dataset repo ID (e.g., "username/dataset-name")
repo_id = "khachuongvn/VietSpeech_voice_cleaned_text_punctuated"  # <-- change this
repo_type = "dataset"  # or "model" for model repos
local_dir = "/home/ubuntu/data/VietSpeech/processed/20250604"

# Create local directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# List all files in the dataset repository
all_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Download each file
for file in tqdm(all_files):
    print(f"Downloading {file}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Set to True if you prefer symlinks
    )
