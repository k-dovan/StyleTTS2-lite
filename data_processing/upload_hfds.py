from datasets import Dataset, Audio
import pandas as pd
import os

DS_DIR = "/home/ubuntu/data/VietSpeech/processed/20250604"

# Load metadata
df = pd.read_csv(f"{DS_DIR}/metadata.csv", sep="|", names=["file", "transcription"])
df["audio"] = f"{DS_DIR}/wavs/" + df["file"]  # Path to wavs

# Convert to Hugging Face Dataset
ds = Dataset.from_pandas(df)
ds = ds.cast_column("audio", Audio())

# Save to disk
ds.push_to_hub("khachuongvn/VietSpeech_voice_cleaned_text_punctuated")
