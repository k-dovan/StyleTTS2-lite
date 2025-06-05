from datasets import load_dataset

dataset = load_dataset("khachuongvn/VietSpeech_voice_cleaned_text_punctuated")  # e.g., "imdb"
dataset.save_to_disk("/home/ubuntu/data/VietSpeech/processed/20250604")
