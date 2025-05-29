#
# This file is to download dataset from Huggingface for piper training/finetuning
#

import os
import csv
import shutil
from pathlib import Path 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import soundfile as sf
import datasets as hugDS

SAMPLING_RATE = 22050

def load_my_data(mode, **kwargs):
  tmp = hugDS.load_dataset(**kwargs, trust_remote_code=True).cast_column("audio", hugDS.Audio(sampling_rate=SAMPLING_RATE))
  match mode:
    case 0:
      return tmp
    case 1:
      return tmp.select_columns(["audio", "transcription"])
    case 2:
      return tmp.select_columns(["audio", "sentence"]).rename_column("sentence", "transcription")
    case _:
      raise ValueError("oh no!")

# data_1 = load_my_data(path="google/fleurs", name="vi_vn", split="train", mode=1)

MY_DATA = hugDS.IterableDatasetDict()
MY_DATA = hugDS.concatenate_datasets([
  # load_my_data(path="google/fleurs", name="vi_vn",           split="train", mode=1),    # 3k
  # load_my_data(path="vivos",                                 split="train", mode=2),    # 11.7k
  # load_my_data(path="doof-ferb/fpt_fosd",                    split="train", mode=0),    # 25.9k
  # load_my_data(path="doof-ferb/infore1_25hours",               split="train", mode=0),  # 14.9k
  # load_my_data(path="doof-ferb/vlsp2020_vinai_100h",           split="train", mode=0),  # 56.4k
  load_my_data(path="NhutP/VietSpeech",                        split="train", mode=0),    # 1.03m
  # load_my_data(path="doof-ferb/LSVSC",                       split="train", mode=1),    # 45k
  # load_my_data(path="quocanh34/viet_vlsp",                   split="train", mode=0),    # 171k
  # load_my_data(path="linhtran92/viet_youtube_asr_corpus_v2", split="train", mode=1),    # 195k
  # load_my_data(path="doof-ferb/infore2_audiobooks",          split="train", mode=0),    # 315k
  # load_my_data(path="linhtran92/viet_bud500",                split="train", mode=0),    # 634k
])

output_dir = Path("data/VietSpeech")
if os.path.exists(output_dir) and os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

wavs_dir = output_dir / "wavs"
wavs_dir.mkdir(parents=True, exist_ok=True)

metadata_file = output_dir / "metadata.csv"

def process_item(idx, item):
    audio = item["audio"]
    transcription = item["transcription"]

    # Save audio
    audio_name = f"audio{idx}"
    audio_filename = f"{audio_name}.wav"
    audio_path = wavs_dir / audio_filename
    sf.write(audio_path, audio["array"], SAMPLING_RATE)

    # Return metadata entry
    # return f"wavs/{audio_name}.wav|{transcription}|@X\n"
    # Return metadata entry as a tuple
    return (f"wavs/{audio_name}.wav", transcription)

with metadata_file.open("w", encoding='utf-8', newline='') as f, ThreadPoolExecutor() as executor:
    writer = csv.writer(f)
    writer = csv.writer(f, delimiter='|')

    writer.writerow(["audio_file", "text"])

    futures = []
    idx = 1

    # Submit each item for processing
    for item in tqdm(MY_DATA):
        futures.append(executor.submit(process_item, idx, item))
        idx += 1

    # Write the results as they are completed
    for future in as_completed(futures):
        result = future.result()
        writer.writerow(result)  # Assuming result is a tuple or list

    print(f"Finished exporting data to {metadata_file}")