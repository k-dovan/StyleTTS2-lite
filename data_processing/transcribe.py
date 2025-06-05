import os
import csv
import glob
import math
import re
import pandas as pd
from time import time
from multiprocessing import Pool
from tqdm import tqdm
from faster_whisper import WhisperModel

from data_processing.correctors.openai_corrector import OpenAICorrector
from data_processing.correctors.ollama_corrector import OllamaCorrector

# === CONFIG ===
BASE_DIR = "/home/ubuntu/data/VietSpeech"
AUDIO_DIR = os.path.join(BASE_DIR, "wavs")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
MODEL_SIZE = "base"
DEVICE = "cuda"  # or "cpu"
NUM_WORKERS = 20
CORRECTOR_TYPE = "azure"  # "openai", "azure", or "ollama"

# === Load metadata globally ===
print ("Loading metadata...")
df = pd.read_csv(METADATA_PATH, sep="|")
ref_dict_global = {os.path.basename(row["audio_file"]): row["text"] for _, row in df.iterrows()}
print ("Finished")

def get_corrector(corrector_type):
    if corrector_type == "openai":
        import openai
        
        OPENAI_MODEL = "gpt-4o-mini"
        client = openai.OpenAI()
        return OpenAICorrector(model=OPENAI_MODEL, client=client)
    elif corrector_type == "azure":
        from openai import AzureOpenAI
        
        endpoint = "https://nlp-team-ttstts.openai.azure.com/"        
        OPENAI_MODEL = "gpt-4o-mini"
        api_version = "2024-12-01-preview"        
        subscription_key = os.getenv("AZURE_API_KEY")
        assert subscription_key, "Azure API KEY must be set!"
        
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        return OpenAICorrector(model=OPENAI_MODEL, client=client)
    elif corrector_type == "ollama":
        
        OLLAMA_MODEL = "gemma3:4b"
        
        return OllamaCorrector(model=OLLAMA_MODEL)
    else:
        raise ValueError("Unsupported corrector type")

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def split_wav_files(wav_files, num_chunks):
    chunk_size = math.ceil(len(wav_files) / num_chunks)
    return [wav_files[i:i + chunk_size] for i in range(0, len(wav_files), chunk_size)]


def worker(args):
    worker_id, wav_files_chunk = args
    print(f"[Worker {worker_id}] Loading Whisper model...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")
    print(f"[Worker {worker_id}] Model loaded.")

    corrector = get_corrector(CORRECTOR_TYPE)

    output_csv = os.path.join(BASE_DIR, f"corrected_worker_{worker_id+1}.csv")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow(["audio_file", "corrected_transcription"])

    for wav_file in tqdm(wav_files_chunk, desc=f"Worker {worker_id+1} processing"):
        t1 = time()
        audio_path = os.path.join(AUDIO_DIR, wav_file)
        ref_text = ref_dict_global.get(wav_file)

        try:
            segments, _ = model.transcribe(audio_path)
            stt_text = ". ".join(segment.text for segment in segments).strip()
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR transcribing {wav_file}: {e}")
            stt_text = ""

        if ref_text:
            corrected = corrector.correct(stt_text, ref_text).lower()
        else:
            print(f"[Worker {worker_id}] No reference for {wav_file}, using raw STT.")
            corrected = stt_text.lower()

        with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow([wav_file, corrected])

        t2 = time()
        print(f"[Worker {worker_id}] {wav_file} done in {t2 - t1:.2f}s")

    return f"[Worker {worker_id}] Finished {len(wav_files_chunk)} files."


if __name__ == "__main__":

    # === List & split wav files ===
    all_wav_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")])
    chunks = split_wav_files(all_wav_files, NUM_WORKERS)
    args_list = [(i, chunk) for i, chunk in enumerate(chunks)]

    # === Run workers ===
    with Pool(NUM_WORKERS) as pool:        
        t_start = time()
        results = list(pool.imap(worker, args_list))

    t_end = time()
    print(f"\nTotal inference time: {t_end - t_start:.2f}s")
    
    for res in results:
        print(res)


    # === Merge all corrected_worker_*.csv files ===
    merged_output = os.path.join(BASE_DIR, "metadata_corrected.csv")
    all_rows = []

    corrected_files = sorted(glob.glob(os.path.join(BASE_DIR, "corrected_worker_*.csv")))
    for file in corrected_files:
        with open(file, newline='', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='|')
            next(reader)  # skip header
            all_rows.extend(list(reader))

    # Sort merged rows by numeric order of audio file names
    all_rows.sort(key=lambda row: extract_number(row[0]))

    with open(merged_output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(["audio_file", "corrected_transcription"])
        writer.writerows(all_rows)

    print(f"[INFO] Merged and sorted output saved to: {merged_output}")

    # === Cleanup individual worker files ===
    for file in corrected_files:
        try:
            os.remove(file)
            print(f"[INFO] Deleted: {file}")
        except Exception as e:
            print(f"[WARN] Could not delete {file}: {e}")
