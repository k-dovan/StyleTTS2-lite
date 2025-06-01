import os
import csv
import pandas as pd
from time import time
from multiprocessing import Pool
from tqdm import tqdm
from faster_whisper import WhisperModel
import openai

# === CONFIG ===
dict_audio_dir = {
    0: "data_processing/samples/staging",
    1: "data_processing/samples/tested"
}
NUM_WORKERS = len(dict_audio_dir)
BASE_DIR = "data_processing/samples"
METADATA_PATH = f"{BASE_DIR}/metadata.csv"
MODEL_SIZE = "tiny"
DEVICE = "cuda"  # or "cpu"

# Load metadata once, share as global for workers
df = pd.read_csv(METADATA_PATH, sep="|")
ref_dict_global = {os.path.basename(row['audio_file']): row['text'] for _, row in df.iterrows()}

def correct_transcription(client, stt_text, ref_text):
    prompt = f"""
You are a text correction assistant for speech transcription.
Given:
- Speech-to-text output (possibly noisy transcription): "{stt_text}"
- Reference text (high-confidence): "{ref_text}"

Please produce a corrected, punctuated, natural Vietnamese transcription,
using the reference text as the primary source but
considering and reasoning about the Speech-to-tex output as well. Remove quotation marks if allowed.

Return only the corrected transcription.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        corrected = response.choices[0].message.content.strip()
        return corrected
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return stt_text  # fallback

def worker(worker_id):
    AUDIO_DIR = dict_audio_dir[worker_id]
    print(f"[Worker {worker_id}] Loading model...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")
    print(f"[Worker {worker_id}] Model loaded.")

    # Setup OpenAI client inside worker
    client = openai.OpenAI()

    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
    print(f"[Worker {worker_id}] {len(wav_files)} audio files found.")

    output_csv = f"{BASE_DIR}/corrected_worker_{worker_id+1}.csv"
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow(["audio_file", "corrected_transcription"])

    for wav_file in tqdm(wav_files, desc=f"Worker {worker_id+1} processing"):
        t1 = time()
        audio_path = os.path.join(AUDIO_DIR, wav_file)
        ref_text = ref_dict_global.get(wav_file)

        # Transcribe
        segments, _ = model.transcribe(audio_path)
        stt_text = ".".join(segment.text for segment in segments).strip()

        if ref_text:
            corrected = correct_transcription(client, stt_text, ref_text)
        else:
            print(f"[Worker {worker_id}] No reference for {wav_file}, using raw STT output.")
            corrected = stt_text

        t2 = time()

        with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow([wav_file, corrected])

        print(f"[Worker {worker_id}] {wav_file} done in {t2-t1:.2f}s")

    return f"[Worker {worker_id}] Finished processing {len(wav_files)} files."

if __name__ == "__main__":
    with Pool(NUM_WORKERS) as pool:
        results = list(pool.imap(worker, range(NUM_WORKERS)))

    for res in results:
        print(res)
