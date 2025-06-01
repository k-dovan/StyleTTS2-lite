import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tqdm import tqdm
import argparse

import librosa
import speechmetrics
import pandas as pd
import numpy as np

import torch
from torchaudio.backend.common import AudioMetaData

from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample

import multiprocessing
from multiprocessing import current_process
from functools import partial

class Data_Processing:
    def __init__(self, target_path, output_path, model_path, device):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/metrics", exist_ok=True)
        self.target_path = target_path
        self.output_path = output_path
        self.device = device

        self.model, self.df, _ = init_df(model_path, config_allow_defaults=True)
        self.model = self.model.to(device=device).eval()
        self.metrics = speechmetrics.load(['absolute.mosnet', 'absolute.srmr'])

    # Detect distort audio 
    def rms_jump(self, audio):
        rms = librosa.feature.rms(y=audio)[0]
        rms_diff = np.diff(rms)
        max_jump = np.max(np.abs(rms_diff))
        return max_jump

    # Quantify how much noise-like a sound is, as opposed to being tone-like
    def spectral_flatness(self, audio):
        flatness = librosa.feature.spectral_flatness(y=audio)
        return np.mean(flatness)

    # Trim silent
    def trim_silent(self, audio: torch.Tensor, top_db: float = 40.0) -> torch.Tensor:
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio

    # Evaluate audio
    def eval_audio(self, audio,
        filename: str, output_csv: str,
        sample_rate: int = 24000
    ):
        try:
            if audio.ndim > 1:
                audio = audio[0]
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            if len(audio) < 16000:
                raise ValueError("Audio too short: must be at least 1 second (16000 samples at 16kHz).")

            result = self.metrics(audio, rate=sample_rate)
            result['mosnet'] = result['mosnet'].mean()
            result['srmr'] = result['srmr'].mean()
            result['flatness'] = self.spectral_flatness(audio)
            result['rms'] = self.rms_jump(audio)
            row = {'filename': filename}
            row.update(result)
        except Exception as e:
            print(f"Error evaluating audio for {filename}: {e}")
            return

        # Load existing CSV if exists
        if os.path.exists(output_csv):
            try:
                df_existing = pd.read_csv(output_csv, sep='|')
                if not isinstance(df_existing, pd.DataFrame):
                    df_existing = pd.DataFrame()
            except pd.errors.EmptyDataError:
                df_existing = pd.DataFrame()
        else:
            df_existing = pd.DataFrame()

        if not df_existing.empty and 'filename' in df_existing.columns:
            if filename in df_existing['filename'].values:
                print(f"File '{filename}' already in CSV. Skipping.")
                return

        df = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]
        df.to_csv(output_csv, sep='|', index=False)

    def process_files(self, wav_files):
        current = current_process()
        for wav_file in tqdm(wav_files, leave=True, position=current._identity[0] - 1, desc=f"{current.name} - Denoising audio"):
            sr = config("sr", 24000, int, section="df")
            audio, meta = load_audio(wav_file.resolve(), sr)

            enhanced = enhance(self.model, self.df, audio)
            fade_in = torch.linspace(0.0, 1.0, int(sr * 0.15)).unsqueeze(0)
            fade = torch.cat((fade_in, torch.ones(1, enhanced.shape[1] - fade_in.shape[1])), dim=1)
            enhanced *= fade

            if meta.sample_rate != sr:
                enhanced = resample(enhanced, sr, meta.sample_rate)
                sr = meta.sample_rate

            self.eval_audio(enhanced.numpy(), filename=wav_file.name, sample_rate=sr, output_csv=f"{self.output_path}/metrics/{current.name}_denoise_audio_metrics.csv")
            enhanced = self.trim_silent(enhanced)
            save_audio(os.path.join(self.output_path, wav_file.name), enhanced, sr)

    def denoise(self):
        wav_dir = Path(self.target_path)
        wav_files = list(wav_dir.glob("*.wav"))

        num_workers = min(multiprocessing.cpu_count(), 6)
        chunk_size = (len(wav_files) + num_workers - 1) // num_workers
        chunks = [wav_files[i:i + chunk_size] for i in range(0, len(wav_files), chunk_size)]

        with multiprocessing.Pool(num_workers) as pool:
            pool.map(partial(worker_process, target_path=self.target_path, output_path=self.output_path, model_path="DeepFilterNet2", device=self.device), chunks)

def worker_process(wav_files, target_path, output_path, model_path, device):
    dp = Data_Processing(target_path=target_path, output_path=output_path, model_path=model_path, device=device)
    dp.process_files(wav_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean audio files using DeepFilterNet2.")
    parser.add_argument("--raw_dir", default='data_processing/raw_audio', help="Directory containing raw audio files")
    parser.add_argument("--output_dir", default='data_processing/cleaned_audio', help="Directory to save cleaned audio files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data_Processing(target_path=args.raw_dir,
                        output_path=args.output_dir,
                        model_path="DeepFilterNet2",
                        device=device)
    data.denoise()