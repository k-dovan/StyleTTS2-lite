import librosa
import numpy as np
import soundfile as sf
import os
import argparse
import csv
from scipy.signal import welch
from pathlib import Path
import tempfile
import tensorflow_hub as hub
import tensorflow as tf
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torchaudio.sox_effects as sox_effects
import demucs.separate
from demucs.pretrained import get_model
from demucs.apply import apply_model
import shutil
from pyannote.audio import Pipeline
from pydub import AudioSegment
from uuid import uuid4
from tqdm import tqdm

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

# Energy threshold (adjust as needed)
ENERGY_THRESHOLD = 0.01
PERCENTAGE_THRESH = 70

def compute_rms(y):
    return np.sqrt(np.mean(np.square(y)))

def is_enough_energy(vad_result, full_audio):
    # Filtered output
    enough_energy = 0
    total = 0
    for segment in vad_result.itersegments():
        total += 1
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        segment_audio = full_audio[start_ms:end_ms]
        
        # Convert to numpy for RMS energy calculation
        samples = np.array(segment_audio.get_array_of_samples()).astype(np.float32) / 2**15
        energy = compute_rms(samples)
        
        print (f"Energy of segment: {energy}")

        if energy > ENERGY_THRESHOLD:
            enough_energy += 1

    if total == 0:
        return False
    else:
        percent = 100*enough_energy/total
        print (f"percentage of enough energy semgents: {percent}")
        if percent > PERCENTAGE_THRESH:
            return True
        else:
            return False

def is_speech(audio_path):
    """Check if audio is effectively silent based on energy threshold."""
    
    vad_result = pipeline(audio_path)   
     
    full_audio = AudioSegment.from_mp3(audio_path)
    
    is_enough = is_enough_energy(vad_result, full_audio)
    
    return is_enough

def classify_sound(audio, sr, input_path="", score_threshold=0.5):
    """Detect if audio contains speech, child speech, or conversation using YAMNet."""
    try:
        # Load YAMNet model
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Ensure audio is mono and resampled to 16kHz (YAMNet's expected input)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Prepare audio for YAMNet (expects float32, normalized to [-1, 1])
        audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio))
        
        assert audio.ndim == 1, "Input must be mono (1D)"
        assert np.any(audio), "Audio is completely silent or empty"
        assert not np.isnan(audio).any(), "audio contains NaNs"
        
        # Run YAMNet inference
        scores, embeddings, spectrogram = yamnet_model(audio)
        
        # Load YAMNet class labels (from the 521 classes)
        class_map_path = yamnet_model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)
        
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        infered_class = class_names[scores_np.mean(axis=0).argmax()]
        print(f'The main sound is: {infered_class}')

        speech_related_classes = [
            'Speech', 'Conversation', 'Child speech, kid speaking', 'Narration, monologue']
        
        # Consider audio as speech
        is_speech = (infered_class in speech_related_classes)
        
        return is_speech
    except Exception as e:
        print(f"Error detecting speech for {input_path}: {e}")
        return False

def clean_audio(input_path, output_dir, source="vocals"):
    """Apply Demucs source separation to remove noise/background and return the specified source."""
    # Create a temporary directory for Demucs outputs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Run Demucs to separate sources
        demucs.separate.main(["--mp3", "-o", temp_dir, input_path])
        
        # Construct path to the desired source (e.g., vocals.mp3)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        source_path = os.path.join(temp_dir, "htdemucs", base_name, f"{source}.mp3")
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Demucs failed to generate {source} for {input_path}")
        
        # Load the separated source audio
        y, sr = librosa.load(source_path, sr=None)
        
        if not is_speech(source_path):
            return None, None
        else:       
            return y, sr
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_audio_directory(raw_dir, output_dir, min_score=0, source="vocals"):
    """Process all audio files in raw_dir, check for speech using YAMNet, clean with Demucs if speech, and save to output_dir if quality score exceeds min_score."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported audio extensions
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    audio_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(audio_extensions)]
    
    if not audio_files:
        print(f"No audio files found in {raw_dir}")
        return [], []
    
    cleaned_paths = []
    stat_data = []
    
    for audio_file in tqdm(audio_files, "Preprocess audios"):
        input_path = os.path.join(raw_dir, audio_file)
        output_path = os.path.join(output_dir, audio_file)
        
        print(f"Processing {audio_file}...")
        try:
            # Load raw audio for speech detection
            y_raw, sr = librosa.load(input_path, sr=None)
            
            # Check if the raw audio contains speech
            if not classify_sound(y_raw, sr, input_path=input_path):
                print(f"Skipped {audio_file}: No speech detected")
                stat_data.append((audio_file, input_path, False))
                continue
            
            # Clean audio using Demucs only if speech is detected
            y_clean, sr = clean_audio(input_path, output_dir, source=source)
            
            if y_clean is not None:   
                stat_data.append((audio_file, input_path, True))
                
                sf.write(output_path, y_clean, sr)
                cleaned_paths.append(output_path)
                print(f"Cleaned {source} audio saved to {output_path}")
            else:
                stat_data.append((audio_file, input_path, False))
                cleaned_paths.append(output_path)
                print(f"Skipped saving {audio_file}: audio likely silent, then ignore it.")
                 
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            stat_data.append((audio_file, input_path, False))
    
    # Write csv
    preprocessed_metadata = os.path.join(output_dir, f"{str(uuid4())}.csv")
    with open(preprocessed_metadata, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["File", "Path", "Is Kept"])
        
        for filename, input_path, is_speech in stat_data:
            writer.writerow([filename, input_path, "Yes" if is_speech else "No"])
    
    print(f"\nMetadata written to {preprocessed_metadata}")
    return cleaned_paths, stat_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clean audio files containing speech using Demucs source separation, after detecting speech with YAMNet, and rank based on quality.")
    parser.add_argument("--raw_dir", default='~/data/VietSpeech/raw_audio', help="Directory containing raw audio files")
    parser.add_argument("--output_dir", default='data_processing/cleaned_audio', help="Directory to save cleaned audio files")
    args = parser.parse_args()
    
    # Process all audio files
    cleaned_paths, stat_data = process_audio_directory(args.raw_dir, args.output_dir)
    
    if stat_data:
        for i, (filename, input_path, is_speech) in enumerate(stat_data, 1):
            print(f"{i}. {filename}: Is Kept={'Yes' if is_speech else 'No'}")

if __name__ == "__main__":
    main()
