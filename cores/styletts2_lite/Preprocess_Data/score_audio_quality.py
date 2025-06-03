import librosa
import numpy as np
import soundfile as sf
import os
import demucs.separate
import argparse
import csv
from scipy.signal import welch
from pathlib import Path
import tempfile
import shutil
import tensorflow_hub as hub
import tensorflow as tf

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

def detect_speech(audio, sr, input_path="", score_threshold=0.5):
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
        
        return y, sr
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def compute_snr(audio, sr):
    """Compute Signal-to-Noise Ratio (SNR) for an audio signal."""
    signal_power = np.mean(audio ** 2)
    # Estimate noise from low-energy frames
    stft = np.abs(librosa.stft(audio))
    energy = np.mean(stft ** 2, axis=0)
    noise_frames = stft[:, energy < np.percentile(energy, 10)]
    noise_power = np.mean(noise_frames ** 2) if noise_frames.size > 0 else 1e-10
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_spectral_flatness(audio, sr):
    """Compute spectral flatness for an audio signal."""
    freqs, psd = welch(audio, fs=sr)
    psd = np.clip(psd, 1e-10, None)
    geom_mean = np.exp(np.mean(np.log(psd)))
    arith_mean = np.mean(psd)
    flatness = geom_mean / arith_mean
    return flatness

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
    all_scores = []
    
    for audio_file in audio_files:
        input_path = os.path.join(raw_dir, audio_file)
        output_path = os.path.join(output_dir, audio_file)
        
        print(f"Processing {audio_file}...")
        try:
            # Load raw audio for speech detection
            y_raw, sr = librosa.load(input_path, sr=None)
            
            # Check if the raw audio contains speech
            if not detect_speech(y_raw, sr, input_path=input_path):
                print(f"Skipped {audio_file}: No speech detected")
                all_scores.append((audio_file, input_path, 0.0, 0.0, 0.0, False))
                continue
            
            # Clean audio using Demucs only if speech is detected
            y_clean, sr = clean_audio(input_path, output_dir, source=source)
            
            # Compute score
            snr = compute_snr(y_clean, sr)
            flatness = compute_spectral_flatness(y_clean, sr)
            score = snr - 50 * flatness
            
            # Store score for ranking
            all_scores.append((audio_file, input_path, score, snr, flatness, True))
            
            # Save only if score exceeds min_score
            if score >= min_score:
                sf.write(output_path, y_clean, sr)
                cleaned_paths.append(output_path)
                print(f"Cleaned {source} audio saved to {output_path} (Score: {score:.2f})")
            else:
                print(f"Skipped saving {audio_file}: Score {score:.2f} below threshold {min_score}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            all_scores.append((audio_file, input_path, 0.0, 0.0, 0.0, False))
    
    # Write quality_scores.csv
    quality_file = os.path.join(output_dir, "quality_scores.csv")
    with open(quality_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["File", "Path", "QualityScore", "SNR (dB)", "Flatness", "Is Speech", "Status"])
        # Write scores for filtered files
        for filename, input_path, quality_score, snr, flatness, is_speech in sorted(all_scores, key=lambda x: x[2], reverse=True):
            status = ("Saved" if os.path.join(output_dir, filename) in cleaned_paths 
                      else "Skipped (no speech)" if not is_speech 
                      else "Skipped (low quality score)")
            writer.writerow([filename, input_path, f"{quality_score:.3f}", f"{snr:.2f}", f"{flatness:.4f}", 
                            "Yes" if is_speech else "No", status])
    
    print(f"\nQuality scores written to {quality_file}")
    return cleaned_paths, all_scores

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clean audio files containing speech using Demucs source separation, after detecting speech with YAMNet, and rank based on quality.")
    parser.add_argument("--raw_dir", help="Directory containing raw audio files")
    parser.add_argument("--output_dir", help="Directory to save cleaned audio files")
    parser.add_argument("--min_score", type=float, default=0, help="Minimum quality score to save cleaned audio (default: 0)")
    parser.add_argument("--source", default="vocals", choices=["vocals", "drums", "bass", "other"],
                        help="Demucs source to extract (default: vocals)")
    args = parser.parse_args()
    
    # Process all audio files
    cleaned_paths, all_scores = process_audio_directory(args.raw_dir, args.output_dir, args.min_score, args.source)
    
    if all_scores:
        # Rank all processed files
        print("\nRanking all processed audio files...")
        ranked_audios = sorted(all_scores, key=lambda x: x[2], reverse=True)
        
        # Print rankings
        print("\nRanked Audio Files (Higher score = better quality/smoothness):")
        for i, (filename, input_path, score, snr, flatness, is_speech) in enumerate(ranked_audios, 1):
            status = ("Saved" if os.path.join(args.output_dir, filename) in cleaned_paths 
                      else "Skipped (no speech)" if not is_speech 
                      else "Skipped (low quality score)")
            print(f"{i}. {filename}: Score={score:.2f}, SNR={snr:.2f} dB, Flatness={flatness:.4f}, Is Speech={'Yes' if is_speech else 'No'} ({status})")

if __name__ == "__main__":
    main()
