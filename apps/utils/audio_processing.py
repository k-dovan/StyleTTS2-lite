from pydub import AudioSegment
from io import BytesIO
import numpy as np

def audiosegment_to_numpy(audio: AudioSegment):
    """Convert a AudioSegment to a NumPy array and sample rate."""
    
    print (f'[DEBUG] audio segment: {audio}')

    samples = np.array(audio.get_array_of_samples())  # Convert to NumPy array

    # Convert to float32 [-1, 1] if audio is stereo
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))  # Stereo audio (2 channels)
    
    samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max  # Normalize to [-1, 1]
    
    print (f'[DEBUG] sample rate, data: {audio}, {samples}')

    return audio.frame_rate, samples  # Return sample rate and NumPy array 

def generate_audio(wav_buffers: list):
    """Combine .wav buffers."""
    
    print (f'[DEBUG] .wav buffers: {wav_buffers}')
    
    sep_silence = AudioSegment.silent(duration=500)
    
    
    print (f'[DEBUG] created silence audio segment.')
    
    # Combine WAV buffers
    combined_audio = AudioSegment()
    for wav_buffer in wav_buffers:
        combined_audio += AudioSegment.from_wav(wav_buffer) + sep_silence        
        
    print (f'[DEBUG] combined audio: {combined_audio}')

    return audiosegment_to_numpy(combined_audio)