import gradio as gr
from piper import PiperVoice
import numpy as np
import wave
import tempfile

VOICE_MODELS = {
    "en": {
        "lessac-medium": {
            "model_path": "models/en_US-lessac-medium.onnx",
            "config_path": "models/en_US-lessac-medium.onnx.json"
        },
        "ljspeech-high": {
            "model_path": "models/en_US-ljspeech-high.onnx",
            "config_path": "models/en_US-ljspeech-high.onnx.json"
        }
    },
    "vi": {
        "vivos-x_low": {
            "model_path": "models/vi_VN-vivos-x_low.onnx",
            "config_path": "models/vi_VN-vivos-x_low.onnx.json"
        },
        "25hours_single-low": {
            "model_path": "models/vi_VN-25hours_single-low.onnx",
            "config_path": "models/vi_VN-25hours_single-low.onnx.json"
        },
        "vais1000-medium": {
            "model_path": "models/vi_VN-vais1000-medium.onnx",
            "config_path": "models/vi_VN-vais1000-medium.onnx.json"
        }
    }
}

# Cache voices
loaded_voices = {}

def load_voice(lang, voice_name):
    key = f"{lang}:{voice_name}"
    if key not in loaded_voices:
        model_info = VOICE_MODELS[lang][voice_name]
        voice = PiperVoice.load(
            model_path=model_info["model_path"],
            config_path=model_info["config_path"],
            use_cuda=False
        )
        loaded_voices[key] = voice
    return loaded_voices[key]

def synthesize(lang, voice_name, text):
    voice = load_voice(lang, voice_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        with wave.open(tmp_wav.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            voice.synthesize(text, wav_file)
        return tmp_wav.name

def update_voice_radio(lang):
    voices = list(VOICE_MODELS[lang].keys())
    return gr.update(choices=voices, value=voices[0])

with gr.Blocks() as app:
    gr.Markdown("## 🗣️ Piper TTS Synthesizer")

    lang = gr.Radio(choices=["en", "vi"], label="Select Language", value="en", interactive=True)
    voice = gr.Radio(choices=list(VOICE_MODELS["en"].keys()), label="Select Voice")
    text = gr.Textbox(label="Enter Text", placeholder="Type your sentence here...")
    generate_btn = gr.Button("🔊 Generate Speech")
    audio_output = gr.Audio(label="Synthesized Audio", type="filepath")

    lang.change(fn=update_voice_radio, inputs=lang, outputs=voice)
    generate_btn.click(fn=synthesize, inputs=[lang, voice, text], outputs=audio_output)

if __name__ == "__main__":
    app.launch(
               server_name="0.0.0.0",
               server_port=7865, 
               root_path="/tts/piper"
               )
