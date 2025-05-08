import gradio as gr
from TTS.api import TTS

# Cached model instance
cached_model = {"lang": None, "tts": None}

# Predefined models for each language
MODEL_MAPPING = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "vi": "tts_models/vi/vais1000/glow-tts"
}

def load_model(language):
    """Load and cache TTS model based on the selected language."""
    if cached_model["lang"] != language:
        print(f"Loading new model for language: {language}")
        model_name = MODEL_MAPPING.get(language)
        tts = TTS(model_name)
        cached_model["lang"] = language
        cached_model["tts"] = tts
    return cached_model["tts"]

def synthesize(text, language):
    if not text.strip():
        return None
    tts = load_model(language)
    # Return the audio path (gradio handles playing)
    output_path = "./.cached/output.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## 🗣️ Multilingual Text-to-Speech App")
    lang = gr.Radio(["en", "vi"], label="Choose Language", value="en")
    textbox = gr.Textbox(label="Enter Text", placeholder="Type something...")
    btn = gr.Button("Generate Voice")
    audio_output = gr.Audio(label="Generated Speech")

    btn.click(fn=synthesize, inputs=[textbox, lang], outputs=audio_output)

if __name__ == "__main__":
    app.launch(
               server_name="0.0.0.0",
               server_port=7865, 
               root_path="/tts/piper"
               )
