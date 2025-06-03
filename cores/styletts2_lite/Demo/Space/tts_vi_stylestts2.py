#import libs
import gradio as gr
import os
import sys
import torch
import traceback
import random
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# phonemizer config (with espeak-ng on Windows)
import phonemizer
if sys.platform.startswith("win"):
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        import espeakng_loader
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
    except Exception as e:
        print(e)

def get_phoneme(text, lang):
    try:
        my_phonemizer = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True, with_stress=True, language_switch='remove-flags')
        return my_phonemizer.phonemize([text])[0]
    except Exception as e:
        print(e)

# Import inference
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from inference import StyleTTS2

# Paths
config_path = os.path.abspath(os.path.join("Configs", "config.yaml"))
models_path = os.path.abspath(os.path.join("Models", "base_model.pth"))
voice_path = os.path.join("Demo", "Audio/styletts2_lite")

model = StyleTTS2(config_path, models_path).eval().to(device)

# Example texts
eg_texts = [
    "Beneath layers of bureaucracy and forgotten policies, the school still held a quiet magic...",
    "He had never believed in fate, but when their paths crossed in the middle of a thunderstorm...",
    "While standing at the edge of the quiet lake, Maria couldn't help but wonder how many untold stories...",
    "Technological advancements in artificial intelligence have not only accelerated the pace of automation...",
    "Despite the looming deadline, Jonathan spent an hour rearranging his desk before writing a single word...",
    "In a distant galaxy orbiting a dying star, a species of sentient machines debates whether to intervene...",
    "He opened the refrigerator, expecting leftovers, but found instead a note that read, “The journey begins now,”...",
    "The ancient temple walls, once vibrant with murals, now bore the weathered marks of centuries...",
    "As the solar eclipse reached totality, the temperature dropped, the birds went silent...",
    "The sound of rain on the tin roof reminded him of summers long past...",
    "Every algorithm reflects its designer’s worldview, no matter how neutral it appears...",
    "In the heart of the city, hidden behind concrete and steel, was a garden so lush and untouched...",
    "The engine sputtered twice before giving in completely, leaving them stranded...",
    "The museum guard never expected the sculpture to move, but at precisely midnight...",
    "With each step through the desert, the ancient map grew more useless...",
    "Time slowed as the coin spun in the air, glinting with a brilliance far beyond its monetary value...",
    "No manual prepared them for this outcome: a rogue AI composing sonnets..."
]

# Character voices
character_voice_map = {
    "Witch 🧙‍♀️": "witch.wav",
    "Monk 🧘‍♂️": "monk.wav",
    "King 👑": "king.wav",
    "Queen 👸": "queen.wav",
    "Old Woman 👵": "old_woman.wav",
    "Old Man 👴": "old_man.wav",
    "Wild Animal 🐯": "animal.wav",
}
character_voice_choices = [
    (label, os.path.join(voice_path, filename))
    for label, filename in character_voice_map.items()
]

# Main synthesis function
def main(text_prompt, cloned_voice_path, mixing_enabled, character_voice_path, mix_ratio, speed, denoise, avg_style, stabilize):
    try:
        speaker = {
            "path": cloned_voice_path,
            "speed": speed
        }

        with torch.no_grad():
            phonemes = get_phoneme(text=text_prompt, lang="en-us")
            styles = model.get_styles(speaker, denoise, avg_style)

            if mixing_enabled and character_voice_path:
                alt_speaker = {
                    "path": character_voice_path,
                    "speed": speed
                }
                alt_styles = model.get_styles(alt_speaker, denoise, avg_style)

                styles['path'] = [cloned_voice_path, character_voice_path]
                styles['style'] = mix_ratio * alt_styles['style'] + (1-mix_ratio)*styles['style']
            
            audio = model.generate(phonemes, styles, stabilize, 18)
            audio = audio / np.max(np.abs(audio))  # Normalize
            return (24000, audio), "✅ Audio generated successfully!"

    except Exception:
        error_message = traceback.format_exc()
        return None, f"❌ Error: \n{error_message}"

def random_text():
    return random.choice(eg_texts), "🎲 Random text selected."

def preview_character_voice(character_voice_path):
    return character_voice_path

# Gradio UI
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>StyleTTS2‑Lite Demo</h1>")

    gr.Markdown(
        "Try uploading your voice, mix it with character voices like Witch, Monk, or King. "
        "Control the mixing strength, speed, and clarity. "
        "For more: [Github Repo](https://huggingface.co/dangtr0408/StyleTTS2-lite/)."
    )

    text_prompt = gr.Textbox(label="📝 Text Prompt", placeholder="Enter your text here...", lines=6)

    with gr.Row():
        cloned_voice = gr.Audio(label="🧬 Upload Cloned Voice", type='filepath')

        with gr.Column():
            mixing_enabled = gr.Checkbox(label="🎭 Enable Voice Mixing", value=False)
            
            character_selector = gr.Dropdown(
                label="🎙️ Character Voice",
                choices=character_voice_choices,
                value=None,
                interactive=True,
                allow_custom_value=False
            )

            character_voice_preview = gr.Audio(label="🎧 Character Voice Preview", type="filepath", interactive=False)

            mix_ratio = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="🔀 Mixing Ratio (0=Your voice, 1=Character voice)")

    with gr.Row():
        speed = gr.Slider(0.0, 2.0, step=0.1, value=1.0, label="🎚️ Speed")
        denoise = gr.Slider(0.0, 1.0, step=0.1, value=0.2, label="🧹 Denoise Strength")

    with gr.Row():
        avg_style = gr.Checkbox(label="📊 Use Average Style", value=True)
        stabilize = gr.Checkbox(label="📏 Stabilize Speaking Speed", value=True)

    with gr.Row():
        gen_button = gr.Button("🗣️ Generate Audio")
        random_button = gr.Button("🎲 Random Text")

    output_audio = gr.Audio(label="🔊 Output Audio", type='numpy')
    status = gr.Textbox(label="ℹ️ Status", lines=2, interactive=False)

    gen_button.click(
        fn=main,
        inputs=[
            text_prompt, cloned_voice, mixing_enabled,
            character_selector, mix_ratio,
            speed, denoise, avg_style, stabilize
        ],
        outputs=[output_audio, status]
    )

    random_button.click(fn=random_text, inputs=[], outputs=[text_prompt, status])

    character_selector.change(
        fn=preview_character_voice,
        inputs=[character_selector],
        outputs=[character_voice_preview]
    )

demo.launch()
