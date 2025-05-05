import os
import torch
import gradio as gr
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import Xtts, XttsArgs
from huggingface_hub import snapshot_download
import numpy as np

from tts_text_norm.utils.helper import split_text_for_inference

import time

from nemo_text_processing.text_normalization.normalize import Normalizer

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

# Define paths
MODEL_DIR = "apps/models/en"

# Get the directory where the current script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

torch.serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        BaseDatasetConfig,
        XttsArgs
    ])

# Define available voices with Vietnamese labels
VOICES = {
    "Nam": {
        "Andrew Chipper": "Andrew Chipper",
        "Badr Odhiambo": "Badr Odhiambo",
        "Dionisio Schuyler": "Dionisio Schuyler",
        # "Royston Min": "Royston Min",
        # "Viktor Eka": "Viktor Eka",
        # "Abrahan Mack": "Abrahan Mack",
        # "Adde Michal": "Adde Michal",
        # "Baldur Sanjin": "Baldur Sanjin",
        # "Craig Gutsy": "Craig Gutsy",
        # "Damien Black": "Damien Black",
        # "Gilberto Mathias": "Gilberto Mathias",
        # "Ilkin Urbano": "Ilkin Urbano",
        # "Kazuhiko Atallah": "Kazuhiko Atallah",
        # "Ludvig Milivoj": "Ludvig Milivoj",
        # "Suad Qasim": "Suad Qasim",
        # "Torcull Diarmuid": "Torcull Diarmuid",
        # "Viktor Menelaos": "Viktor Menelaos",
        # "Zacharie Aimilios": "Zacharie Aimilios",
        # "Filip Traverse": "Filip Traverse",
        # "Damjan Chapman": "Damjan Chapman",
        # "Wulf Carlevaro": "Wulf Carlevaro",
        # "Aaron Dreschner": "Aaron Dreschner",
        # "Kumar Dahl": "Kumar Dahl",
        # "Eugenio Mataracı": "Eugenio Mataracı",
        # "Ferran Simen": "Ferran Simen",
        # "Xavier Hayasaka": "Xavier Hayasaka",
        # "Luis Moray": "Luis Moray",
        # "Marcos Rudaski": "Marcos Rudaski",
        # "Ige Behringer": "Ige Behringer"
    },
    "Nữ": {
        "Claribel Dervla": "Claribel Dervla",
        "Daisy Studious": "Daisy Studious",
        "Gracie Wise": "Gracie Wise",
        # "Tammie Ema": "Tammie Ema",
        # "Alison Dietlinde": "Alison Dietlinde",
        # "Ana Florence": "Ana Florence",
        # "Annmarie Nele": "Annmarie Nele",
        # "Asya Anara": "Asya Anara",
        # "Brenda Stern": "Brenda Stern",
        # "Gitta Nikolina": "Gitta Nikolina",
        # "Henriette Usha": "Henriette Usha",
        # "Sofia Hellen": "Sofia Hellen",
        # "Tammy Grit": "Tammy Grit",
        # "Tanja Adelina": "Tanja Adelina",
        # "Vjollca Johnnie": "Vjollca Johnnie",
        # "Nova Hogarth": "Nova Hogarth",
        # "Maja Ruoho": "Maja Ruoho",
        # "Uta Obando": "Uta Obando",
        # "Lidiya Szekeres": "Lidiya Szekeres",
        # "Chandra MacFarland": "Chandra MacFarland",
        # "Szofi Granger": "Szofi Granger",
        # "Camilla Holmström": "Camilla Holmström",
        # "Lilya Stainthorpe": "Lilya Stainthorpe",
        # "Zofija Kendrick": "Zofija Kendrick",
        # "Narelle Moon": "Narelle Moon",
        # "Barbora MacLean": "Barbora MacLean",
        # "Alexandra Hisakawa": "Alexandra Hisakawa",
        # "Alma María": "Alma María",
        # "Rosemary Okafor": "Rosemary Okafor"
    }
}

def load_model():
    """Download, load, and initialize the XTTS model."""
    # print("🔄 Đang tải mô hình XTTS...")
    # snapshot_download(repo_id="coqui/XTTS-v2", repo_type="model", local_dir=MODEL_DIR)
    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))
    config.model_config = {"arbitrary_types_allowed": True}
    xtts_model = Xtts.init_from_config(config)
    xtts_model.load_checkpoint(config, checkpoint_dir=MODEL_DIR)
    xtts_model.eval()
    if torch.cuda.is_available():
        xtts_model.cuda()
    print("✅ Mô hình XTTS đã tải thành công!")
    return xtts_model

def load_normalizer():
    tik = time.time()
    text_normalizer = Normalizer(lang="en", cache_dir="./.cached/nemo", input_case="cased")
    print(f"[*] take {time.time() - tik} seconds to load normalizer model")
    tik = time.time()   
        
    return text_normalizer

XTTS_MODEL = load_model()
text_normalizer = load_normalizer()

# list available `EN` speakers
# print ("Available speakers:")
# for speaker in XTTS_MODEL.speaker_manager.speakers:
#     print("-", speaker)

def text_to_speech(text, gender, speaker):
    """Generate speech dynamically using a specific voice style."""
    if gender not in VOICES or speaker not in VOICES[gender]:
        return {"error": "Giọng hoặc phong cách không hợp lệ."}

    ref_speaker = VOICES[gender][speaker]

    try:
        # gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        #     audio_path=ref_speaker,
        #     gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        #     max_ref_length=XTTS_MODEL.config.max_ref_len,
        #     sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        # )
        
        # split text to multiple chunks
        normalized_text = text_normalizer.normalize(text)
        text_chunks = split_text_for_inference(text=normalized_text)
        
        # Generate speech
        out_wavs = [XTTS_MODEL.synthesize(
            text=text_chunk,
            language="en",
            speaker_id=ref_speaker,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )['wav'] for text_chunk in text_chunks]
        
        data = np.concatenate(out_wavs)
        
        print (f'[DEBUG] sample_rate, data: 24000, {data}')
        
        return (24000, data)
    
    except Exception as e:
        return {"error": f"Lỗi tạo giọng nói: {e}"}

# Gradio Interface
def update_styles(selected_gender):
    """Dynamically update available speaking styles based on gender selection."""
    return gr.Radio(choices=list(VOICES[selected_gender].keys()), value=list(VOICES[selected_gender].keys())[0])

with gr.Blocks() as app:
    gr.Markdown("# 🎙️ Trình tạo giọng nói tiếng Việt")
    with gr.Row():
        text_input = gr.Textbox(label="Nhập văn bản", placeholder="Nhập văn bản vào đây...")

    gender_select = gr.Radio(choices=list(VOICES.keys()), label="Chọn giọng nói", value="Nam")
    style_select = gr.Radio(choices=list(VOICES["Nam"].keys()), label="Chọn phong cách", value="Andrew Chipper")

    gender_select.change(update_styles, inputs=[gender_select], outputs=[style_select])

    generate_button = gr.Button("🎧 Tạo giọng nói")
    output_audio = gr.Audio(type="numpy")

    generate_button.click(text_to_speech, inputs=[text_input, gender_select, style_select], outputs=[output_audio])

if __name__ == "__main__":
    app.launch(
               server_name="0.0.0.0",
               server_port=7862, 
               root_path="/tts/en/ref-default"
               )
