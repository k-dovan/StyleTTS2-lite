import os
import torch
import gradio as gr
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
import numpy as np

from tts_text_norm.utils.helper import split_text_for_inference

import time
from tts_text_norm.cores.normalizer import TextNormalizer
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

# Define paths
MODEL_DIR = "apps/models/vi"

# Get the directory where the current script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define available voices with Vietnamese labels
VOICES = {
    "Nam": {
        "Điềm đạm": os.path.join(BASE_DIR, "voices/nam-calm.wav"),
        "Chậm": os.path.join(BASE_DIR, "voices/nam-cham.wav"),
        "Truyền cảm": os.path.join(BASE_DIR, "voices/nam-truyen-cam.wav"),
        "Nhanh": os.path.join(BASE_DIR, "voices/nam-nhanh.wav")
    },
    "Nữ": {
        "Điềm đạm": os.path.join(BASE_DIR, "voices/nu-calm.wav"),
        "Chậm": os.path.join(BASE_DIR, "voices/nu-cham.wav"),
        "Lưu loát": os.path.join(BASE_DIR, "voices/nu-luu-loat.wav"),
        "Nhẹ nhàng": os.path.join(BASE_DIR, "voices/nu-nhe-nhang.wav"),
        "Nhã nhặn": os.path.join(BASE_DIR, "voices/nu-nhan-nha.wav")
    }
}

def load_model():
    """Download, load, and initialize the XTTS model."""
    # print("🔄 Đang tải mô hình XTTS...")
    # snapshot_download(repo_id="capleaf/viXTTS", repo_type="model", local_dir=MODEL_DIR)
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
    text_normalizer = TextNormalizer("./.cached/exps/vncorenlp/")
    print(f"[*] take {time.time() - tik} seconds to load normalizer model")
    tik = time.time()   
        
    return text_normalizer

XTTS_MODEL = load_model()
text_normalizer = load_normalizer()

def text_to_speech(text, gender, style):
    """Generate speech dynamically using a specific voice style."""
    if gender not in VOICES or style not in VOICES[gender]:
        return {"error": "Giọng hoặc phong cách không hợp lệ."}

    ref_audio = VOICES[gender][style]

    try:
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=ref_audio,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        
        # split text to multiple chunks
        normalized_text = text_normalizer(text)
        text_chunks = split_text_for_inference(text=normalized_text)
        
        # Generate speech
        out_wavs = [XTTS_MODEL.inference(
            text=text_chunk,
            language="vi",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
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
    style_select = gr.Radio(choices=list(VOICES["Nam"].keys()), label="Chọn phong cách", value="Điềm đạm")

    gender_select.change(update_styles, inputs=[gender_select], outputs=[style_select])

    generate_button = gr.Button("🎧 Tạo giọng nói")
    output_audio = gr.Audio(type="numpy")

    generate_button.click(text_to_speech, inputs=[text_input, gender_select, style_select], outputs=[output_audio])

if __name__ == "__main__":
    app.launch(
               server_name="0.0.0.0",
               server_port=7861, 
               root_path="/tts/vi/ref-default"
               )
