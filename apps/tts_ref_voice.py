import os
import torch
import gradio as gr
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
from utils.text_processing import process_text_for_tts, split_text_for_inference
from utils.audio_processing import generate_audio

# Define paths
MODEL_DIR = "model"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    """Download, load, and initialize the XTTS model."""
    print("🔄 Đang tải mô hình XTTS...")
    snapshot_download(repo_id="capleaf/viXTTS", repo_type="model", local_dir=MODEL_DIR)
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

XTTS_MODEL = load_model()

def text_to_speech(text, ref_audio):
    """Generate speech dynamically using an uploaded voice sample."""
    if ref_audio is None:
        return {"error": "Vui lòng tải lên một tệp giọng nói mẫu."}

    try:
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=ref_audio,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        
        # Process text
        normalized_text = process_text_for_tts(text=text)
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
        )["wav"] for text_chunk in text_chunks]
        
        data = np.concatenate(out_wavs)
        
        print(f'[DEBUG] sample_rate, data: 24000, {data}')
        
        return (24000, data)
    
    except Exception as e:
        return {"error": f"Lỗi tạo giọng nói: {e}"}

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# 🎙️ Trình tạo giọng nói tiếng Việt")
    
    text_input = gr.Textbox(label="Nhập văn bản", placeholder="Nhập văn bản vào đây...")
    ref_audio_input = gr.Audio(label="Tải lên giọng nói mẫu", type="filepath")
    
    generate_button = gr.Button("🎧 Tạo giọng nói")
    output_audio = gr.Audio(type="numpy")
    
    generate_button.click(text_to_speech, inputs=[text_input, ref_audio_input], outputs=[output_audio])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7862, root_path="/tts-demo-ref-voice")
