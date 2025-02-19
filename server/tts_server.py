import os

import torch

from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from scipy.io.wavfile import write

snapshot_download(repo_id="capleaf/viXTTS",
                  repo_type="model",
                  local_dir="model")

config = XttsConfig()
config.load_json("./model/config.json")
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_dir="./model/")
XTTS_MODEL.eval()
if torch.cuda.is_available():
    XTTS_MODEL.cuda()

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path="server/voices/voice_nf02.wav",
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

out_wav = XTTS_MODEL.inference(
    text="Lấy bối cảnh miền Tây Nam Bộ vào thập niên năm một nghìn chín trăm bốn lăm. Phim xoay quanh An - một cậu bé thành thị theo học trường Pháp. Nhưng sau ngày mồng hai tháng chính năm một nghìn chín trăm bốn lăm, khi thực dân Pháp quay trở lại nước ta và tổng tấn công vào Nam bộ khiến cho mọi người phải đi di tản khắp nơi.",
    language="vi",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.3,
    length_penalty=1.0,
    repetition_penalty=10.0,
    top_k=30,
    top_p=0.85,
)

# Save to WAV file
write("server/out/output.wav", 24000, out_wav['wav'])