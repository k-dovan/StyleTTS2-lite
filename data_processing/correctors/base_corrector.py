import requests
from abc import ABC, abstractmethod

class BaseCorrector(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def correct(self, stt_text: str, ref_text: str) -> str:
        pass

    def build_prompt(self, stt_text, ref_text):
        return f"""
You are a text correction assistant for speech transcription.
Given:
- Speech-to-text output (possibly noisy transcription): "{stt_text}"
- Reference text (high-confidence): "{ref_text}"

Please produce a corrected, punctuated, natural Vietnamese transcription by combining
using the reference text as the source of text and speech-to-text output as the source of punctuation information. 

Notes:
- Remove leading/trailing quotation marks if allowed.
- Do NOT think or add <think> to your response.

Return only the corrected transcription.
""".strip()
