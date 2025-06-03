import requests
from data_processing.correctors.base_corrector import BaseCorrector

class OllamaCorrector(BaseCorrector):
    def __init__(self, model, api_url="http://localhost:11434/api/chat"):
        super().__init__(model)
        self.api_url = api_url

    def correct(self, stt_text, ref_text):
        prompt = self.build_prompt(stt_text, ref_text)

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()['message']['content'].strip()
        except Exception as e:
            print(f"[ERROR] Ollama correction failed: {e}")
            return stt_text
