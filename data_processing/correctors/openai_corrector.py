from data_processing.correctors.base_corrector import BaseCorrector

class OpenAICorrector(BaseCorrector):
    def __init__(self, model, client):
        super().__init__(model)
        self.client = client

    def correct(self, stt_text, ref_text):
        prompt = self.build_prompt(stt_text, ref_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] OpenAI correction failed: {e}")
            return stt_text
