from openai import OpenAI


class GuideAgent:
    def __init__(self, client: OpenAI, model_name: str = "gpt-4.1-mini"):
        self.client = client
        self.model_name = model_name

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
