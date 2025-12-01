from openai import OpenAI


class CalmAgent:
    def __init__(self, client: OpenAI, model_name: str = "gpt-4.1-mini"):
        self.client = client
        self.model_name = model_name

    def generate_tip(self, emotion_label: str) -> str:
        prompt = f"""
당신은 콜센터 상담사 교육 코치입니다.
고객의 감정은 '{emotion_label}' 입니다.

상담사가 감정적으로 휘말리지 않고 차분하게 응대하기 위한
심리 안정 팁을 2~3문장으로 작성하세요.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()
