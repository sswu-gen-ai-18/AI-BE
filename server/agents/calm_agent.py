from openai import OpenAI
import os

class CalmAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate(self, emotion_label, emotion_score=None):
        """
        상담사만을 위한 감정 안정 가이드 생성
        (고객에게 전달할 문장은 절대 포함 X)
        """
        prompt = f"""
다음 감정에 대해 상담사만을 위한 감정 안정 피드백을 생성하세요.

고객 감정: {emotion_label}

출력 형식:
- 상담사가 스스로 안정할 수 있는 팁 1개
- 고객 감정을 다루는 상담 전략 1개

주의:
- 고객에게 말하는 문장은 절대 쓰지 말 것
"""

        res = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return res.choices[0].message.content.strip()