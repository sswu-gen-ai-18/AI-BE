from openai import OpenAI
import os

class CalmAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    # emotion_score는 추가되지만 사용하지 않아도 됨
    def generate(self, emotion_label, emotion_score=None):
        prompt = f"""
상담사에게 줄 감정 안정 가이드를 생성하세요.

고객 감정: {emotion_label}

형식:
- 상담사가 스스로 안정할 수 있는 팁 1개
- 상담 전략 1개
"""

        res = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return res.choices[0].message.content
