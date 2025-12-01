from typing import List
from openai import OpenAI

INTENT_LABELS: List[str] = [
    "환불요청",
    "배송문의",
    "불만",
    "파손문의",
    "결제문제",
    "일반문의",
]


class IntentAgent:
    def __init__(self, client: OpenAI, model_name: str = "gpt-4.1-mini"):
        self.client = client
        self.model_name = model_name

    def classify_intent(self, text: str) -> str:
        prompt = f"""
다음 고객 발화의 의도를 아래 라벨 중 하나로 분류하세요.

가능한 라벨: {INTENT_LABELS}

고객 발화: "{text}"

규칙:
- 라벨만 출력하세요.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )

        label = response.choices[0].message.content.strip()
        return label if label in INTENT_LABELS else "일반문의"
