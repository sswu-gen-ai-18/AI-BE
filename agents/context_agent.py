from typing import Tuple

class ContextAgent:
    def __init__(self):
        self.emotion_policies = {
            "anger": "감정을 진정시키고 상황 설명 + 해결 절차 제시",
            "sad": "위로와 지지를 중심으로 따뜻한 톤으로 안내",
            "fear": "불안감을 줄이기 위해 상태와 다음 절차를 명확하게 설명",
        }

    def get_emotion_level(self, score: float) -> str:
        if score < 0.34:
            return "low"
        elif score < 0.67:
            return "medium"
        else:
            return "high"

    def build_prompts(
        self,
        customer_text: str,
        emotion_label: str,
        emotion_score: float,
        intent_label: str,
    ) -> Tuple[str, str]:

        strategy = self.emotion_policies.get(
            emotion_label, "정중하게 공감하고 문제 해결 방법을 안내"
        )

        level = self.get_emotion_level(emotion_score)

        # 감정 강도별 추가 전략
        emotion_addons = {
            "anger": {
                "low": "약한 분노로 판단되므로 공감은 가볍게, 해결책 중심으로 작성하세요.",
                "medium": "중간 수준의 분노이므로 공감과 설명 균형을 맞추세요.",
                "high": "매우 높은 분노 상태이므로 진정시키는 문장을 우선적으로 사용하세요.",
            },
            "sad": {
                "low": "가벼운 아쉬움 수준이므로 부드럽게 공감하세요.",
                "medium": "속상함이 느껴지므로 차분히 위로하세요.",
                "high": "큰 슬픔이 느껴지므로 정서적 안정이 우선입니다.",
            },
            "fear": {
                "low": "약한 걱정이므로 간단히 안심시켜주세요.",
                "medium": "불안함이 있으므로 절차를 차분히 설명하세요.",
                "high": "강한 불안 상태이므로 단계별로 아주 명확하게 안내하세요.",
            }
        }

        level_strategy = emotion_addons[emotion_label][level]

        system_prompt = f"""
당신은 고객센터 상담사 보조 AI입니다.

- 고객 감정: {emotion_label}
- 감정 점수: {emotion_score}
- 감정 강도: {level}
- 기본 감정 전략: {strategy}
- 감정 강도 추가 전략: {level_strategy}
- 고객 의도: {intent_label}

이 정보들을 반영하여 최적의 응대문을 생성하세요.
"""

        user_prompt = f"""
고객 발화: "{customer_text}"

규칙:
1. 공감 → 상황 설명 → 해결책 → 마무리
2. 감정 강도(level)에 맞게 말투와 응대 수위를 조절할 것
3. 3~5문장
4. 존댓말
"""

        return system_prompt, user_prompt