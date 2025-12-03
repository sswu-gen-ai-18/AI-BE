class ContextAgent:
    def build_prompts(self, text, intent, emotion_label, emotion_score):
        system_prompt = f"""
당신은 콜센터 상담사 보조 AI입니다.
아래 정보를 기반으로 고객에게 적절한 대응 멘트를 생성하세요.
"""

        user_prompt = f"""
고객 발화: {text}
의도: {intent}
감정: {emotion_label} (확신도 {emotion_score})

요구사항:
1) 고객 감정을 공감
2) 사실 기반 안내
3) 해결 방법 1~2개 제시
4) 정중하고 간결하게 작성
"""

        return system_prompt, user_prompt
