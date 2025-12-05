from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from server.agents.policy_rag import POLICY_RETRIEVER
import os

class GuideAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            openai_api_key=api_key
        )

        # --- UPDATED PromptTemplate ---
        self.template = PromptTemplate(
            input_variables=[
                "system_prompt",
                "user_text",
                "policy_context",
                "intent",
                "emotion_label",
                "emotion_score"
            ],
            template="""
{system_prompt}

[정책 정보 참고]
{policy_context}

[고객 발화]
{user_text}

[고객 감정 분석]
- 감정 레이블: {emotion_label}
- 감정 강도(확률): {emotion_score}

[고객 의도]
{intent}


==================================================
감정 × 강도 기반 공감 멘트 생성 규칙
==================================================

1) anger (분노/짜증)
    - 강도 ≥ 0.7:
        강한 사과 + 고객의 강한 불편감 인정 + 즉시 해결 의지
    - 0.4 ≤ 강도 < 0.7:
        기대와 다름으로 인한 실망감 공감
    - 강도 < 0.4:
        부드러운 공감 + 문제 확인

2) fear (불안/공포/걱정)
    - 강도 ≥ 0.7:
        강한 안심 + 보호적 언어 + 절차 안내
    - 0.4 ≤ 강도 < 0.7:
        차분한 안정 + 상황 설명
    - 강도 < 0.4:
        불편 최소화 + 침착한 대응

3) sadness (슬픔/속상함)
    - 강도 ≥ 0.7:
        깊은 위로 + 고객 감정 인정
    - 0.4 ≤ 강도 < 0.7:
        공감 + 문제 요약
    - 강도 < 0.4:
        공감 한 문장 → 해결 안내

4) confusion (혼란/당황)
    - 강도 ≥ 0.7:
        매우 혼란스러웠을 상황을 이해 → 절차 정리
    - 강도 < 0.7:
        차분한 정리 + 방향 제시

5) neutral
    - 짧은 공감 후 바로 해결 중심

==================================================
응답 작성 규칙
==================================================

- 반드시 감정 기반 공감 문장으로 시작할 것
- 이어서 문제 요약 → 조치 안내 순서
- 정책 정보(policy_context)를 적용해 안내
- 전체는 2~4문장으로 자연스럽게 작성
- 너무 딱딱하지 말고, 상담사 톤으로 따뜻하게

==================================================

[응답]
고객에게 전달할 최종 응답 문장을 작성하세요.
"""
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.template
        )

    def generate(self, system_prompt: str, user_text: str, intent: str, emotion_label: str, emotion_score: float):

        # RAG 검색
        related_docs = POLICY_RETRIEVER.get_relevant_documents(user_text)
        policy_context = "\n\n".join(doc.page_content for doc in related_docs)

        response = self.chain.run(
            system_prompt=system_prompt,
            user_text=user_text,
            policy_context=policy_context,
            intent=intent,
            emotion_label=emotion_label,
            emotion_score=emotion_score
        )

        return response.strip()
