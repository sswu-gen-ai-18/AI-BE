from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from agents.policy_rag import POLICY_RETRIEVER
from agents.calm_agent import CalmAgent
import os
import json

class GuideAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=api_key
        )

        # CalmAgent 인스턴스
        self.calm_agent = CalmAgent()

        # ---------------------------------------
        #  "AI Agent의 계획 능력"을 담당하는 프롬프트
        # ---------------------------------------
        self.planner_prompt = PromptTemplate(
            input_variables=["intent", "emotion_label", "emotion_score"],
            template="""
너는 고객센터 상담 에이전트이다.
아래 규칙을 바탕으로 어떤 행동(Action)을 먼저 수행해야 할지 스스로 결정하라.

### 규칙
- emotion_label이 "anger" 또는 "fear" 또는 "sad"이고 emotion_score ≥ 0.6이면: calm_message 생성 필요.
- intent가 환불/교환/배송이면: policy_search 필요.
- 일반 문의라면: basic_response만 수행.

### Output Format (JSON):
{{
    "actions": ["calm", "policy", "basic"]
}}

intent: {intent}
emotion_label: {emotion_label}
emotion_score: {emotion_score}

JSON만 출력하라.
"""
        )

        self.planner_chain = LLMChain(
            llm=self.llm,
            prompt=self.planner_prompt
        )

        # -------------------------------------------------
        # Guide Response 템플릿
        # -------------------------------------------------
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
응답 규칙
- 감정 공감 문장 생성 금지 (이 부분은 CalmAgent가 이미 제공함), 만약 감정 공감을 하고싶다면 고객의 감정을 공감하는 문장을 넣어라.
- 문제 요약 → 조치 안내 순서
- 정책 정보(policy_context) 적용
- 고객에게 전달하는 말만 작성 (상담사 안정 문장 금지)
- 전체는 2~4문장, 상담사 톤
==================================================

[응답]
고객에게 전달할 최종 응답 문장을 작성하세요.
"""
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.template)

    # =====================================================================
    # 실제 실행: LLM이 'actions'를 계획하고 → Tool들을 실행하는 반자율 구조
    # =====================================================================
    def generate(self, system_prompt, user_text, intent, emotion_label, emotion_score):

        # 1) LLM에게 어떤 행동(Action)을 할지 PLAN 결정 요청
        raw_plan = self.planner_chain.run(
            intent=intent,
            emotion_label=emotion_label,
            emotion_score=emotion_score
        )

        try:
            plan = json.loads(raw_plan)
            actions = plan.get("actions", [])
        except:
            # 실패 시 기본 행동
            actions = ["policy", "basic"]

        # ------------------------------
        # 2) PLAN 기반 실행
        # ------------------------------
        policy_context = ""
        calm_message = ""

        for act in actions:
            if act == "calm":
                calm_message = self.calm_agent.generate(
                    emotion_label=emotion_label,
                    emotion_score=emotion_score
                )

            elif act == "policy":
                docs = POLICY_RETRIEVER.get_relevant_documents(user_text)
                policy_context = "\n".join(doc.page_content for doc in docs)

        # ------------------------------
        # 3) 고객 대응문 생성
        # ------------------------------
        guide_reply = self.chain.run(
            system_prompt=system_prompt,
            user_text=user_text,
            policy_context=policy_context,
            intent=intent,
            emotion_label=emotion_label,
            emotion_score=emotion_score
        ).strip()

        # ------------------------------
        # 4) 최종 response_text 조합
        #    → LLM이 섞지 못하게 "고정 문자열"로 조립
        # ------------------------------

        return guide_reply