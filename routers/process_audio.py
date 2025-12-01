from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

from agents.intent_agent import IntentAgent
from agents.context_agent import ContextAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

intent_agent = IntentAgent(client)
context_agent = ContextAgent()
guide_agent = GuideAgent(client)
calm_agent = CalmAgent(client)

router = APIRouter(prefix="/api", tags=["customer-care"])


# -------------------------
# 입력 JSON 모델
# -------------------------
class EmotionInput(BaseModel):
    text: str
    emotion_label: Literal["anger", "sad", "fear"]
    emotion_score: float


# -------------------------
# 출력 JSON 모델
# -------------------------
class CareResponse(BaseModel):
    customer_text: str
    emotion_label: str
    emotion_score: float
    intent_label: str
    customer_response: str
    counselor_tip: str


@router.post("/analyze", response_model=CareResponse)
async def analyze_call(data: EmotionInput):

    # 1) Intent 분류
    intent_label = intent_agent.classify_intent(data.text)

    # 2) Prompt 구성
    system_prompt, user_prompt = context_agent.build_prompts(
        customer_text=data.text,
        emotion_label=data.emotion_label,
        emotion_score=data.emotion_score, 
        intent_label=intent_label,
    )

    # 3) 고객 응대 멘트 생성
    customer_response = guide_agent.generate_response(system_prompt, user_prompt)

    # 4) 상담사 심리 안정 팁 생성
    counselor_tip = calm_agent.generate_tip(data.emotion_label)

    # 5) 최종 전체 JSON 반환
    return CareResponse(
        customer_text=data.text,
        emotion_label=data.emotion_label,
        emotion_score=data.emotion_score,
        intent_label=intent_label,
        customer_response=customer_response,
        counselor_tip=counselor_tip,
    )
