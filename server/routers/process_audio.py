from fastapi import APIRouter
from pydantic import BaseModel

from schemas import CallInput, CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother
from agents.solar_emotion_agent import SolarEmotionAgent


router = APIRouter()

# ==========================
# 에이전트 초기화
# ==========================
intent_agent = IntentAgent()
guide_agent = GuideAgent()
calm_agent = CalmAgent()

emotion_smoother = EmotionSmoother(window=3)

solar_emotion_agent = SolarEmotionAgent



# ==========================
# 기존 오디오 기반 분석 API
# ==========================
@router.post("/analyze", response_model=CallAnalysisResult)
def analyze_call(data: CallInput):
    """
    Whisper → text → (emotion_label + score 제공됨) → smoothing → response 생성
    """

    # 1) Intent 분류
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 score smoothing
    smoothed_score = emotion_smoother.add_score(
        data.session_id,
        data.emotion_score
    )

    # 3) GuideAgent 호출
    response_text = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=data.emotion_label,
        emotion_score=smoothed_score
    )

    # 4) 반환 패키지
    result = ResponseGuide(
        intent=intent,
        emotion_label=data.emotion_label,
        response_text=response_text,
    )
    return CallAnalysisResult(result=result)


# ==========================
# Solar 기반 텍스트-only 분석 API
# ==========================

class SolarCallInput(BaseModel):
     session_id: str
     text: str


@router.post("/analyze-solar", response_model=CallAnalysisResult)
def analyze_call_solar(data: SolarCallInput):
    """
    텍스트만 받아서 Solar 기반 감정 분석 → intent 분류 → smoothing → 가이드 생성
    """

    # 0) Solar 감정 분석
    emotion_result = solar_emotion_agent.predict(data.text)
    emotion_label = emotion_result["emotion_label"]
    raw_emotion_score = emotion_result["emotion_score"]


    # 1) Intent 분류
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 smoothing
    smoothed_score = emotion_smoother.add_score(
        data.session_id,
        raw_emotion_score
    )

    # 3) GuideAgent 호출
    response_text = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=emotion_label,
        emotion_score=smoothed_score
    )

    # 4) 패키징
    result = ResponseGuide(
        intent=intent,
        emotion_label=emotion_label,
        response_text=response_text,
    )
    return CallAnalysisResult(result=result)

class SolarCallInput(BaseModel):
    session_id: str
    text: str


@router.post("/analyze-solar", response_model=CallAnalysisResult)
def analyze_call_solar(data: SolarCallInput):
    """
    텍스트만 받아서 Solar 기반 감정 분석 → intent 분류 → smoothing → 가이드 생성
    """

    # 0) Solar 감정 분석
    emotion_result = solar_emotion_agent.predict(data.text)
    emotion_label = emotion_result["emotion_label"]
    raw_emotion_score = emotion_result["emotion_score"]


    # 1) Intent 분류
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 smoothing
    smoothed_score = emotion_smoother.add_score(
        data.session_id,
        raw_emotion_score
    )

    # 3) GuideAgent 호출
    response_text = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=emotion_label,
        emotion_score=smoothed_score
    )

    # 4) 패키징
    result = ResponseGuide(
        intent=intent,
        emotion_label=emotion_label,
        response_text=response_text,
    )
    return CallAnalysisResult(result=result)



# ==========================
# 정책 파일 확인 API
# ==========================
@router.get("/debug/policies")
def debug_policies():
    import os
    from server.agents.policy_rag import POLICY_DIR

    files = os.listdir(POLICY_DIR)
    return {"policies": files}
