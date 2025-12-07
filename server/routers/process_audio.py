# server/routers/process_audio.py

from fastapi import APIRouter
from pydantic import BaseModel

from schemas import CallInput, CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother
from agents.emotion_agent import EmotionAgent      # KoBERT 감정 분류

router = APIRouter()

# ==========================
# Initialize Agents
# ==========================
intent_agent = IntentAgent()
guide_agent = GuideAgent()
calm_agent = CalmAgent()
emotion_smoother = EmotionSmoother(window=3)
emotion_agent = EmotionAgent()


# ==========================
# 1) /api/analyze
#    - 프론트에서 이미 emotion_label, emotion_score까지 계산해서 보내주는 케이스
# ==========================
@router.post("/analyze", response_model=CallAnalysisResult)
def analyze_call(data: CallInput):

    # 1) Intent
    intent = intent_agent.classify_intent(data.text)

    # 2) Smooth emotion score
    smoothed_score = emotion_smoother.add_score(
        data.session_id, data.emotion_score
    )

    # 3) 고객용 응답 생성 (GuideAgent)
    customer_response = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=data.emotion_label,
        emotion_score=smoothed_score,
    )

    # 4) 상담사용 안정 가이드 (CalmAgent)
    agent_calm_message = calm_agent.generate(data.emotion_label)

    # 5) 패키징
    result = ResponseGuide(
        intent=intent,
        emotion_label=data.emotion_label,
        response_text=customer_response,
        customer_response_text=customer_response,
        agent_calm_guide=agent_calm_message,
    )
    return CallAnalysisResult(result=result)


# ==========================
# 2) /api/analyze-solar (텍스트 only)
#    - 프론트에서 STT(Solar/Whisper 등) 끝낸 후 text + session_id만 보내는 케이스
# ==========================
class SolarCallInput(BaseModel):
    session_id: str
    text: str


@router.post("/analyze-solar", response_model=CallAnalysisResult)
def analyze_call_solar(data: SolarCallInput):

    # 0) KoBERT 감정 분석
    emotion_result = emotion_agent.predict(data.text)
    emotion_label = emotion_result["emotion_label"]
    raw_emotion_score = emotion_result["emotion_score"]

    # 1) Intent
    intent = intent_agent.classify_intent(data.text)

    # 2) Smooth
    smoothed_score = emotion_smoother.add_score(
        data.session_id, raw_emotion_score
    )

    # 3) 고객 응답
    customer_response = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=emotion_label,
        emotion_score=smoothed_score,
    )

    # 4) 상담사용 안정 가이드
    agent_calm_message = calm_agent.generate(emotion_label)

    # 5) Package
    result = ResponseGuide(
        intent=intent,
        emotion_label=emotion_label,
        response_text=customer_response,
        customer_response_text=customer_response,
        agent_calm_guide=agent_calm_message,
    )
    return CallAnalysisResult(result=result)


# ==========================
# 3) 정책 파일 디버그
# ==========================
@router.get("/debug/policies")
def debug_policies():
    import os
    from agents.policy_rag import POLICY_DIR

    files = os.listdir(POLICY_DIR)
    return {"policies": files}
