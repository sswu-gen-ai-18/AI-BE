from fastapi import APIRouter
from models import CallInput, CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother

router = APIRouter()

intent_agent = IntentAgent()
guide_agent = GuideAgent()
calm_agent = CalmAgent()

emotion_smoother = EmotionSmoother(window=3)


@router.post("/analyze", response_model=CallAnalysisResult)
def analyze_call(data: CallInput):

    # 1) 의도 분류
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 smoothing
    smoothed_score = emotion_smoother.add_score(
        data.session_id,
        data.emotion_score
    )

    # 3) GuideAgent 직접 호출
    response_text = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=data.emotion_label,
        emotion_score=smoothed_score
    )

    # 4) 최종 패키징
    result = ResponseGuide(
        intent=intent,
        emotion_label=data.emotion_label,
        response_text=response_text,
    )

    return CallAnalysisResult(result=result)


@router.get("/debug/policies")
def debug_policies():
    import os
    from agents.policy_rag import POLICY_DIR

    files = os.listdir(POLICY_DIR)
    return {"policies": files}
