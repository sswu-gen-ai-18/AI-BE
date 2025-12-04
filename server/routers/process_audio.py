from fastapi import APIRouter
from models import CallInput, CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.context_agent import ContextAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother

router = APIRouter()

intent_agent = IntentAgent()
context_agent = ContextAgent()
guide_agent = GuideAgent()
calm_agent = CalmAgent()

emotion_smoother = EmotionSmoother(window=3)   # ⭐ moving average smoothing


@router.post("/analyze", response_model=CallAnalysisResult)
def analyze_call(data: CallInput):

    # 1) 의도 분류
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 smoothing
    smoothed_score = emotion_smoother.add_score(
        data.session_id,       # ⭐ 세션 단위 누적
        data.emotion_score
    )

    # 3) 프롬프트 생성
    system_prompt, user_prompt = context_agent.build_prompts(
        data.text,
        intent,
        data.emotion_label,
        smoothed_score       # ⭐ smoothing된 점수 넣기
    )

    # 4) 대응문 생성
    response_text = guide_agent.generate(system_prompt, user_prompt)

    # 5) 최종 패키징
    result = ResponseGuide(
        intent=intent,
        emotion_label=data.emotion_label,
        response_text=response_text,
    )

    return CallAnalysisResult(result=result)
