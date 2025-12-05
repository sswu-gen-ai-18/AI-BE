<<<<<<< HEAD
from fastapi import APIRouter
from models import CallInput, CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother
=======
# routers/process_audio.py
from fastapi import APIRouter
from pydantic import BaseModel

from models import CallInput, CallAnalysisResult, ResponseGuide
from server.agents import IntentAgent
from server.agents.context_agent import ContextAgent
from server.agents.guide_agent import GuideAgent
from server.agents.calm_agent import CalmAgent
from server.agents.emotion_smoothing import EmotionSmoother
from server.agents import SolarEmotionAgent
>>>>>>> cca7933 (server디렉토리에 옮김,solar emotion추가)

router = APIRouter()

intent_agent = IntentAgent()
<<<<<<< HEAD
guide_agent = GuideAgent()
calm_agent = CalmAgent()

emotion_smoother = EmotionSmoother(window=3)


=======
context_agent = ContextAgent()
guide_agent = GuideAgent()
calm_agent = CalmAgent()

emotion_smoother = EmotionSmoother(window=3)   # ⭐ moving average smoothing
solar_emotion_agent = SolarEmotionAgent()      # ⭐ Solar 기반 감정 분석 에이전트


# ====================================================
# 기존: 감정 점수/레이블을 이미 받아서 사용하는 /analyze
# ====================================================
>>>>>>> cca7933 (server디렉토리에 옮김,solar emotion추가)
@router.post("/analyze", response_model=CallAnalysisResult)
def analyze_call(data: CallInput):

    # 1) 의도 분류
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 smoothing
    smoothed_score = emotion_smoother.add_score(
<<<<<<< HEAD
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
=======
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
>>>>>>> cca7933 (server디렉토리에 옮김,solar emotion추가)
    result = ResponseGuide(
        intent=intent,
        emotion_label=data.emotion_label,
        response_text=response_text,
    )

    return CallAnalysisResult(result=result)


<<<<<<< HEAD
@router.get("/debug/policies")
def debug_policies():
    import os
    from agents.policy_rag import POLICY_DIR

    files = os.listdir(POLICY_DIR)
    return {"policies": files}
=======
# =========================================
# Solar 버전: 텍스트만 받아서 감정까지 계산
# =========================================
class SolarCallInput(BaseModel):
    session_id: str
    text: str


@router.post("/analyze-solar", response_model=CallAnalysisResult)
def analyze_call_solar(data: SolarCallInput):
    """
    텍스트만 받아서 Solar로 감정(anger/sad/fear) 분석까지 한 번에 하는 엔드포인트.
    프론트/다른 서비스에서 이걸 호출하면,
    의도 + 감정 + 가이드 멘트까지 한 번에 받을 수 있음.
    """

    # 0) 감정 분석 (Solar)
    emotion_result = solar_emotion_agent.predict(data.text)
    emotion_label = emotion_result["emotion_label"]
    raw_emotion_score = emotion_result["emotion_score"]

    # 1) 의도 분류 (기존 그대로)
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 점수 smoothing (기존 로직 재사용)
    smoothed_score = emotion_smoother.add_score(
        data.session_id,
        raw_emotion_score
    )

    # 3) 프롬프트 생성 (기존과 동일, 다만 emotion_label/score가 Solar 값)
    system_prompt, user_prompt = context_agent.build_prompts(
        data.text,
        intent,
        emotion_label,
        smoothed_score
    )

    # 4) 대응문 생성
    response_text = guide_agent.generate(system_prompt, user_prompt)

    # 5) 최종 패키징
    result = ResponseGuide(
        intent=intent,
        emotion_label=emotion_label,
        response_text=response_text,
    )

    return CallAnalysisResult(result=result)
>>>>>>> cca7933 (server디렉토리에 옮김,solar emotion추가)
