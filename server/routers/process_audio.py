# server/routers/process_audio.py

from fastapi import APIRouter
from pydantic import BaseModel

from schemas import CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother
from agents.emotion_agent import EmotionAgent

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
# /api/analyze-solar
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

    # 2) Smooth emotion score
    smoothed_score = emotion_smoother.add_score(
        data.session_id, raw_emotion_score
    )

    # 3) 고객 대응문 생성 (GuideAgent)
    customer_response = guide_agent.generate(
        system_prompt="""
당신은 고객센터 상담사입니다.
고객에게 전달할 실제 대응문만 생성하세요.
'감정 안정', '심호흡', '상담사 교육' 같은 문구는 절대 생성하지 마세요.
""",
        user_text=data.text,
        intent=intent,
        emotion_label=emotion_label,
        emotion_score=smoothed_score,
    )

    # 4) 상담사 안정 피드백 (CalmAgent)
    agent_calm_message = calm_agent.generate(
        emotion_label=emotion_label,
        emotion_score=smoothed_score  # calm_agent가 score 필요 없으면 무시해도 됨
    )

    # 5) 템플릿 패키징
    final_text = f"""
### 🟩 상담사 안정 피드백
{agent_calm_message}

### 🟦 추천 대응문
{customer_response}
""".strip()

    # 6) 최종 응답
    result = ResponseGuide(
        intent=intent,
        emotion_label=emotion_label,
        emotion_score=smoothed_score,
        response_text=final_text
    )

    return CallAnalysisResult(result=result)
