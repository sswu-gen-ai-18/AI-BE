# server/routers/process_audio.py

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from schemas import CallInput, CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother
from agents.emotion_agent import EmotionAgent      # KoBERT 감정 분류
from agents.stt_agent import STTAgent              # Solar STT

router = APIRouter()

# ==========================
# Initialize Agents
# ==========================
intent_agent = IntentAgent()
guide_agent = GuideAgent()
calm_agent = CalmAgent()
emotion_smoother = EmotionSmoother(window=3)

stt_agent = STTAgent()
emotion_agent = EmotionAgent()


# ==========================
# 1) /api/analyze
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
        customer_response_text=customer_response,
        agent_calm_guide=agent_calm_message
    )
    return CallAnalysisResult(result=result)



# ==========================
# 2) /api/analyze-solar (텍스트 only)
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
        customer_response_text=customer_response,
        agent_calm_guide=agent_calm_message
    )
    return CallAnalysisResult(result=result)



# ==========================
# 3) /api/analyze-audio (음성)
# ==========================
@router.post("/analyze-audio", response_model=CallAnalysisResult)
async def analyze_call_audio(
    session_id: str,
    file: UploadFile = File(...),
):

    import tempfile
    import os

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 1) STT
        text = stt_agent.transcribe(tmp_path)

        if not text:
            intent = "일반문의"
            emotion_label = "neutral"
            raw_emotion_score = 0.0
        else:
            # 2) emotion
            emotion_result = emotion_agent.predict(text)
            emotion_label = emotion_result["emotion_label"]
            raw_emotion_score = emotion_result["emotion_score"]

            # 3) intent
            intent = intent_agent.classify_intent(text)

        # 4) smooth
        smoothed_score = emotion_smoother.add_score(
            session_id, raw_emotion_score
        )

        # 5) 고객 응대 답변 생성
        customer_response = guide_agent.generate(
            system_prompt="당신은 고객센터 전문 상담사입니다.",
            user_text=text if text else "고객님의 말씀을 정확히 인식하지 못했습니다.",
            intent=intent,
            emotion_label=emotion_label,
            emotion_score=smoothed_score,
        )

        # 6) 상담사 안정 가이드
        agent_calm_message = calm_agent.generate(emotion_label)

        # 7) package
        result = ResponseGuide(
            intent=intent,
            emotion_label=emotion_label,
            customer_response_text=customer_response,
            agent_calm_guide=agent_calm_message
        )
        return CallAnalysisResult(result=result)

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass



# ==========================
# 4) 정책 파일 디버그
# ==========================
@router.get("/debug/policies")
def debug_policies()_
