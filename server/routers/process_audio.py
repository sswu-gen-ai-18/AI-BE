# server/routers/process_audio.py
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from schemas import CallInput, CallAnalysisResult, ResponseGuide
from agents.intent_agent import IntentAgent
from agents.guide_agent import GuideAgent
from agents.calm_agent import CalmAgent
from agents.emotion_smoothing import EmotionSmoother
from agents.emotion_agent import EmotionAgent  # ✅ KoBERT 감정 분류


router = APIRouter()

# ==========================
# 에이전트 초기화
# ==========================
intent_agent = IntentAgent()
guide_agent = GuideAgent()
calm_agent = CalmAgent()
emotion_smoother = EmotionSmoother(window=3)
stt_agent = STTAgent()  # ← 음성 → 텍스트용


# KoBERT 감정 에이전트
emotion_agent = EmotionAgent()


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
        data.emotion_score,
    )

    # 3) GuideAgent 호출
    response_text = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=data.emotion_label,
        emotion_score=smoothed_score,
    )

    # 4) 반환 패키지
    result = ResponseGuide(
        intent=intent,
        emotion_label=data.emotion_label,
        response_text=response_text,
    )
    return CallAnalysisResult(result=result)


# ==========================
# 텍스트-only 분석 API (Solar STT 없이 KoBERT 감정만)
# ==========================

class SolarCallInput(BaseModel):
    session_id: str
    text: str


@router.post("/analyze-solar", response_model=CallAnalysisResult)
def analyze_call_solar(data: SolarCallInput):
    """
    텍스트만 받아서 KoBERT 감정 분석 → intent 분류 → smoothing → 가이드 생성
    """

    # 0) KoBERT 감정 분석
    emotion_result = emotion_agent.predict(data.text)
    emotion_label = emotion_result["emotion_label"]
    raw_emotion_score = emotion_result["emotion_score"]

    # 1) Intent 분류
    intent = intent_agent.classify_intent(data.text)

    # 2) 감정 smoothing
    smoothed_score = emotion_smoother.add_score(
        data.session_id,
        raw_emotion_score,
    )

    # 3) GuideAgent 호출
    response_text = guide_agent.generate(
        system_prompt="당신은 고객센터 전문 상담사입니다.",
        user_text=data.text,
        intent=intent,
        emotion_label=emotion_label,
        emotion_score=smoothed_score,
    )

    # 4) 패키징
    result = ResponseGuide(
        intent=intent,
        emotion_label=emotion_label,
        response_text=response_text,
    )
    return CallAnalysisResult(result=result)


@router.post("/analyze-audio", response_model=CallAnalysisResult)
async def analyze_call_audio(
    session_id: str,              # 쿼리 파라미터로 받을 session_id
    file: UploadFile = File(...), # 업로드된 음성 파일
):
    """
    음성 파일을 받아서:
      1) STTAgent.run()으로 텍스트 추출
      2) KoBERT 감정 분석
      3) Intent 분류
      4) GuideAgent로 상담 답변 생성
    """

    # 1) 업로드된 파일을 임시 파일로 저장
    import tempfile, os

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 2) STT 실행 (음성 -> 텍스트)
        text = stt_agent.run(tmp_path)

        # 3) KoBERT 감정 분석
        emotion_result = emotion_agent.predict(text)
        emotion_label = emotion_result["emotion_label"]
        raw_emotion_score = emotion_result["emotion_score"]

        # 4) Intent 분류
        intent = intent_agent.classify_intent(text)

        # 5) 감정 smoothing
        smoothed_score = emotion_smoother.add_score(
            session_id,
            raw_emotion_score,
        )

        # 6) GuideAgent 호출
        response_text = guide_agent.generate(
            system_prompt="당신은 고객센터 전문 상담사입니다.",
            user_text=text,
            intent=intent,
            emotion_label=emotion_label,
            emotion_score=smoothed_score,
        )

        # 7) 패키징
        result = ResponseGuide(
            intent=intent,
            emotion_label=emotion_label,
            response_text=response_text,
        )
        return CallAnalysisResult(result=result)

    finally:
        # 임시 파일 삭제
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# ==========================
# 정책 파일 확인 API
# ==========================
@router.get("/debug/policies")
def debug_policies():
    import os
    from agents.policy_rag import POLICY_DIR  # ✅ server. prefix 제거

    files = os.listdir(POLICY_DIR)
    return {"policies": files}
