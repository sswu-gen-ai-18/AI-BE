from pydantic import BaseModel
from typing import Literal

class CallInput(BaseModel):
    session_id: str
    text: str
    emotion_label: Literal["anger", "sad", "fear"]
    emotion_score: float


class ResponseGuide(BaseModel):
    intent: str = ""
    emotion_label: str = ""
    emotion_score: float = 0.0
    response_text: str = ""   # 최종 패키징된 텍스트 (고객 대응 + 안정 피드백)


class CallAnalysisResult(BaseModel):
    result: ResponseGuide
