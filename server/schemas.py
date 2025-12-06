from pydantic import BaseModel
from typing import Literal

class CallInput(BaseModel):
    session_id: str 
    text: str
    emotion_label: Literal["anger", "sad", "fear"]
    emotion_score: float

class ResponseGuide(BaseModel):
    intent: str
    emotion_label: str
    response_text: str
    customer_response_text: str   # GuideAgent가 만든 고객응대 문장
    agent_calm_guide: str         # CalmAgent가 만든 상담사 안정 가이드

class CallAnalysisResult(BaseModel):
    result: ResponseGuide