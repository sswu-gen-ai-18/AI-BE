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

class CallAnalysisResult(BaseModel):
    result: ResponseGuide