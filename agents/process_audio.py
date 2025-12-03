# kobert_emotion_final/agents/final_agent.py

from stt_agent import STTAgent
from emotion_agent import EmotionAgent

import os

class CallcenterAudioProcessor:
    """
    음성 파일을 입력받아 STT → 감정분석까지 한 번에 처리하는 최종 헬퍼 클래스.
    """

    def __init__(self, stt_device="mps"):
        self.stt_agent = STTAgent(device=stt_device)
        self.emotion_agent = EmotionAgent()

    def process_audio(self, audio_path: str) -> dict:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없음: {audio_path}")

        # 1) STT
        text = self.stt_agent.run(audio_path)

        # 2) 감정분석
        emotion = self.emotion_agent.predict(text)

        return {
            "audio_path": audio_path,
            "text": text,
            "emotion": emotion,
        }
