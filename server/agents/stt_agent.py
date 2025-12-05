# kobert_emotion_final/agents/stt_agent.py
import whisper

class STTAgent:
    def __init__(self, device="mps"):
        print("[STTAgent] Whisper 모델 로딩 중...")
        self.model = whisper.load_model("small", device=device)

    def run(self, audio_path: str) -> str:
        result = self.model.transcribe(
            audio_path,
            language="ko",
            fp16=False,
        )
        return result["text"]
