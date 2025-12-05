# server/agents/stt_agent.py
import whisper
from .solar_client import solar_chat   # ← Solar 호출

class STTAgent:
    def __init__(self, device="mps"):
        print("[STTAgent] Whisper 모델 로딩 중...")
        self.model = whisper.load_model("small", device=device)

    def run(self, audio_path: str) -> str:
        """
        1) Whisper로 음성 → 텍스트 추출
        2) Solar로 문장 정제 (맞춤법/구어체 → 상담 문장)
        """
        # 1) Whisper STT
        whisper_result = self.model.transcribe(
            audio_path,
            language="ko",
            fp16=False,
        )
        raw_text = whisper_result["text"]

        # 2) Solar로 자연스러운 상담 발화로 후처리
        messages = [
            {
                "role": "system",
                "content": (
                    "너는 한국어 고객센터 음성 인식 후처리 전문가야. "
                    "STT 결과를 자연스러운 문장으로 정제해줘. "
                    "불필요한 반복, 잡음 단어를 제거하고, "
                    "정확한 한국어 문장으로 만들어."
                ),
            },
            {"role": "user", "content": raw_text},
        ]

        solar_resp = solar_chat(
            messages,
            model="solar-1-mini-chat",
        )

        cleaned_text = solar_resp.choices[0].message.content.strip()
        return cleaned_text
