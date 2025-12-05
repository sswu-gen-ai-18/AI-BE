# server/agents/stt_agent.py
import os
import requests

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# Upstage Solar STT 엔드포인트
SOLAR_STT_URL = "https://api.upstage.ai/v1/audio/transcriptions"


class STTAgent:
    def __init__(self):
        if UPSTAGE_API_KEY is None:
            raise ValueError("UPSTAGE_API_KEY is not set!")

    def transcribe(self, audio_path: str) -> str:
        """
        Solar STT를 사용하여 음성을 텍스트로 변환.
        긴 음성도 내부에서 발화 단위로 잘라서 처리해줌.
        """
        with open(audio_path, "rb") as f:
            response = requests.post(
                SOLAR_STT_URL,
                headers={
                    "Authorization": f"Bearer {UPSTAGE_API_KEY}"
                },
                files={
                    "file": (os.path.basename(audio_path), f, "audio/wav")
                },
                data={
                    "language": "ko"
                },
            )

        try:
            data = response.json()
        except Exception as e:
            print("[Solar STT] JSON 파싱 에러:", e)
            print("[Solar STT] Raw response:", response.text)
            return ""

        if "text" not in data:
            print("[Solar STT] text 필드 없음. 응답:", data)
            return ""

        return data["text"].strip()
