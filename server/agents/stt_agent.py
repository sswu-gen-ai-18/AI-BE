# server/agents/stt_agent.py

import os
import whisper


class STTAgent:
    """
    Whisper 기반 STT 에이전트 (음성 -> 텍스트)
    """

    def __init__(self, device: str | None = None):
        # 로컬(Mac)에서는 mps, Render 같은 서버에서는 cpu 사용
        if device is None:
            # macOS면 mps, 그 외는 cpu
            if os.uname().sysname == "Darwin":
                device = "mps"
            else:
                device = "cpu"

        print(f"[STTAgent] Whisper 모델 로딩 중... (device={device})")
        self.model = whisper.load_model("small", device=device)

    def transcribe(self, audio_path: str) -> str:
        """
        Whisper를 사용해서 음성을 한국어 텍스트로 변환
        """
        print(f"[STTAgent] STT 시작: {audio_path}")

        result = self.model.transcribe(
            audio_path,
            language="ko",
            fp16=False,  # cpu/mps 환경에서 안전하게
        )

        text = (result.get("text") or "").strip()
        print(f"[STTAgent] STT 결과: {text}")
        return text
