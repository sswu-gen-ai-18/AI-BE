from collections import deque

class EmotionSmoother:
    def __init__(self, window=3):
        self.window = window
        self.sessions = {}  # 세션별 smoothing 저장

    def add_score(self, session_id: str, score: float) -> float:
        """
        특정 session_id의 감정 점수를 업데이트
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.window)
        """
        average로 누적 평균 매김.
        """
        history = self.sessions[session_id]
        history.append(score)
        return sum(history) / len(history)

    def reset(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
