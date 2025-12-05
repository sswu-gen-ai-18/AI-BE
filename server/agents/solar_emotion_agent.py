# server/agents/solar_emotion_agent.py

import json
from typing import Dict, Any

from agents.solar_client import solar_chat


class SolarEmotionAgent:
    """
    Upstage Solar(LLM)을 호출해서
    텍스트 감정을 [anger, sad, fear, neutral] 중 하나로 분류하고
    0~1 사이 감정 강도 score를 반환하는 에이전트.
    """

    def __init__(self):
        # 필요하면 나중에 프롬프트, 모델명 등을 __init__에서 받도록 확장 가능
        pass

    def _build_messages(self, text: str):
        """
        Solar에 보낼 messages 구성.
        JSON 형식으로만 답을 달라고 강하게 요구한다.
        """
        system_prompt = (
            "당신은 콜센터 고객 발화를 감정 분석하는 AI입니다. "
            "사용자의 한국어 발화를 보고 다음 중 하나의 감정을 판단하세요: "
            "['anger', 'sad', 'fear', 'neutral'].\n\n"
            "그리고 감정 강도를 0.0부터 1.0 사이의 실수로 표현하세요.\n\n"
            "반드시 아래 JSON 형식 하나만 출력하세요. 추가 설명은 쓰지 마세요.\n"
            '{ "emotion_label": "<감정라벨>", "emotion_score": <0.0~1.0 숫자> }'
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

    def _parse_response(self, raw_response: Any) -> Dict[str, Any]:
        """
        solar_chat 응답에서 emotion_label, emotion_score를 안전하게 파싱.
        JSON 파싱 실패 시 neutral / 0.5로 fallback.
        """
        # solar_client.solar_chat 이 어떤 타입을 리턴하는지에 따라 분기.
        # 보통 OpenAI 호환이면 completion.choices[0].message.content 형태일 것.
        try:
            # 1) content 추출
            # 예: res.choices[0].message.content
            content = None
            if hasattr(raw_response, "choices"):
                content = raw_response.choices[0].message.content
            elif isinstance(raw_response, dict):
                # 혹시 함수에서 이미 dict로 반환한다면
                content = raw_response.get("content") or raw_response.get("message") or str(raw_response)
            else:
                content = str(raw_response)

            # 2) content 안에서 JSON 부분만 뽑아서 파싱 시도
            content_str = content.strip()

            # 바로 JSON일 가능성이 높다고 가정
            data = json.loads(content_str)

            emotion_label = data.get("emotion_label", "neutral")
            emotion_score = float(data.get("emotion_score", 0.5))

            # score 범위 클리핑
            emotion_score = max(0.0, min(1.0, emotion_score))

            return {
                "emotion_label": emotion_label,
                "emotion_score": emotion_score,
            }

        except Exception:
            # 파싱 실패하면 기본값으로 fallback
            return {
                "emotion_label": "neutral",
                "emotion_score": 0.5,
            }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        외부에서 호출하는 메인 함수.
        text를 받아서 emotion_label, emotion_score dict로 반환.
        """
        messages = self._build_messages(text)
        res = solar_chat(messages)
        return self._parse_response(res)

