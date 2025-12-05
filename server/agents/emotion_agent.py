# agents/emotion_agent.py
import os
import json  # 👈 추가
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .solar_client import solar_chat  # 👈 추가 (같은 패키지 안의 solar_client 사용)

# AI-BE/agents 기준으로 한 단계 올라가서(models 폴더 찾기)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "kobert_emotion_final")

tokenizer = AutoTokenizer.from_pretrained(
    "monologg/kobert",
    trust_remote_code=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()

if hasattr(model.config, "id2label") and model.config.id2label:
    id2label = {int(k): v for k, v in model.config.id2label.items()}
else:
    id2label = {0: "anger", 1: "sad", 2: "fear"}


class EmotionAgent:

    # -------------------------------
    # 1) 기존: 대표 감정 1개만 반환
    # -------------------------------
    def predict(self, text: str) -> dict:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            score, idx = torch.max(probs, dim=1)

        label = id2label[int(idx.item())]
        return {
            "emotion_label": label,
            "emotion_score": float(score),
        }

    # ---------------------------------------------------------
    # 2) 새로 추가: anger, sad, fear 모든 확률을 반환 (그래프용)
    # ---------------------------------------------------------
    def predict_proba(self, text: str) -> dict:
        """
        감정 3개(anger, sad, fear)의 확률 전체를 반환하는 함수.
        시각화(막대그래프/파이그래프) 만들 때 팀원이 그대로 사용 가능.
        """
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].tolist()

        return {
            "anger": float(probs[0]),
            "sad": float(probs[1]),
            "fear": float(probs[2]),
        }


# ===========================
# 3) Solar 기반 Emotion Agent
# ===========================
class SolarEmotionAgent:
    """
    Upstage Solar를 이용해 감정(anger, sad, fear)을 분류하는 에이전트.
    인터페이스를 기존 EmotionAgent와 최대한 비슷하게 맞춰서
    predict / predict_proba 둘 다 제공.
    """

    def __init__(self):
        # 필요하면 나중에 옵션 추가 가능
        pass

    def _call_solar(self, text: str) -> dict:
        """
        내부에서만 쓰는 함수: Solar에 실제로 요청 보내고 JSON 받기.
        """
        system_prompt = """
        너는 한국어 콜센터 고객 발화의 감정을 분류하는 분석가야.

        이 발화의 감정을 아래 세 가지 중에서만 선택해:
        - anger
        - sad
        - fear

        각 레이블에 대해 0.0~1.0 사이의 확률을 생각하고,
        가장 확률이 높은 레이블을 emotion_label 로,
        그 레이블의 확률을 emotion_score 로 반환해.

        반드시 아래 JSON 형식으로만, 다른 설명 없이 답해.

        {
          "emotion_label": "anger",
          "emotion_score": 0.87,
          "probs": {
            "anger": 0.87,
            "sad": 0.02,
            "fear": 0.11
          }
        }
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"고객 발화: {text}"},
        ]

        resp = solar_chat(
            messages,
            model="solar-1-mini-chat",
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 혹시 JSON 실패해도 코드 안 터지게 기본값
            data = {
                "emotion_label": "fear",
                "emotion_score": 0.0,
                "probs": {},
                "raw": content,
            }

        # 누락된 값들 기본값 세팅
        data.setdefault("emotion_label", "fear")
        data.setdefault("emotion_score", 0.0)
        data.setdefault("probs", {})

        return data

    def predict(self, text: str) -> dict:
        """
        기존 EmotionAgent.predict 와 같은 포맷:
        {"emotion_label": str, "emotion_score": float}
        """
        data = self._call_solar(text)
        return {
            "emotion_label": data.get("emotion_label", "fear"),
            "emotion_score": float(data.get("emotion_score", 0.0)),
        }

    def predict_proba(self, text: str) -> dict:
        """
        기존 EmotionAgent.predict_proba 와 같은 포맷:
        {"anger": float, "sad": float, "fear": float}
        """
        data = self._call_solar(text)
        probs = data.get("probs") or {}

        return {
            "anger": float(probs.get("anger", 0.0)),
            "sad": float(probs.get("sad", 0.0)),
            "fear": float(probs.get("fear", 0.0)),
        }
