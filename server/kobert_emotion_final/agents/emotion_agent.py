# agents/emotion_agent.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

