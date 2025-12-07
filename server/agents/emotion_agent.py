# server/agents/emotion_agent.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

MODEL_REPO = "hozziii/kobert-emotion-final"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "kobert_emotion_final")

os.makedirs(MODEL_DIR, exist_ok=True)
local_model_file = os.path.join(MODEL_DIR, "model.safetensors")

if not os.path.exists(local_model_file):
    print("[EmotionAgent] 로컬에 KoBERT 감정 모델이 없어 HF에서 다운로드합니다...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.msgpack"],
    )
    print("[EmotionAgent] 다운로드 완료:", MODEL_DIR)

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
    """
    KoBERT 기반 감정 분류 에이전트
    """

    # 🔹 대표 감정 1개만 반환 (+ 인사/중립 휴리스틱으로 neutral 처리)
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
            score_tensor, idx_tensor = torch.max(probs, dim=1)

        label = id2label[int(idx_tensor.item())]
        score = float(score_tensor.item())

        # -------------------------------
        # 🔸 인사/중립 문장 휴리스틱 처리
        # -------------------------------
        clean = (text or "").strip().replace(" ", "")

        greeting_phrases = [
            "안녕하세요",
            "안녕하십니까",
            "여보세요",
            "감사합니다",
            "수고하세요",
            "네", "예", "알겠어요",
        ]

        is_greeting = any(p in clean for p in greeting_phrases)
        is_very_short = len(clean) <= 3        # 너무 짧은 단답
        low_confidence = score < 0.6          # 모델 확신도 낮을 때만 중립으로 덮어쓰기

        if (is_greeting or is_very_short) and low_confidence:
            # 👉 이런 경우는 그냥 neutral 로 강제 캐스팅
            return {
                "emotion_label": "neutral",
                "emotion_score": score,  # 혹은 0.0 으로 고정해도 됨
            }

        # 기본: KoBERT 결과 그대로 사용
        return {
            "emotion_label": label,
            "emotion_score": score,
        }

    # anger, sad, fear 전체 확률 반환 (그래프용)
    def predict_proba(self, text: str) -> dict:
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
