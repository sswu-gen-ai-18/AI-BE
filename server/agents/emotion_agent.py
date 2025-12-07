# server/agents/emotion_agent.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download  # ← 이 줄 있는지 확인!

# Hugging Face에 만든 리포 이름
MODEL_REPO = "hozziii/kobert-emotion-final"

# AI-BE/server/agents 기준으로 한 단계 올라가서(models 폴더 찾기)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "kobert_emotion_final")

# 1) 모델이 없으면 HuggingFace에서 다운로드
os.makedirs(MODEL_DIR, exist_ok=True)
local_model_file = os.path.join(MODEL_DIR, "model.safetensors")

if not os.path.exists(local_model_file):
    print("[EmotionAgent] 로컬에 KoBERT 감정 모델이 없어 HF에서 다운로드합니다...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.msgpack"],  # 있어도 되고 없어도 됨
    )
    print("[EmotionAgent] 다운로드 완료:", MODEL_DIR)

# 2) 토크나이저 & 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(
    "monologg/kobert",
    trust_remote_code=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,              # HF에서 받은 폴더에서 바로 로딩
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

    # 대표 감정 1개만 반환
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
