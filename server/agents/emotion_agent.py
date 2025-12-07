# server/agents/emotion_agent.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download  # HF에서 모델 다운로드

# 🔹 Hugging Face에 올린 네 모델 리포 이름
MODEL_REPO = "hozziii/kobert-emotion-final"

# 🔹 AI-BE/server/agents 기준으로 한 단계 올라가서(models 폴더 찾기)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "kobert_emotion_final")

# 🔹 로컬 모델 디렉토리 준비 & 없으면 HF에서 받아오기
os.makedirs(MODEL_DIR, exist_ok=True)
local_model_file = os.path.join(MODEL_DIR, "model.safetensors")

if not os.path.exists(local_model_file):
    print("[EmotionAgent] 로컬에 KoBERT 감정 모델이 없어 HF에서 다운로드합니다...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.msgpack"],  # 선택 옵션
    )
    print("[EmotionAgent] 다운로드 완료:", MODEL_DIR)

# 🔹 토크나이저 & 모델 로딩
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

# 🔹 id2label 설정 (모델 config에 있으면 그것 우선)
if hasattr(model.config, "id2label") and model.config.id2label:
    id2label = {int(k): v for k, v in model.config.id2label.items()}
else:
    # 기본 매핑 (모델이 3-class라고 가정)
    id2label = {0: "anger", 1: "sad", 2: "fear"}


# ✅ 인삿말/형식 멘트 패턴 (무조건 neutral로 처리할 후보들)
GREETING_PATTERNS = [
    "안녕하세요",
    "안녕하십니까",
    "전화드렸습니다",
    "전화 드렸습니다",
    "도와주셔서 감사합니다",
    "도와 주셔서 감사합니다",
    "감사합니다",
    "수고하세요",
    "수고하셨습니다",
]

# ✅ 감정 확신도가 너무 낮으면 neutral로 돌리는 threshold
NEUTRAL_THRESHOLD = 0.55


class EmotionAgent:
    """
    KoBERT 기반 감정 분류 에이전트
    """

    # 대표 감정 1개만 반환
    def predict(self, text: str) -> dict:
        # -----------------------------
        # 0) 인삿말/형식 멘트 휴리스틱
        # -----------------------------
        cleaned = text.strip()
        no_space = cleaned.replace(" ", "")

        for pattern in GREETING_PATTERNS:
            if pattern.replace(" ", "") in no_space:
                # 인삿말류는 강한 감정이 없다고 보고 neutral 고정
                return {
                    "emotion_label": "neutral",
                    "emotion_score": 0.7,  # 적당한 중간값
                }

        # -----------------------------
        # 1) KoBERT 모델 추론
        # -----------------------------
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

        score_val = float(score)
        label = id2label[int(idx.item())]

        # -----------------------------
        # 2) 확신도가 낮으면 neutral로 강등
        # -----------------------------
        if score_val < NEUTRAL_THRESHOLD:
            return {
                "emotion_label": "neutral",
                "emotion_score": score_val,
            }

        # -----------------------------
        # 3) 일반적인 감정 결과 반환
        # -----------------------------
        return {
            "emotion_label": label,
            "emotion_score": score_val,
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

        # 여기서는 여전히 모델이 가진 3개 클래스 분포 그대로 반환
        # (프론트 그래프용)
        return {
            "anger": float(probs[0]),
            "sad": float(probs[1]),
            "fear": float(probs[2]),
        }
