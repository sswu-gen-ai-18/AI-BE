# agents/solar_client.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Upstage 공식 가이드: base_url="https://api.upstage.ai/v1" :contentReference[oaicite:0]{index=0}
client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1",
)

def solar_chat(messages, model: str = "solar-1-mini-chat", **kwargs):
    """
    Upstage Solar용 chat 래퍼.
    messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
