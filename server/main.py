from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from routers import process_audio

app = FastAPI(title="AI Customer Care Backend")

# /api/analyze_call
app.include_router(process_audio.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "AI Customer Care Backend is running"}


# -----------------------------
# Render에서 필요한 실행 부분
# -----------------------------
import os

port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)
