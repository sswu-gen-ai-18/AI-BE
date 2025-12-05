from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from server.routers import process_audio

app = FastAPI(title="AI Customer Care Backend")

# /api/analyze_call
app.include_router(process_audio.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "AI Customer Care Backend is running"}