from fastapi import FastAPI
from routers.process_audio import router as care_router

app = FastAPI(
    title="Customer Care Multi-Agent Backend",
    version="1.0.0",
)

app.include_router(care_router)


@app.get("/")
async def root():
    return {"message": "Customer Care Backend is running."}
