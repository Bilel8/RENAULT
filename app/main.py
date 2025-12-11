from fastapi import FastAPI
from app.api.routes.voice import router as voice_router

app = FastAPI(title="Voice Chatbot")

app.include_router(voice_router, prefix="/voice")