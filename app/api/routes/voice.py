from fastapi import APIRouter, File, Depends
from app.api.dependencies import get_asr, get_rag, get_llm, get_tts
from app.models.schemas import ChatResponse
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def voice_chat(
    audio: bytes = File(...),
    asr = Depends(get_asr),
    llm = Depends(get_llm),
    tts = Depends(get_tts),
):
    logging.debug(f"Received audio of size: {len(audio)} bytes")
    transcription = asr.transcribe(audio)

    # RAG plus tard ; pour l’instant docs=None
    answer = llm.generate(transcription, docs=None)
    audio_reply = tts.synthesize(answer)

    return ChatResponse(
        transcription=transcription,
        answer=answer,
        audio_reply=audio_reply.hex(),  
    )
