from fastapi import APIRouter, File, Depends, HTTPException
from app.api.dependencies import get_asr, get_rag, get_llm, get_tts
from app.models.schemas import ChatResponse
import logging
import base64

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def voice_chat(
    audio: bytes = File(...),
    asr = Depends(get_asr),
    llm = Depends(get_llm),
    tts = Depends(get_tts),
):
    try:
        logging.debug(f"Received audio of size: {len(audio)} bytes")
        
        # 1. ASR
        transcription = asr.transcribe(audio)

        # 2. LLM
        # RAG plus tard ; pour l’instant docs=None
        answer = llm.generate(transcription, docs=None)
        
        # 3. TTS
        audio_reply = tts.synthesize(answer)
        
        # 4. Encoding Base64
        audio_b64 = base64.b64encode(audio_reply).decode('utf-8')

        return ChatResponse(
            transcription=transcription,
            answer=answer,
            audio_reply=audio_b64,  
        )
    except Exception as e:
        logging.error(f"Error in voice_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
