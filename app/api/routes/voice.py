from fastapi import APIRouter, File, Depends
from app.api.dependencies import get_asr, get_rag, get_llm, get_tts
from app.models.schemas import ChatResponse

router = APIRouter() 

@router.post("/chat", response_model=ChatResponse)
async def voice_chat(
    audio: bytes = File(...),
    asr = Depends(get_asr),
    rag = Depends(get_rag),
    llm = Depends(get_llm),
    tts = Depends(get_tts),
):
    transcription = asr.transcribe(audio)
    docs = rag.retrieve(transcription)
    answer = llm.generate(transcription, docs)
    audio_reply = tts.synthesize(answer)

    return ChatResponse(
        transcription=transcription,
        answer=answer,
        audio_reply=audio_reply.hex()
    )