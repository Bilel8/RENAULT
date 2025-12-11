from app.services.asr import ASRService
from app.services.rag import RAGService
from app.services.llm import LLMService
from app.services.tts import TTSService

asr_service = ASRService()
rag_service = RAGService()
llm_service = LLMService()
tts_service = TTSService()


def get_asr():
    return asr_service


def get_rag():
    return rag_service


def get_llm():
    return llm_service


def get_tts():
    return tts_service