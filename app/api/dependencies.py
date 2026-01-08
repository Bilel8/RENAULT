from functools import lru_cache

from app.services.asr import ASRService
from app.services.llm import LLMService, LLMConfig
from app.services.tts import TTSService, TTSConfig


# CONFIG LLM 

LLM_MODEL_PATH = ("/home/virgaux/Desktop/chatbot/models/model_llm/""qwen2.5-3b-instruct-q4_k_m.gguf")
LLM_N_CTX = 2048
LLM_N_GPU_LAYERS = 0
LLM_MAX_TOKENS = 256

# CONFIG TTS 
PIPER_BIN = "piper"
PIPER_MODEL_PATH = "/home/virgaux/Desktop/chatbot/models/model_tts/fr_FR-upmc-medium.onnx"
PIPER_CONFIG_PATH = "/home/virgaux/Desktop/chatbot/models/model_tts/fr_FR-upmc-medium.onnx.json"
PIPER_LENGTH_SCALE = 1.0  


@lru_cache(maxsize=1)
def get_rag():
    return None


@lru_cache(maxsize=1)
def get_asr():
    # Charge Whisper une seule fois
    return ASRService()


# @lru_cache(maxsize=1)
# def get_llm():
#     cfg = LLMConfig(
#         model_path=LLM_MODEL_PATH,
#         n_ctx=LLM_N_CTX,
#         n_gpu_layers=LLM_N_GPU_LAYERS,
#         max_tokens=LLM_MAX_TOKENS,
#     )
#     return LLMService(cfg)


# @lru_cache(maxsize=1)
# def get_tts():
#     cfg = TTSConfig(
#         piper_bin=PIPER_BIN,
#         model_path=PIPER_MODEL_PATH,
#         config_path=PIPER_CONFIG_PATH,
#         length_scale=PIPER_LENGTH_SCALE,
#     )
#     return TTSService(cfg)