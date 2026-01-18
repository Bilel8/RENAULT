import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM CONFIG ---
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "/home/virgaux/Desktop/chatbot/models/model_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "512"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "Tu es un assistant vocal embarqué sur un Jetson Orin Nano. "
    "Réponds en français, de façon concise, claire, et utile. "
    "Si tu ne sais pas, dis-le."
)

# --- TTS CONFIG ---
PIPER_BIN = os.getenv("PIPER_BIN", "piper")
PIPER_MODEL_PATH = os.getenv(
    "PIPER_MODEL_PATH",
    "/home/virgaux/Desktop/chatbot/models/model_tts/fr_FR-upmc-medium.onnx"
)
PIPER_CONFIG_PATH = os.getenv(
    "PIPER_CONFIG_PATH",
    "/home/virgaux/Desktop/chatbot/models/model_tts/fr_FR-upmc-medium.onnx.json"
)
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))

# --- ASR CONFIG ---
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base")
ASR_SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))

# --- LOGGING CONFIG ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")


CSV_PATH = "/home/virgaux/Desktop/chatbot/bdd.csv"