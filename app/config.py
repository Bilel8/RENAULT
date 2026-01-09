import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM CONFIG ---
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "")
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "2048"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", str(max(os.cpu_count() or 4, 4))))
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
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
PIPER_CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH", "")
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))

# --- ASR CONFIG ---
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base")
ASR_SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
