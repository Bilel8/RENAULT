import whisper
import soundfile as sf
import io
import numpy as np
from scipy.signal import resample


class ASRService:
    def __init__(self, model_name: str = "small"):
        # Charge un modèle Whisper une seule fois
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_bytes: bytes) -> str:

        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file, dtype="float32")

        # Transcription directe
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples)

        audio = audio.astype(np.float32)

        # Utilisation de l'API de haut niveau transcribe qui gère:
        # - Les fichiers longs (sliding window)
        # - La ponctuation et le formatage automatique
        # - Pas de limite de 30s
        # fp16=True est le défaut sur GPU, mais peut générer un warning sur CPU. 
        # On le garde à True explicitement comme dans l'implémentation précédente si souhaité, 
        # ou on laisse whisper gérer (par défaut il tente True et fallback si CPU).
        # Ici on force comme avant, mais on peut le changer si warning.
        result = self.model.transcribe(audio, language="fr")

        return result["text"]