import whisper
import soundfile as sf
import io

class ASRService:
    def __init__(self):
        # Charge un modèle Whisper une seule fois
        self.model = whisper.load_model("base")

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcrit un fichier WAV PCM 16 kHz mono déjà compatible Whisper.
        Aucun besoin de ffmpeg ou de conversion.
        """

        # Lire le fichier WAV depuis les bytes
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file)  # Doit être déjà 16 kHz mono

        # Transcription directe
        result = self.model.transcribe(audio, fp16=False)

        return result["text"]