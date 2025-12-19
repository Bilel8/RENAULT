import whisper
import soundfile as sf
import io
import numpy as np
from scipy.signal import resample


class ASRService:
    def __init__(self):
        # Charge un modèle Whisper une seule fois
        self.model = whisper.load_model("base")

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcrit un fichier WAV PCM 16 kHz mono déjà compatible Whisper.
        Aucun besoin de ffmpeg ou de conversion.
        """

        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file, dtype="float32")

        # Transcription directe
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples)

        audio = audio.astype(np.float32)

        max_length = 30 * 16000
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))

        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        options = whisper.DecodingOptions(fp16=False,
                                        task='transcribe',
                                        language='fr')
        result = whisper.decode(self.model, mel, options)

        return result.text