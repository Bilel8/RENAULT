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

        # Lire le fichier WAV depuis les bytes
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file, dtype="float32")  # Doit être déjà 16 kHz mono

        # Transcription directe
        if len(audio.shape) > 1:  # Check if the audio has multiple channels
            audio = np.mean(audio, axis=1)  # Convert to mono by averaging channels

        # Resample the audio to 16 kHz (required by Whisper)
        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples)

        # Convert the audio to float32 (required by Whisper)
        audio = audio.astype(np.float32)

        # Trim or pad the audio to 30 seconds (required by Whisper)
        max_length = 30 * 16000  # 30 seconds at 16 kHz
        if len(audio) > max_length:
            audio = audio[:max_length]  # Trim to 30 seconds
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))  # Pad with zeros

        # Convert the audio to the format expected by Whisper
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Transcribe the audio
        options = whisper.DecodingOptions(fp16=False,
                                        task='transcribe',
                                        language='fr')
        result = whisper.decode(self.model, mel, options)

        # Return the transcription
        return result.text