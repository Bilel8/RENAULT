import gradio as gr
import requests

FASTAPI_URL = "http://127.0.0.1:8000/voice/chat"

def send_audio(audio):
    """
    audio = (sample_rate, numpy_array)
    Gradio fournit l'audio brut.
    On l'encode en WAV et on l'envoie au backend.
    """
    if audio is None:
        return "Aucun audio reçu."

    sr, data = audio

    # Gradio fournit un numpy array → convertir en WAV en mémoire
    import soundfile as sf
    import io
    import numpy as np

    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(data) * 16000 / sr)
        data = resample(data, num_samples)
        sr = 16000

    # Convert to mono if necessary
    if len(data.shape) > 1:  # If stereo, average the channels
        data = np.mean(data, axis=1)

    buffer = io.BytesIO()
    sf.write(buffer, data, sr, format="WAV")
    buffer.seek(0)

    files = {"audio": ("audio.wav", buffer, "audio/wav")}
    response = requests.post(FASTAPI_URL, files=files)

    if response.status_code == 200:
        return response.json()["transcription"] + "cool"
    else:
        return  f"Erreur backend. Status code: {response.status_code}, Content: {response.content}"

with gr.Blocks() as app:
    gr.Markdown("# Assistant Vocal – Gradio Frontend")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="Clique pour enregistrer / arrêter",
        interactive=True,
    )

    button = gr.Button("Envoyer la question")
    output = gr.Textbox(label="Réponse")

    button.click(send_audio, inputs=audio_input, outputs=output)

app.launch()