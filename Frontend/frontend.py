import gradio as gr
import requests
import time

FASTAPI_URL = "http://127.0.0.1:8000/voice/chat"

def send_audio(audio):
    """
    audio = (sample_rate, numpy_array)
    Gradio fournit l'audio brut.
    On l'encode en WAV et on l'envoie au backend.
    """
    if audio is None:
        return "Aucun audio reçu.", "Il faut parler si tu veux une reponse bg.", None

    sr, data = audio
    print("dtype:", data.dtype)

    # Gradio fournit un numpy array → convertir en WAV en mémoire
    import soundfile as sf
    import io
    import tempfile
    import os

    buffer = io.BytesIO()
    sf.write(buffer, data, sr, format="WAV")
    buffer.seek(0)

    files = {"audio": ("audio.wav", buffer, "audio/wav")}
    t0 = time.perf_counter()
    response = requests.post(FASTAPI_URL, files=files)

    if response.status_code == 200:

        data = response.json()
        transcription = data.get("transcription", "")
        # answer = data.get("answer", "")

        # audio_path = None
        # audio_hex = data.get("audio_reply")
        # if audio_hex :
        #     audio_bytes = bytes.fromhex(audio_hex)
        #     tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        #     tmp.write(audio_bytes)
        #     tmp.close()
        #     audio_path = tmp.name
        
        t1 = time.perf_counter()
        print("La latence vaut", (t1-t0)*1000)

        return transcription#, answer, audio_path

    else:
        return  f"Erreur backend. Status code: {response.status_code}, Content: {response.content}", "", None

with gr.Blocks() as app:
    gr.Markdown("# Assistant Vocal – Renault")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="Clique pour enregistrer / arrêter",
        interactive=True,
    )

    button = gr.Button("Envoyer la question")
    transcription_out = gr.Textbox(label="Transcription (ASR)")
    # llm_out = gr.Textbox(label="Réponse (LLM)")
    # tts_audio_out = gr.Audio("Audio généré", type="filepath")
    button.click(send_audio, inputs=audio_input, outputs=[transcription_out
    #, llm_out, tts_audio_out
    ])

app.launch()