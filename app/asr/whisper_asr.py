import whisper
from app.config import ASR_MODEL

model = whisper.load_model(ASR_MODEL)

def transcribe_audio(audio_path):

    result = model.transcribe(audio_path)

    text = result["text"].strip()

    return text
