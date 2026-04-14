from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

from app.pipeline.voicebot_pipeline import run_pipeline

app = FastAPI(title="AI Voicebot Customer Support")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/voicebot")
async def voicebot(audio: UploadFile = File(...)):

    temp_audio = "temp_input.wav"

    with open(temp_audio, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    text, intent, confidence, response, audio_file = run_pipeline(temp_audio)

    return {
        "transcription": text,
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "audio_file": audio_file
    }
    