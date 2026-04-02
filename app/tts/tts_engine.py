from gtts import gTTS
import os
import uuid

STATIC_DIR = "static"
MAX_FILES = 50  # limit for cleanup


def synthesize_speech(text):

    # Ensure static folder exists
    os.makedirs(STATIC_DIR, exist_ok=True)

    #  Cleanup old files if limit exceeded
    files = os.listdir(STATIC_DIR)
    if len(files) > MAX_FILES:
        files.sort(key=lambda x: os.path.getctime(os.path.join(STATIC_DIR, x)))
        for file in files[:40]:  # delete oldest 40 files
            try:
                os.remove(os.path.join(STATIC_DIR, file))
            except Exception:
                pass

    # Generate unique filename
    file_name = f"response_{uuid.uuid4().hex}.wav"
    file_path = os.path.join(STATIC_DIR, file_name)

    # Generate speech
    tts = gTTS(text)
    tts.save(file_path)

    # Return API path
    return f"/static/{file_name}"
