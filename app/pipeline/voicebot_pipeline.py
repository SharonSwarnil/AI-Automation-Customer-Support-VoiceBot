from app.asr.whisper_asr import transcribe_audio
from app.intent.intent_model import predict_intent
from app.response.response_generator import generate_response
from app.tts.tts_engine import synthesize_speech

CONFIDENCE_THRESHOLD = 0.70

def run_pipeline(audio_path):

    text = transcribe_audio(audio_path)
    intent, confidence = predict_intent(text)
    text_lower = text.lower()

    if confidence < CONFIDENCE_THRESHOLD:

        if "support" in text_lower or "agent" in text_lower:
            intent = "speak_agent"

        elif "refund" in text_lower or "money back" in text_lower:
            intent = "refund_request"

        elif "cancel" in text_lower:
            intent = "cancel_order"

        else:
            intent = "fallback"

    response_text = generate_response(intent)
    audio_url = synthesize_speech(response_text)

    return text, intent, confidence, response_text, audio_url


