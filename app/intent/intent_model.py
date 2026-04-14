import os
import torch
import json
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Detect environment
USE_API = os.getenv("USE_HF_API", "false").lower() == "true"

# HuggingFace config
HF_API_URL = "https://api-inference.huggingface.co/models/SharonSwarnil/intent-classifier"
HF_TOKEN = os.getenv("HF_TOKEN")

if not USE_API:
    # LOCAL MODEL (Option B)
    MODEL_PATH = "app/intent/model"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    with open(os.path.join(MODEL_PATH, "label_map.json"), "r") as f:
        label_map = json.load(f)

    id_to_label = {int(k): v for k, v in label_map.items()}


def predict_intent(text):

    # OPTION A → API MODE
    if USE_API:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": text}

        response = requests.post(HF_API_URL, headers=headers, json=payload)
        result = response.json()

        top = max(result[0], key=lambda x: x["score"])
        return top["label"], top["score"]

    # OPTION B → LOCAL MODE
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs.pop("token_type_ids", None)

        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        confidence, predicted_class = torch.max(probs, dim=1)
        intent = id_to_label[predicted_class.item()]

        return intent, confidence.item()
    