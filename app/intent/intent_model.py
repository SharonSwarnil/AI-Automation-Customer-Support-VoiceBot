import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

MODEL_PATH = "app/intent/model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Load label mapping saved during training
label_map_path = os.path.join(MODEL_PATH, "label_map.json")

with open(label_map_path, "r") as f:
    label_map = json.load(f)

# convert keys from string → int
id_to_label = {int(k): v for k, v in label_map.items()}


def predict_intent(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs.pop("token_type_ids", None)
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    intent = id_to_label[predicted_class.item()]

    return intent, confidence.item()


