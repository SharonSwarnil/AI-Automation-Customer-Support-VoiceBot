import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# Replace with your actual HF repo
MODEL_PATH = "SharonSwarnil/intent-classifier"

# Load tokenizer & model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Download label_map.json from HuggingFace repo
label_map_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="label_map.json"
)

with open(label_map_path, "r") as f:
    label_map = json.load(f)

# Convert keys from string → int
id_to_label = {int(k): v for k, v in label_map.items()}


def predict_intent(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Some models include this, some don’t
    inputs.pop("token_type_ids", None)

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    intent = id_to_label[predicted_class.item()]

    return intent, confidence.item()

