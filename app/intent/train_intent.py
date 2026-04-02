import os
import json
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/intent_dataset.csv"
MODEL_SAVE_PATH = "app/intent/model"

BATCH_SIZE = 8
EPOCHS = 10
MAX_LEN = 64
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def main():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    texts = df["text"].tolist()
    intents = df["intent"].tolist()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)

    label_map = {int(i): label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(MODEL_SAVE_PATH, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)

        val_accuracy = correct / total if total > 0 else 0
        print(f"Validation accuracy: {val_accuracy:.4f}")

    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print("\nModel training complete")
    print("Model saved to:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
    