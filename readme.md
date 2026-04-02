# AI Voicebot for Customer Support Automation

## Overview

This project implements an AI-powered voicebot that can handle basic customer support queries through speech interaction.

The system accepts voice input from a microphone or an uploaded audio file, converts speech into text, detects the user's intent, generates a response, and returns the reply as synthesized speech.

The goal of this project is to demonstrate an end-to-end machine learning pipeline combining speech recognition, natural language understanding, and speech synthesis.

---

## System Architecture

The system follows a simple modular pipeline:

User Voice  
↓  
Speech Recognition (Whisper ASR)  
↓  
Intent Classification (DistilBERT)  
↓  
Response Generation  
↓  
Text-to-Speech (gTTS)  
↓  
Audio Response

Architecture diagram:  
`architecture_diagram.png`

---

## Technologies Used

- Speech Recognition: **OpenAI Whisper**
- Intent Classification: **DistilBERT (HuggingFace Transformers)**
- Text-to-Speech: **gTTS**
- Backend API: **FastAPI**
- Frontend: **HTML + JavaScript**
- Language: **Python**

---

## Supported Customer Support Intents

The intent classifier supports the following intents:

- order_status
- cancel_order
- refund_request
- payment_issue
- address_change
- product_info
- delivery_delay
- subscription_issue
- account_help
- speak_agent

---

## Dataset

The intent classifier was trained on a custom dataset of customer support queries.

Example samples:

Where is my order → order_status  
Cancel my order → cancel_order  
I want a refund → refund_request  

Dataset file: data/intent_dataset.csv

---

## Model Choices

**Whisper ASR**

Whisper is used for speech recognition because it performs well with short conversational speech and can handle moderate background noise.

**DistilBERT**

DistilBERT was chosen for intent classification because it provides strong language understanding while remaining lightweight enough for fast inference.

**gTTS**

gTTS is used to convert the generated text response into natural sounding speech.

---

## Evaluation Metrics

The intent classification model was evaluated using common NLP metrics.

Accuracy: 0.98  
Precision: 0.97  
Recall: 0.97  
F1 Score: 0.97  

Speech recognition quality can also be evaluated using **Word Error Rate (WER)**, which measures how closely the transcription matches the spoken input.

---

## Project Structure

voicebot-customer-support

app/
asr/
intent/
pipeline/
response/
tts/
main.py

data/
intent_dataset.csv

samples/
order_status.wav
cancel_order.wav
refund_request.wav

static/
index.html

architecture_diagram.png
requirements.txt
README.md

---

## Installation

Install dependencies:

---

## Running the Application

Start the FastAPI server: uvicorn app.main:app --reload


Open the web interface: http://127.0.0.1:8000

---

## Voice Interaction

Click **Start Recording** and speak a query such as:
"Where is my order? "

The system will:
1. convert speech to text
2. detect the user intent
3. generate a response
4. return audio speech output

---

## API Testing

The backend API can be tested using FastAPI documentation.

Open: http://127.0.0.1:8000/docs

Upload an audio file from the `samples` folder to test the `/voicebot` endpoint.

---

## Sample Audio Files

Example test audio files are provided:

samples/order_status.wav
samples/cancel_order.wav
samples/refund_request.wav

---

## Demo

The repository includes a short demo video showing:

- project overview
- system architecture
- frontend voice interaction
- backend API testing

---

## Conclusion

This project demonstrates a complete voice-based customer support assistant using speech recognition, transformer-based intent classification, and speech synthesis.

The system is modular, easy to run locally, and can be extended for more advanced conversational support systems.
