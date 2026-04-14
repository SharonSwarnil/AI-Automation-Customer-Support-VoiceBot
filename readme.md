# AI Voicebot Customer Support System

This project is an end-to-end AI-based voicebot that can handle basic customer support queries using voice input. It converts speech to text, identifies user intent, and responds back with audio output.

---

## Features

- Voice Input through browser
- Intent Detection using NLP model (DistilBERT)
- Automated response generation
- Text-to-Speech reply
- FastAPI backend
- Modular pipeline (ASR → NLP → TTS)

---

## Tech Stack

- Python, FastAPI  
- HuggingFace Transformers  
- Whisper (Speech-to-Text)  
- gTTS (Text-to-Speech)  
- HTML, JavaScript (Frontend)  

---

## Architecture

1. Audio Input → Whisper ASR  
2. Text → Intent Classification Model  
3. Intent → Response Generator  
4. Response → gTTS Audio Output  

---

## Live Demo

*(Will be added after deployment on Render)*

---

## Dataset

Custom dataset created for intent classification including:

- Order Status  
- Refund Request  
- Cancel Order  
- Customer Support  
- Payment Issues  
- etc.  

---

## Key Highlights

- Built a complete pipeline from voice input to audio response  
- Implemented fallback logic for low-confidence predictions  
- Designed modular code structure for easy extension  
- Focused on real-world customer support use cases  

---

## Ongoing Improvements

Currently, the project is working on a **local model setup** for accurate predictions.

I am actively working on:

- Integrating HuggingFace Inference API (to avoid loading heavy models locally)
- Deploying the project on Render for public access
- Optimizing API-based inference for better performance and scalability
- Improving intent accuracy and handling edge cases

Due to memory and deployment constraints, API integration is still under refinement, but the working pipeline is fully functional in the local environment.

---

## Author

**Sharon Swarnil Choudhary**  
B.Tech (Artificial Intelligence & Data Science)
