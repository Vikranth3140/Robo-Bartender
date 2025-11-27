# Conversational Bartender Assistant

This project implements a **voice-controlled AI bartender** that listens to the user's speech, converts it to text using **ElevenLabs Realtime STT**, reasons using **Groq LLM**, and responds with **ElevenLabs TTS**.
It also accepts **bottle color input** from an external vision module via a simple Python function call.

---

## Features

### **Realtime Speech Recognition (STT)**

* Uses **ElevenLabs Realtime Speech-to-Text**
* Detects silence and commits the transcript automatically
* Streams microphone audio in PCM16 chunks

### **Groq LLM Reasoning**

* Uses `moonshotai/kimi-k2-instruct`
* Maintains conversation history
* Follows strict prompts not to hallucinate bottle colors
* Makes cocktail recommendations based on visible bottles

### **Text-to-Speech (TTS)**

* Uses **ElevenLabs WebSocket Streaming TTS**
* Plays high-quality, low-latency speech using `sounddevice`

### **Cocktail Logic**

The assistant knows which cocktails are possible based on **visible bottle colors**:

| Cocktail     | Required Colors |
| ------------ | --------------- |
| Martini      | red + blue      |
| Mojito       | green + blue    |
| Bloody Mary  | red             |
| Blue Lagoon  | blue            |
| Cosmopolitan | red + green     |

---

## Architecture Overview

```
Microphone → ElevenLabs STT → Transcript
                ↓
            Conversation Manager → Groq LLM
                ↓
              Reply → ElevenLabs TTS → Audio Output
                ↑
        Visible Bottles (update via Python call)
```

## Configuration

Edit the config section at the top of the file:

```python
ELEVEN_API_KEY = "your_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"
TTS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
```
