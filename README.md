
### Overview

This project implements a **real-time Text-to-Speech (TTS)** service using a **WebSocket** interface.
It leverages Microsoft’s **SpeechT5** model to generate high-quality, natural-sounding speech.
The setup includes a simple **client UI**, **WebSocket server**, and **testing script** to demonstrate end-to-end functionality.

---

### 🚀 Setup Instructions

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Run the Server

Open and run **server.ipynb** — this starts the WebSocket service locally.

#### 3. Launch the Client

Open **client.html** in your browser.
It connects to the running WebSocket and streams text input to the TTS engine in real time.


---

### 🧠 How It Works

1. The **client** sends streamed text input to the WebSocket server.
2. The **SpeechT5 model** converts the text into an audio waveform.
3. The waveform is **Base64-encoded** and sent back to the client in real time.
4. The **browser client** decodes and plays the audio instantly for seamless text-to-speech conversion.
