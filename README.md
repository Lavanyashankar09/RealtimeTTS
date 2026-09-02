# RealtimeTTS

RealtimeTTS is a local text-to-speech studio built with FastAPI and Microsoft's SpeechT5. Type up to 500 characters and speech streams back chunk by chunk as you pause, continuing as you add more words. Play, replay, stop, or download the finished WAV from the browser.

The application is text-to-speech only.

The application is text-to-speech only. It does not record a microphone, transcribe speech, or send text to a hosted application server. Hugging Face is contacted when model files are downloaded for the first time.

## Setup

Python 3.11 is recommended. From this project directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the single local server, bound only to the loopback interface:

```bash
uvicorn app:app --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The browser connects automatically to the same server over `/ws`; do not open `client.html` directly.

## First generation

On the first synthesis request, the server downloads and loads the SpeechT5 processor, TTS model, HiFi-GAN vocoder, and CMU Arctic speaker embedding dataset. This requires an internet connection and can take several minutes. Later requests reuse those resources.

SpeechT5 inference is compute intensive. CPU generation can take noticeably longer than the spoken audio, especially for longer passages. The server processes one generation at a time to avoid competing model calls.

### macOS on Apple silicon

Use a native arm64 Python installation, ideally Python 3.11. If `soundfile` cannot find `libsndfile`, install it with Homebrew and reinstall the Python dependency:

```bash
brew install libsndfile
pip install --force-reinstall soundfile
```

PyTorch automatically uses the supported local build. This app currently performs the proven SpeechT5 inference path on CPU rather than forcing the experimental MPS path.

## Studio controls

Generation is automatic. Roughly 300 ms after you stop typing, the completed words are sent to the local SpeechT5 model; there is no Generate button. Words you add onto the end are queued behind whatever is already playing, so the passage continues instead of restarting.

- **Stop** stops playback and abandons the queue. Whatever was already generated stays available to replay or download. The model call in flight finishes in the background.
- **Replay** starts the finished take from the beginning.
- **Download WAV** saves the completed take as a single WAV file.
- **Copy text** copies the editor contents to the clipboard.
- **Clear** removes the text and current audio.

Keep only one tab open. Each tab holds its own WebSocket and generates independently, so two tabs speak over each other and compete for the same model.

## How it works

Speech is generated in chunks and streamed over a WebSocket as you type, so playback starts before the whole passage has been synthesized.

For the full explanation of the model, the chunking strategy, the gapless playback, and how the append queue works, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Measured latency

Taken on an Apple silicon Mac, CPU inference, model already warm. Median of three runs each, measured over a real WebSocket from the request being sent to the first `audio_chunk` arriving back.

| Passage | Time to first sound | Total generation | Audio produced | Chunks | Realtime factor |
|---|---|---|---|---|---|
| Short (4 words) | 605 ms | 0.61 s | 1.31 s | 1 | 0.46x |
| Medium (12 words) | 1081 ms | 2.29 s | 4.90 s | 3 | 0.47x |
| Long (34 words) | 1125 ms | 6.26 s | 13.41 s | 7 | 0.47x |

Two things matter here.

**Time to first sound stays roughly flat.** The long passage is more than eight times the length of the short one, but you start hearing it after about the same delay. That is the entire point of chunking: you wait for the first chunk, not the whole passage. Without streaming, the long passage would be silent for the full 6.26 s.

**The realtime factor is about 0.47x.** Generation takes roughly half as long as the audio takes to play, so every chunk is ready well before the previous one finishes. The queue never runs dry and the speech does not stutter. If this number ever crept above 1.0x, streaming would break down and you would hear gaps.

Add roughly 300 ms of typing debounce on top of the first-sound figure for the delay you actually perceive while typing.

The header shows the live measurement for your own machine: the time to the first chunk of the current take, and the realtime factor once it finishes.

## WebSocket protocol

Client to server:

| Field | Meaning |
|---|---|
| `request_id` | Non-empty string, echoed on every reply |
| `text` | The text to speak. For `append` this is only the new words |
| `mode` | `replace` (default) or `append` |

Server to client, all carrying the originating `request_id`:

| Type | Meaning |
|---|---|
| `status` | `loading` on first model load, then `generating` per chunk |
| `audio_chunk` | Base64 WAV for one chunk, with `index`, `total`, and its `text` |
| `done` | Every chunk for that request has been sent |
| `error` | Validation failure or synthesis failure |

## Tests

The test suite injects a fake TTS service. It does not load models, access Hugging Face, or require network access.

```bash
pytest -q
```

The tests cover the root UI, WebSocket validation, loading and generation states, structured failures, request ID preservation, text chunking, append queueing, replace superseding queued work, and streamed `RIFF` WAV chunks followed by a `done` message.
