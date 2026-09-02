# How this code works

A plain walkthrough of what the app does and how it got here.

## The model

We use **SpeechT5**, Microsoft's text-to-speech model, from Hugging Face. Three pieces work together:

1. **`SpeechT5Processor`** turns your text into tokens the model understands.
2. **`SpeechT5ForTextToSpeech`** turns those tokens into a mel-spectrogram, which is a picture of the sound.
3. **`SpeechT5HifiGan`** (the vocoder) turns that spectrogram into an actual waveform you can hear.

A fourth piece, the **speaker embedding**, decides *whose* voice it is. It is a single vector from the CMU Arctic dataset that carries pitch, tone, and accent. We always use index 7306, which is why the voice sounds the same every time.

All of this lives in `SpeechT5Service` in `app.py`.

## Version 1: give it text, get back a voice

The first version was the obvious one. You typed a sentence, clicked a button, and the whole sentence went into the model at once:

```
your text ──> processor ──> model ──> vocoder ──> one WAV file ──> play it
```

It worked. The problem was waiting. Nothing came out of the speakers until the *last* word had finished generating. A short sentence meant a short wait, but a long paragraph meant staring at a silent screen for a long time. The longer you wrote, the worse it got.

There was a second, bigger problem. The model was being loaded from scratch, and the speaker embedding was being pulled by downloading the entire CMU Arctic dataset. That made startup crawl.

## Fixing the startup

Two changes:

- **Load the model once and keep it.** `SpeechT5Service` loads the processor, model, and vocoder on the first request only, then holds them in memory for as long as the server runs. Every request after that reuses them. This is why the first generation after starting the server is slow and everything after it is quick.
- **Read one vector instead of a whole dataset.** Instead of loading the full CMU Arctic dataset, we open the cached x-vector archive with `zipfile` and read out only the one embedding we need.

## Version 2: real-time streaming

The core idea: **you should not have to wait for the end of the sentence to hear the beginning of it.**

So instead of sending the whole passage to the model at once, we cut it into pieces and send each piece as soon as it is ready.

### Cutting the text

`chunk_text()` in `app.py` splits your text, and it tries to cut in places where a human speaker would naturally pause:

1. First it tries **sentence endings** (`.` `!` `?`).
2. If a sentence is still too long, it tries **clause punctuation** (`,` `;` `:`).
3. If it is *still* too long, it splits into **evenly sized word groups** of at most eight words.

That last step splits evenly on purpose. A greedy split would leave a stray tail like `signal.` alone in its own chunk, and the model would then read that one word as if it were a whole sentence, which sounds wrong.

### Sending the pieces

A **WebSocket** connects the browser to the server once, when the page loads, and stays open. This matters because a normal HTTP request is one question and one answer. A WebSocket lets the server keep pushing messages whenever it wants, which is exactly what we need when one piece of text produces many pieces of audio.

For each chunk the server sends an `audio_chunk` message, and when it runs out of chunks it sends `done`.

```
text ──> chunk 1 ──> synthesize ──> send ──┐
     ──> chunk 2 ──> synthesize ──> send ──┼──> browser plays them in order
     ──> chunk 3 ──> synthesize ──> send ──┘
```

The win: sound starts when the **first** chunk is ready instead of the last one. Because the model generates faster than the audio takes to play, each next chunk is ready before the previous one finishes, so the audio never runs dry.

### Playing the pieces without gaps

This part is easy to get wrong. The obvious approach is to play chunk 2 when chunk 1 finishes. But "when chunk 1 finishes" is something JavaScript finds out slightly late, so you hear a small gap at every seam, and the speech sounds chopped up.

Instead the browser uses the **Web Audio API**. Each chunk is decoded into an audio buffer, and then *scheduled* to start at an exact moment on the audio clock:

```
start of next chunk = where the previous chunk ends
```

The audio hardware handles the handoff, so the seams are exact and the speech sounds continuous.

## Version 3: continuing as you type

The button is gone. Now the app watches what you type and speaks it on its own, about 300 ms after you stop.

The first attempt got this wrong. Every time you typed, it cancelled whatever was playing and started the whole passage again from the beginning. Annoying.

So now the client compares what is in the box against what it has already spoken:

- **You added words to the end.** Only the new words get sent, marked `append`. The server puts them in a queue behind whatever it is already working on, and the browser schedules the audio after whatever is still playing. The passage continues.
- **You edited something already spoken.** The whole text gets resent, marked `replace`. That audio is already queued and cannot be un-said, so the only honest option is to start over. The server bumps a counter that throws away all the old work, and the browser stops its scheduled audio.

One more detail: if you pause in the middle of typing a word, the app holds that half-typed word back rather than speaking a fragment. A longer pause of about 1.2 seconds flushes it, so nothing gets stranded.

## How the server juggles all this

Inside `app.py` the WebSocket handler runs two things at the same time:

- A **reader** that does nothing but listen for your messages and put them in a queue.
- A **worker** that pulls from that queue and does the slow work of generating audio.

They are separate on purpose. If a single loop had to both listen and generate, it would be deaf while it was generating, and your new words would pile up unheard until it finished. Splitting them means the server can accept new text the instant you type it, even mid-generation.

Because the actual model call blocks, it runs on a background thread via `asyncio.to_thread`, so it never freezes the connection.

## The honest tradeoff

Each chunk is generated on its own, which means the model has no memory of what came just before it. Pitch and pacing reset at every chunk boundary. A streamed passage sounds a little less natural than one generated in a single pass.

That is the price of not waiting. It is also why the chunker works so hard to cut where a speaker would already have paused, since a reset is much less noticeable at a full stop than in the middle of a phrase.
