"""Local FastAPI application for SpeechT5 text-to-speech."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import re
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

MAX_TEXT_LENGTH = 500
SAMPLE_RATE = 16_000
CLIENT_PATH = Path(__file__).with_name("client.html")

MAX_CHUNK_WORDS = 8
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]*")
CLAUSE_PATTERN = re.compile(r"[^,;:]+[,;:]*")

XVECTOR_REPO = "Matthijs/cmu-arctic-xvectors"
XVECTOR_ARCHIVE = "spkrec-xvect.zip"
XVECTOR_INDEX = 7306

logger = logging.getLogger(__name__)


class TTSService(Protocol):
    """Interface used by the WebSocket handler and model-free tests."""

    @property
    def is_loaded(self) -> bool: ...

    def synthesize(self, text: str) -> bytes: ...


def _load_speaker_xvector(index: int) -> Any:
    """Return one CMU Arctic x-vector by archive position.

    Sorted-name order matches the row order the retired dataset script
    produced, so index 7306 still selects the same SLT speaker.
    """

    import numpy as np
    from huggingface_hub import hf_hub_download

    archive_path = hf_hub_download(
        XVECTOR_REPO, XVECTOR_ARCHIVE, repo_type="dataset"
    )
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".npy"))
        if index >= len(names):
            raise RuntimeError(
                f"Speaker index {index} is outside the {len(names)} available x-vectors"
            )
        return np.load(io.BytesIO(archive.read(names[index])))


class SpeechT5Service:
    """Lazily loads SpeechT5 resources and returns complete WAV files."""

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None
        self._vocoder: Any | None = None
        self._speaker_embeddings: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.is_loaded:
            return

        import torch
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        self._processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts"
        )
        self._model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        )
        self._vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        )
        self._speaker_embeddings = torch.tensor(
            _load_speaker_xvector(XVECTOR_INDEX)
        ).unsqueeze(0)

    def synthesize(self, text: str) -> bytes:
        self._load()

        import soundfile as sf
        import torch

        if (
            self._processor is None
            or self._model is None
            or self._vocoder is None
            or self._speaker_embeddings is None
        ):
            raise RuntimeError("SpeechT5 resources did not load correctly")

        inputs = self._processor(text=text, return_tensors="pt")
        with torch.inference_mode():
            waveform = self._model.generate_speech(
                inputs["input_ids"],
                self._speaker_embeddings,
                vocoder=self._vocoder,
            )

        buffer = io.BytesIO()
        sf.write(buffer, waveform.detach().cpu().numpy(), SAMPLE_RATE, format="WAV")
        return buffer.getvalue()


def _split_by_words(text: str, max_words: int) -> list[str]:
    """Split into evenly sized pieces so no chunk is left a stranded tail.

    A greedy split leaves remainders like "signal." alone in their own chunk,
    which the model then voices with full sentence intonation.
    """

    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    parts = math.ceil(len(words) / max_words)
    size = math.ceil(len(words) / parts)
    return [
        " ".join(words[start : start + size])
        for start in range(0, len(words), size)
    ]


def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split text into speakable pieces, preferring sentence then clause breaks.

    Synthesising a whole passage delays the first sound until the slowest word
    is done, so the text is cut at punctuation the voice would pause on anyway.
    """

    chunks: list[str] = []
    for raw_sentence in SENTENCE_PATTERN.findall(text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if len(sentence.split()) <= max_words:
            chunks.append(sentence)
            continue
        for raw_clause in CLAUSE_PATTERN.findall(sentence):
            clause = raw_clause.strip()
            if not clause:
                continue
            if len(clause.split()) <= max_words:
                chunks.append(clause)
            else:
                chunks.extend(_split_by_words(clause, max_words))
    return chunks


def _parse_request(message: Any) -> tuple[str, str, str]:
    if not isinstance(message, dict):
        raise ValueError("Request must be a JSON object.")

    request_id = message.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string.")

    raw_text = message.get("text")
    if not isinstance(raw_text, str):
        raise ValueError("text must be a string.")

    text = raw_text.strip()
    if not text:
        raise ValueError("Enter some text before generating speech.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text must be {MAX_TEXT_LENGTH} characters or fewer.")

    mode = message.get("mode", "replace")
    if mode not in ("append", "replace"):
        raise ValueError("mode must be 'append' or 'replace'.")

    return request_id, text, mode


def _request_id_of(message: Any) -> str | None:
    if isinstance(message, dict):
        value = message.get("request_id")
        if isinstance(value, str) and value:
            return value
    return None


def create_app(tts_service: TTSService | None = None) -> FastAPI:
    """Build the application, optionally using an injected TTS service."""

    app = FastAPI(title="RealtimeTTS", docs_url=None, redoc_url=None)
    app.state.tts_service = tts_service or SpeechT5Service()
    app.state.inference_lock = asyncio.Lock()

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(CLIENT_PATH, media_type="text/html")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()

        # Appended text queues up behind whatever is already generating.
        # A "replace" request bumps the generation, which retires every queued
        # and in-flight item belonging to the previous one.
        queued: deque[tuple[str, str, int]] = deque()
        arrived = asyncio.Event()
        generation = 0
        # reader and worker both send, and Starlette does not serialize sends.
        send_lock = asyncio.Lock()

        async def send(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def stream(request_id: str, text: str, era: int) -> None:
            service: TTSService = app.state.tts_service
            try:
                if not service.is_loaded:
                    await send(
                        {
                            "type": "status",
                            "request_id": request_id,
                            "status": "loading",
                            "message": "Loading SpeechT5 for the first time...",
                        }
                    )

                chunks = chunk_text(text)
                for index, chunk in enumerate(chunks):
                    if era != generation:
                        return

                    await send(
                        {
                            "type": "status",
                            "request_id": request_id,
                            "status": "generating",
                            "message": f"Generating chunk {index + 1} of {len(chunks)}...",
                            "index": index,
                            "total": len(chunks),
                        }
                    )

                    async with app.state.inference_lock:
                        wav_bytes = await asyncio.to_thread(service.synthesize, chunk)

                    if era != generation:
                        return

                    await send(
                        {
                            "type": "audio_chunk",
                            "request_id": request_id,
                            "index": index,
                            "total": len(chunks),
                            "text": chunk,
                            "mime_type": "audio/wav",
                            "audio": base64.b64encode(wav_bytes).decode("ascii"),
                        }
                    )

                if era != generation:
                    return

                await send(
                    {
                        "type": "done",
                        "request_id": request_id,
                        "total": len(chunks),
                    }
                )
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("Speech generation failed")
                try:
                    await send(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "message": "Speech generation failed. Check the server log and try again.",
                        }
                    )
                except Exception:
                    logger.exception("Could not report the failure to the client")

        async def reader() -> None:
            nonlocal generation
            while True:
                message = await websocket.receive_json()
                try:
                    request_id, text, mode = _parse_request(message)
                except (ValueError, TypeError) as exc:
                    await send(
                        {
                            "type": "error",
                            "request_id": _request_id_of(message),
                            "message": str(exc),
                        }
                    )
                    continue

                if mode == "replace":
                    generation += 1
                    queued.clear()
                queued.append((request_id, text, generation))
                arrived.set()

        async def worker() -> None:
            while True:
                await arrived.wait()
                arrived.clear()
                while queued:
                    request_id, text, era = queued.popleft()
                    if era != generation:
                        continue
                    try:
                        await stream(request_id, text, era)
                    except WebSocketDisconnect:
                        raise
                    except Exception:
                        # One failed generation must not leave the reader
                        # accepting requests that nothing will ever consume.
                        logger.exception(
                            "Dropping request %s after stream failure", request_id
                        )

        reader_task = asyncio.create_task(reader())
        worker_task = asyncio.create_task(worker())
        tasks = (reader_task, worker_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    continue
                error = task.exception()
                if error is not None and not isinstance(error, WebSocketDisconnect):
                    logger.error("WebSocket session ended unexpectedly", exc_info=error)
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except (asyncio.CancelledError, WebSocketDisconnect):
                    pass
                except Exception:
                    logger.exception("WebSocket task failed during shutdown")

    return app


app = create_app()
