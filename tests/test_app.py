import base64
import io

import soundfile as sf
from fastapi.testclient import TestClient

from app import MAX_TEXT_LENGTH, SAMPLE_RATE, chunk_text, create_app


class FakeTTSService:
    def __init__(self, *, loaded: bool = True, error: Exception | None = None) -> None:
        self._loaded = loaded
        self.error = error
        self.synthesized_texts: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, text: str) -> bytes:
        self.synthesized_texts.append(text)
        if self.error:
            raise self.error
        self._loaded = True
        output = io.BytesIO()
        sf.write(output, [0.0, 0.1, -0.1, 0.0], SAMPLE_RATE, format="WAV")
        return output.getvalue()


def test_root_serves_audio_studio() -> None:
    with TestClient(create_app(FakeTTSService())) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Voice Foundry" in response.text
    assert 'maxlength="500"' in response.text


def test_chunk_text_prefers_sentence_boundaries() -> None:
    assert chunk_text("Hello there. How are you?") == [
        "Hello there.",
        "How are you?",
    ]


def test_chunk_text_falls_back_to_clauses_then_balanced_words() -> None:
    sentence = "one two three four five six seven eight nine, ten eleven twelve."
    assert chunk_text(sentence, max_words=4) == [
        "one two three",
        "four five six",
        "seven eight nine,",
        "ten eleven twelve.",
    ]


def test_chunk_text_ignores_empty_input() -> None:
    assert chunk_text("   ") == []


def test_empty_text_returns_structured_error() -> None:
    with TestClient(create_app(FakeTTSService())) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"request_id": "empty-1", "text": "   "})
            response = websocket.receive_json()

    assert response == {
        "type": "error",
        "request_id": "empty-1",
        "message": "Enter some text before generating speech.",
    }


def test_oversized_text_returns_structured_error() -> None:
    with TestClient(create_app(FakeTTSService())) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"request_id": "long-1", "text": "x" * (MAX_TEXT_LENGTH + 1)}
            )
            response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["request_id"] == "long-1"
    assert str(MAX_TEXT_LENGTH) in response["message"]


def test_success_streams_chunks_then_done() -> None:
    service = FakeTTSService(loaded=False)
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"request_id": "speech-42", "text": "  Hello local voice. Speak up.  "}
            )
            loading = websocket.receive_json()
            messages = [websocket.receive_json() for _ in range(5)]

    assert loading == {
        "type": "status",
        "request_id": "speech-42",
        "status": "loading",
        "message": "Loading SpeechT5 for the first time...",
    }

    statuses = [m for m in messages if m["type"] == "status"]
    chunks = [m for m in messages if m["type"] == "audio_chunk"]
    done = messages[-1]

    assert [m["status"] for m in statuses] == ["generating", "generating"]
    assert [m["index"] for m in chunks] == [0, 1]
    assert all(m["request_id"] == "speech-42" for m in messages)
    assert all(m["mime_type"] == "audio/wav" for m in chunks)
    assert all(base64.b64decode(m["audio"]).startswith(b"RIFF") for m in chunks)
    assert done == {"type": "done", "request_id": "speech-42", "total": 2}
    assert service.synthesized_texts == ["Hello local voice.", "Speak up."]


def test_each_chunk_carries_its_own_text() -> None:
    service = FakeTTSService()
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"request_id": "chunked-1", "text": "First one. Second one."})
            received = [websocket.receive_json() for _ in range(5)]

    chunk_texts = [m["text"] for m in received if m["type"] == "audio_chunk"]
    assert chunk_texts == ["First one.", "Second one."]


def test_append_mode_queues_behind_the_current_request() -> None:
    service = FakeTTSService()
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"request_id": "first", "text": "One two.", "mode": "replace"})
            websocket.send_json({"request_id": "second", "text": "Three four.", "mode": "append"})
            received = []
            while len({m["request_id"] for m in received if m["type"] == "done"}) < 2:
                received.append(websocket.receive_json())

    chunks = [m for m in received if m["type"] == "audio_chunk"]
    assert [m["request_id"] for m in chunks] == ["first", "second"]
    assert service.synthesized_texts == ["One two.", "Three four."]


def test_replace_mode_retires_queued_work() -> None:
    service = FakeTTSService()
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"request_id": "old", "text": "Stale text.", "mode": "replace"})
            websocket.send_json({"request_id": "new", "text": "Fresh text.", "mode": "replace"})
            received = []
            while not any(
                m["type"] == "done" and m["request_id"] == "new" for m in received
            ):
                received.append(websocket.receive_json())

    assert not any(
        m["type"] == "done" and m["request_id"] == "old" for m in received
    )
    assert "Fresh text." in service.synthesized_texts


def test_invalid_mode_is_rejected() -> None:
    with TestClient(create_app(FakeTTSService())) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"request_id": "bad-mode", "text": "Hi", "mode": "sideways"})
            response = websocket.receive_json()

    assert response["type"] == "error"
    assert "append" in response["message"]


def test_synthesis_failure_returns_error_with_request_id() -> None:
    service = FakeTTSService(error=RuntimeError("model exploded"))
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"request_id": "failure-7", "text": "Hello"})
            status = websocket.receive_json()
            error = websocket.receive_json()

    assert status["status"] == "generating"
    assert error == {
        "type": "error",
        "request_id": "failure-7",
        "message": "Speech generation failed. Check the server log and try again.",
    }
