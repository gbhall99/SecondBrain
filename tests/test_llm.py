import anyio
import pytest

from secondbrain.llm.client import MockLLM, OllamaLLM, get_llm
from secondbrain.llm.jsonout import LLMJSONError, parse_json


def test_get_llm_mock_default(settings):
    assert get_llm(settings).backend_name == "mock"


def test_ollama_complete_sets_num_ctx_and_num_predict(settings, monkeypatch):
    import httpx

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ok"}}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    settings.llm.backend = "ollama"
    llm = get_llm(settings)
    assert isinstance(llm, OllamaLLM)

    llm.complete(system="s", prompt="p", max_tokens=64)
    # Small prompt: smallest bucket, and the answer cap rides along.
    assert captured["options"]["num_ctx"] == 8192
    assert captured["options"]["num_predict"] == 64

    llm.complete(system="", prompt="x" * 40000)  # ~13k tokens: needs a bigger window
    assert captured["options"]["num_ctx"] == 16384
    assert "num_predict" not in captured["options"]


def test_llm_astream_default_yields_single_chunk():
    async def run():
        chunks = []
        async for piece in MockLLM(responses=["hello there"]).astream(system="", prompt="q"):
            chunks.append(piece)
        return chunks

    assert anyio.run(run) == ["hello there"]


def test_get_llm_unknown_raises(settings):
    settings.llm.backend = "nope"
    with pytest.raises(ValueError):
        get_llm(settings)


def test_mock_scripted_then_substring_then_default():
    llm = MockLLM(responses=['{"a":1}'], by_substring={"hello": "hi"}, default="def")
    assert llm.complete(system="", prompt="anything").text == '{"a":1}'  # scripted first
    assert llm.complete(system="", prompt="say hello").text == "hi"       # substring
    assert llm.complete(system="", prompt="other").text == "def"          # default


def test_mock_schema_default_is_empty_object():
    llm = MockLLM()
    assert llm.complete(system="", prompt="x", schema={"type": "object"}).text == "{}"


def test_parse_json_handles_fences_and_prose():
    assert parse_json('```json\n{"x": 1}\n```') == {"x": 1}
    assert parse_json('Sure!\n{"y": 2}\nDone') == {"y": 2}
    with pytest.raises(LLMJSONError):
        parse_json("no json here")
