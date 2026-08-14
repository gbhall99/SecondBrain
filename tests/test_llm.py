import anyio
import pytest

from secondbrain.llm.client import MockLLM, OllamaLLM, get_llm
from secondbrain.llm.jsonout import LLMJSONError, parse_json


def test_get_llm_mock_default(settings):
    assert get_llm(settings).backend_name == "mock"


class FakeResp:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {"message": {"content": "ok"}}
        self.text = text
        self.request = object()

    def json(self):
        return self._body


class FakeHttpClient:
    """Stands in for OllamaLLM's reused httpx.Client (the `_http` seam)."""

    def __init__(self, responses=None):
        self.captured: dict = {}
        self.calls = 0
        self._responses = list(responses or [])

    def post(self, url, json=None):
        self.calls += 1
        self.captured = dict(json or {})
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return FakeResp()


def _ollama(settings, fake=None) -> OllamaLLM:
    settings.llm.backend = "ollama"
    llm = get_llm(settings)
    assert isinstance(llm, OllamaLLM)
    llm._http = fake or FakeHttpClient()
    return llm


def test_ollama_complete_sets_num_ctx_and_num_predict(settings):
    fake = FakeHttpClient()
    llm = _ollama(settings, fake)

    llm.complete(system="s", prompt="p", max_tokens=64)
    # Small prompt: smallest bucket, and the answer cap rides along.
    assert fake.captured["options"]["num_ctx"] == 8192
    assert fake.captured["options"]["num_predict"] == 64

    llm.complete(system="", prompt="x" * 40000)  # ~13k tokens: needs a bigger window
    assert fake.captured["options"]["num_ctx"] == 16384
    assert "num_predict" not in fake.captured["options"]


def test_ollama_payload_includes_keep_alive(settings):
    fake = FakeHttpClient()
    llm = _ollama(settings, fake)
    llm.complete(system="s", prompt="p")
    assert fake.captured["keep_alive"] == "30m"  # config default


def test_ollama_retries_on_connect_error_then_succeeds(settings, monkeypatch):
    import httpx

    monkeypatch.setattr("secondbrain.llm.client.time.sleep", lambda s: None)
    fake = FakeHttpClient(responses=[httpx.ConnectError("refused"), FakeResp()])
    llm = _ollama(settings, fake)
    assert llm.complete(system="s", prompt="p").text == "ok"
    assert fake.calls == 2


def test_ollama_gives_up_after_two_retries(settings, monkeypatch):
    import httpx

    monkeypatch.setattr("secondbrain.llm.client.time.sleep", lambda s: None)
    fake = FakeHttpClient(responses=[httpx.ConnectError("refused")] * 3)
    llm = _ollama(settings, fake)
    with pytest.raises(httpx.ConnectError):
        llm.complete(system="s", prompt="p")
    assert fake.calls == 3  # initial try + 2 retries, no more


def test_ollama_retries_5xx_but_never_4xx(settings, monkeypatch):
    import httpx

    monkeypatch.setattr("secondbrain.llm.client.time.sleep", lambda s: None)
    # 500 twice then success → retried to completion.
    fake = FakeHttpClient(responses=[FakeResp(status_code=500, text="boom")] * 2)
    llm = _ollama(settings, fake)
    assert llm.complete(system="s", prompt="p").text == "ok"
    assert fake.calls == 3

    # 404 (model not pulled) → immediate failure with the body in the message.
    fake = FakeHttpClient(
        responses=[FakeResp(status_code=404, text='{"error":"model not found, try pulling it"}')]
    )
    llm = _ollama(settings, fake)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        llm.complete(system="s", prompt="p")
    assert fake.calls == 1
    assert "model not found" in str(exc_info.value)


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


def test_parse_json_rejects_non_object_top_level():
    for bad in ("[1, 2, 3]", '"just a string"', "42", "true"):
        with pytest.raises(LLMJSONError):
            parse_json(bad)


def test_parse_json_keeps_fence_lines_inside_strings():
    # Only the leading fence and its matching trailing fence are stripped; a
    # ``` sequence inside string content must survive.
    text = '```json\n{"snippet": "use ```python fences```"}\n```'
    assert parse_json(text) == {"snippet": "use ```python fences```"}


def test_parse_json_repairs_truncated_output():
    # Generation cut off mid-object: close the open structures.
    assert parse_json('{"a": [1, 2') == {"a": [1, 2]}
    # Trailing comma from truncation is dropped.
    assert parse_json('{"a": 1, "b": {"c": 2},') == {"a": 1, "b": {"c": 2}}
    # An unterminated string is closed too.
    assert parse_json('{"a": "unfinished') == {"a": "unfinished"}
    # Mismatched closers are NOT "repaired" into something else.
    with pytest.raises(LLMJSONError):
        parse_json('{"a": [1 }')


def test_complete_json_reprompts_once_then_raises():
    from secondbrain.llm.jsonout import complete_json

    # First reply is garbage, the reprompt succeeds.
    llm = MockLLM(responses=["not json", '{"ok": true}'])
    assert complete_json(llm, system="s", prompt="p") == {"ok": True}

    # Both replies garbage → LLMJSONError propagates.
    llm = MockLLM(responses=["nope", "still nope"], default="really no")
    with pytest.raises(LLMJSONError):
        complete_json(llm, system="s", prompt="p")


def test_complete_json_includes_reprompt_text_on_retry():
    from secondbrain.llm.jsonout import REPROMPT, complete_json

    prompts: list[str] = []

    class SpyLLM(MockLLM):
        def complete(self, *, system, prompt, schema=None, temperature=0.0, max_tokens=None):
            prompts.append(prompt)
            return super().complete(system=system, prompt=prompt, schema=schema,
                                    temperature=temperature, max_tokens=max_tokens)

    llm = SpyLLM(responses=["garbage", '{"a": 1}'])
    assert complete_json(llm, system="s", prompt="question") == {"a": 1}
    assert prompts[0] == "question"
    assert REPROMPT in prompts[1]


def test_ollama_warns_when_context_need_exceeds_largest_bucket(settings, caplog):
    import logging

    fake = FakeHttpClient()
    llm = _ollama(settings, fake)
    with caplog.at_level(logging.WARNING, logger="secondbrain.llm.client"):
        llm.complete(system="", prompt="x" * 120000)  # ~40k tokens > 32768
    assert any("largest num_ctx bucket" in r.message for r in caplog.records)
    assert fake.captured["options"]["num_ctx"] == 32768  # clamped to the max


def test_parse_json_failure_logs_truncated_prefix(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="secondbrain.llm.jsonout"), \
            pytest.raises(LLMJSONError):
        parse_json("definitely not json " * 100)
    msgs = [r.message for r in caplog.records if "parse failed" in r.message]
    assert msgs
    assert len(msgs[0]) < 400  # prefix is truncated, not the whole reply
