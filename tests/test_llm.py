import pytest

from secondbrain.llm.client import MockLLM, get_llm
from secondbrain.llm.jsonout import LLMJSONError, parse_json


def test_get_llm_mock_default(settings):
    assert get_llm(settings).backend_name == "mock"


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
