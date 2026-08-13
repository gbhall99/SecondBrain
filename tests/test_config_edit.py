"""config_edit.set_hf_token — safe, re-parseable TOML edits used by install.sh."""

from __future__ import annotations

import tomllib

from secondbrain import config_edit


def test_replaces_empty_placeholder():
    out = config_edit.set_hf_token('[diarization]\nenabled = true\nhf_token = ""\n', "hf_abc")
    d = tomllib.loads(out)["diarization"]
    assert d["hf_token"] == "hf_abc" and d["enabled"] is True


def test_replaces_existing_value_once():
    out = config_edit.set_hf_token('[diarization]\nhf_token = "old"\n', "new")
    assert tomllib.loads(out)["diarization"]["hf_token"] == "new"
    assert out.count("hf_token") == 1


def test_inserts_under_existing_section():
    out = config_edit.set_hf_token("[diarization]\nenabled = true\n", "tok")
    d = tomllib.loads(out)["diarization"]
    assert d["hf_token"] == "tok" and d["enabled"] is True


def test_appends_section_when_absent():
    out = config_edit.set_hf_token('[capture]\ninput_device = ""\n', "tok")
    parsed = tomllib.loads(out)
    assert parsed["diarization"]["hf_token"] == "tok"
    assert parsed["capture"]["input_device"] == ""


def test_empty_input_appends():
    assert tomllib.loads(config_edit.set_hf_token("", "tok"))["diarization"]["hf_token"] == "tok"


def test_special_characters_round_trip():
    tok = 'a"b\\c'
    out = config_edit.set_hf_token('hf_token = ""\n', tok)
    assert tomllib.loads(out)["hf_token"] == tok


def test_write_to_existing_file(tmp_path):
    p = tmp_path / "config.local.toml"
    p.write_text('[diarization]\nhf_token = ""\n')
    config_edit.write_hf_token(p, "hf_xyz")
    assert tomllib.loads(p.read_text())["diarization"]["hf_token"] == "hf_xyz"


def test_write_to_missing_file(tmp_path):
    p = tmp_path / "config.local.toml"
    config_edit.write_hf_token(p, "tok")
    assert tomllib.loads(p.read_text())["diarization"]["hf_token"] == "tok"


def test_replacement_is_scoped_to_diarization_section():
    # An hf_token key in ANOTHER section must never be rewritten.
    text = (
        '[other]\nhf_token = "unrelated"\n\n'
        '[diarization]\nenabled = true\nhf_token = "old"\n'
    )
    out = config_edit.set_hf_token(text, "new")
    parsed = tomllib.loads(out)
    assert parsed["other"]["hf_token"] == "unrelated"
    assert parsed["diarization"]["hf_token"] == "new"


def test_insert_scoped_when_token_only_in_other_section():
    text = '[other]\nhf_token = "unrelated"\n\n[diarization]\nenabled = true\n'
    out = config_edit.set_hf_token(text, "tok")
    parsed = tomllib.loads(out)
    assert parsed["other"]["hf_token"] == "unrelated"
    assert parsed["diarization"]["hf_token"] == "tok"
    assert parsed["diarization"]["enabled"] is True


def test_token_in_other_section_appends_diarization():
    text = '[other]\nhf_token = "unrelated"\n'
    out = config_edit.set_hf_token(text, "tok")
    parsed = tomllib.loads(out)
    assert parsed["other"]["hf_token"] == "unrelated"
    assert parsed["diarization"]["hf_token"] == "tok"


def test_write_chmods_file_owner_only(tmp_path):
    p = tmp_path / "config.local.toml"
    p.write_text('[diarization]\nhf_token = ""\n')
    p.chmod(0o644)
    config_edit.write_hf_token(p, "hf_secret")
    assert (p.stat().st_mode & 0o777) == 0o600
