"""Tests for the QA remediation (privacy read-paths, auth hardening, robustness)."""

import pytest
from fastapi.testclient import TestClient

from secondbrain.knowledge import chat
from secondbrain.query import service
from secondbrain.query.api import create_app
from secondbrain.search import combined
from secondbrain.security import auth
from secondbrain.speaker import registry
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def _seg(conn, text, speaker_id, day="2026-06-16"):
    af = models.insert_audio_file(
        conn, AudioFile(path=f"/tmp/{text}.flac", started_at=f"{day}T09:00:00.000Z", sample_rate=16000)
    )
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 2.0, text, start_at=f"{day}T09:00:00.000Z",
                                          speaker_id=speaker_id)])
    return conn.execute("SELECT MAX(id) AS m FROM transcript_segments").fetchone()["m"]


def _opted_out_speaker(conn, name="Private"):
    sid = conn.execute(
        "INSERT INTO speakers (name, kind, display_label, opted_out) VALUES (?, 'known', ?, 1)",
        (name, name),
    ).lastrowid
    return int(sid)


# --- P0: opt-out enforced on every read path ---------------------------------


def test_search_excludes_opted_out(conn, settings):
    priv = _opted_out_speaker(conn)
    keep = conn.execute("INSERT INTO speakers (name, kind) VALUES ('Bob','known')").lastrowid
    _seg(conn, "secret pricing plan", priv)
    _seg(conn, "public pricing plan", int(keep))
    hits = combined.search(conn, "pricing plan", settings=settings)
    texts = [h.text for h in hits]
    assert "public pricing plan" in texts
    assert "secret pricing plan" not in texts


def test_day_view_excludes_opted_out(conn, settings):
    priv = _opted_out_speaker(conn)
    _seg(conn, "secret thing", priv)
    segs = service.day_segments(conn, "2026-06-16", settings)
    assert all(s["text"] != "secret thing" for s in segs)


def test_chat_excludes_opted_out_segment(conn, settings):
    priv = _opted_out_speaker(conn)
    _seg(conn, "confidential merger details", priv)
    from secondbrain.llm.client import MockLLM

    result = chat.answer(conn, "merger", llm=MockLLM(responses=["nothing"]), settings=settings)
    assert all(c["segment_id"] for c in result["citations"]) or result["citations"] == []
    assert all("confidential" not in c["text"] for c in result["citations"])


# --- P0: auth hardening ------------------------------------------------------


def test_secure_cookie_set_over_https(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings), base_url="https://testserver")
    r = client.post("/login", json={"username": "owner", "password": "pw"})
    assert r.status_code == 200
    assert "secure" in r.headers.get("set-cookie", "").lower()


def test_login_rate_limited(conn, settings):
    settings.security.require_auth = True
    auth.set_password(conn, "owner", "pw")
    client = TestClient(create_app(settings))
    for _ in range(5):
        assert client.post("/login", json={"username": "owner", "password": "x"}).status_code == 401
    # 6th attempt within the window is throttled
    assert client.post("/login", json={"username": "owner", "password": "x"}).status_code == 429


# --- P1: robustness ----------------------------------------------------------


def test_web_research_requires_https(conn, settings):
    settings.tasks.web_research_enabled = True
    settings.tasks.web_search_url = "http://insecure.example/search"
    from secondbrain.tasks.research import WebResearcher

    with pytest.raises(RuntimeError, match="https"):
        WebResearcher(settings).research("anything")


def test_merge_back_is_safe_no_cycle(conn):
    a = registry.create_unknown_speaker(conn)
    b = registry.create_unknown_speaker(conn)
    registry.merge_speakers(conn, a, b)            # a -> b
    # merging back is a safe no-op (resolution collapses the chain, no cycle)
    assert registry.merge_speakers(conn, b, a) == 0
    assert registry.resolve_speaker_id(conn, a) == b
    assert registry.resolve_speaker_id(conn, b) == b   # terminates (no infinite loop)


def test_daemon_health_reports_staleness(conn, settings):
    from secondbrain import health
    from secondbrain.storage import state

    # no heartbeat yet → ok-but-informational
    checks = {c.name: c for c in health.run_checks(conn, settings)}
    assert "daemon" in checks
    # a very old heartbeat → flagged
    state.set_state(conn, "heartbeat:worker", "2000-01-01T00:00:00.000Z")
    daemon_check = next(c for c in health.run_checks(conn, settings) if c.name == "daemon")
    assert daemon_check.ok is False
