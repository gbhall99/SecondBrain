import pytest
from fastapi.testclient import TestClient

from secondbrain.query.api import create_app
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


@pytest.fixture
def client(conn, settings):
    # conn fixture has created the DB file at settings.db_path; seed a segment.
    af = models.insert_audio_file(
        conn, AudioFile(path="/a.flac", started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    t = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [Segment(t, af, 0.0, 2.0, "decided to adopt the new onboarding flow",
                 start_at="2026-06-16T09:00:00.000Z")],
    )
    return TestClient(create_app(settings))


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["segments_total"] == 1
    assert "disk_free_gb" in body


def test_search_endpoint(client):
    r = client.get("/api/search", params={"q": "onboarding"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert "onboarding" in results[0]["text"]


def test_pause_resume_toggle(client):
    assert client.post("/api/pause").json()["paused"] is True
    assert client.get("/api/status").json()["paused"] is True
    assert client.post("/api/resume").json()["paused"] is False


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SecondBrain" in r.text


def test_speakers_endpoints(client, conn):
    from secondbrain.speaker import registry

    sid = registry.create_unknown_speaker(conn)
    # list + unknown
    assert client.get("/api/speakers").status_code == 200
    unknown = client.get("/api/speakers/unknown").json()["unknown"]
    assert any(s["id"] == sid for s in unknown)
    # name it
    r = client.post(f"/api/speakers/{sid}/name", json={"name": "Dana"})
    assert r.status_code == 200 and r.json()["ok"]
    names = [s["name"] for s in client.get("/api/speakers").json()["speakers"]]
    assert "Dana" in names


def test_speakers_page_renders(client):
    r = client.get("/speakers")
    assert r.status_code == 200
    assert "Who is this?" in r.text
