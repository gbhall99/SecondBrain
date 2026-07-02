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


def test_stats_endpoint(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["segments"] == 1
    assert "kg_nodes" in body
    assert "goals" in body


def test_person_endpoints(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (5, 'Dana', 'known', 0)")
    r = client.get("/api/person/5")
    assert r.status_code == 200
    assert r.json()["label"] == "Dana"
    assert client.get("/person/5").status_code == 200  # HTML page renders
    assert client.get("/api/person/999").status_code == 404


def test_relationships_endpoints(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (6, 'Dana', 'known', 0)")
    r = client.get("/api/relationships")
    assert r.status_code == 200
    assert "relationships" in r.json()
    assert client.get("/relationships").status_code == 200  # page renders


def test_project_endpoints(client, conn):
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (10, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, valid) "
        "VALUES (20, 10, 'fact', 'on track', 1)"
    )
    r = client.get("/api/projects")
    assert r.status_code == 200
    assert any(p["label"] == "Atlas" for p in r.json()["projects"])
    assert client.get("/projects").status_code == 200  # list page renders
    d = client.get("/api/project/10")
    assert d.status_code == 200 and d.json()["label"] == "Atlas"
    assert client.get("/project/10").status_code == 200  # dossier page renders
    assert client.get("/api/project/999").status_code == 404


def test_opted_out_speaker_audio_blocked(client, conn):
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (8, 'X', 'known', 0, 1)"
    )
    assert client.get("/api/speakers/8/samples").status_code == 403
    assert client.get("/api/speakers/8/clip/1").status_code == 403


def test_timeline_endpoints(client):
    r = client.get("/api/timeline/2026-06-16")
    assert r.status_code == 200
    assert "conversations" in r.json()
    assert client.get("/timeline/2026-06-16").status_code == 200  # page renders
    assert client.get("/timeline").status_code == 200  # today


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
    # unified dashboard: shared nav links present
    for href in ('href="/timeline"', 'href="/relationships"', 'href="/projects"', 'href="/speakers"'):
        assert href in r.text


def test_shared_nav_on_new_pages(client, conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (7, 'Dana', 'known', 0)")
    for path in ("/person/7", "/relationships", "/timeline"):
        r = client.get(path)
        assert r.status_code == 200
        assert 'class="nav"' in r.text  # extends base.html


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


def test_ask_endpoint(client):
    r = client.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "citations" in body


def test_graph_endpoints(client, conn):
    from secondbrain.knowledge import graph

    nid = graph.create_node(conn, type="project", name="Atlas", embedding=None,
                            confidence=0.9, extraction_id=None)
    found = client.get("/api/graph/search", params={"q": "atlas"}).json()["nodes"]
    assert any(n["id"] == nid for n in found)
    detail = client.get(f"/api/graph/node/{nid}").json()
    assert detail["node"]["name"] == "Atlas"


def test_chat_and_graph_pages_render(client):
    assert "Ask your second brain" in client.get("/chat").text
    assert "Knowledge graph" in client.get("/graph").text


def test_tasks_api_and_plan(client):
    tid = client.post("/api/tasks", json={"title": "Write spec", "value": 5,
                                          "estimate_minutes": 30}).json()["id"]
    assert any(t["id"] == tid for t in client.get("/api/tasks").json()["tasks"])
    plan = client.post("/api/plan/today", json={"action": "propose"}).json()
    assert tid in plan["task_ids"]
    assert client.post("/api/plan/today", json={"action": "accept"}).json()["status"] == "accepted"
    assert client.post(f"/api/tasks/{tid}/status", json={"status": "done"}).json()["ok"]


def test_tasks_page_renders(client):
    assert "Tasks" in client.get("/tasks").text


def test_speaker_quality_and_reattribute(client):
    assert client.get("/api/speakers/quality").status_code == 200
    assert client.post("/api/speakers/reattribute").json()["relabeled"] == 0
    assert "Transcript" in client.get("/day").text


def test_goals_api_crud(client):
    gid = client.post("/api/goals", json={"title": "Win Q3", "priority": 1}).json()["id"]
    titles = [g["title"] for g in client.get("/api/goals").json()["goals"]]
    assert "Win Q3" in titles
    assert client.post(f"/api/goals/{gid}/status", json={"status": "done"}).json()["ok"]


def test_brief_and_goals_pages_render(client):
    assert "Morning brief" in client.get("/brief").text
    assert "Goals" in client.get("/goals").text


def test_digest_generate_and_suggestions(client):
    r = client.post("/api/digest/generate", json={"kind": "daily", "force": True})
    assert r.status_code == 200
    assert client.get("/api/suggestions").status_code == 200
