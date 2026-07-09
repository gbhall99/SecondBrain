from pathlib import Path

from secondbrain.pipeline import conversation
from secondbrain.pipeline.diarize import MockDiarizer, deterministic_embedding
from secondbrain.query import service
from secondbrain.speaker import attribution, cluster, enroll, registry
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile, Segment


def test_speaker_samples_blocked_for_opted_out(conn, settings):
    """Opted-out speakers' raw audio must never be served (privacy)."""
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (5, 'X', 'known', 0, 1)"
    )
    assert service.is_opted_out(conn, 5, settings) is True
    assert service.speaker_samples(conn, 5, settings=settings) == []


def test_opted_out_ids_fast_path_no_config_names(conn, settings):
    settings.consent.speaker_opt_out = []
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (1,'Me','owner',1,1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (2,'P','known',0,1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (3,'Q','known',0,0)")
    # owner excluded even if flagged; non-opted excluded.
    assert registry.opted_out_speaker_ids(conn, settings) == {2}


def test_opted_out_ids_by_config_name(conn, settings):
    settings.consent.speaker_opt_out = ["Q"]
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner, opted_out) VALUES (3,'Q','known',0,0)")
    assert registry.opted_out_speaker_ids(conn, settings) == {3}


# --- matching + centroids ----------------------------------------------------


def test_match_owner_above_threshold(conn, settings):
    emb = deterministic_embedding("me", 32)
    owner = registry.get_or_create_owner(conn, "Me")
    registry.update_centroid(conn, owner, emb)
    m = registry.match_embedding(conn, emb, settings)
    assert m.speaker_id == owner and m.is_owner and m.similarity > 0.99


def test_no_match_creates_distance(conn, settings):
    registry.update_centroid(conn, registry.get_or_create_owner(conn), deterministic_embedding("a", 32))
    m = registry.match_embedding(conn, [(-1) ** i for i in range(32)], settings)
    # orthogonal-ish vector should fall below the match threshold
    assert m.speaker_id is None


def test_centroid_running_mean(conn):
    sid = registry.create_unknown_speaker(conn)
    registry.update_centroid(conn, sid, [1.0, 0.0, 0.0, 0.0])
    registry.update_centroid(conn, sid, [0.0, 1.0, 0.0, 0.0])
    row = conn.execute("SELECT centroid, exemplar_count FROM speakers WHERE id=?", (sid,)).fetchone()
    assert row["exemplar_count"] == 2
    vec = registry.deserialize_embedding(row["centroid"])
    assert vec[0] > 0 and vec[1] > 0          # moved toward both exemplars


# --- enrollment --------------------------------------------------------------


def test_enroll_owner_from_files(conn, settings):
    diar = MockDiarizer(dim=settings.diarization.embedding_dim)
    owner = enroll.enroll_owner_from_files(
        conn, [Path("/tmp/a.flac"), Path("/tmp/b.flac")], diarizer=diar, settings=settings
    )
    row = conn.execute("SELECT is_owner, centroid FROM speakers WHERE id=?", (owner,)).fetchone()
    assert row["is_owner"] == 1 and row["centroid"] is not None
    # owner's enrolled voice should now match itself
    m = registry.match_embedding(conn, registry.deserialize_embedding(row["centroid"]), settings)
    assert m.is_owner


# --- conversation segmentation ----------------------------------------------


def _chunk(conn, started, ended, dur=2.0, status="transcribed"):
    return models.insert_audio_file(
        conn,
        AudioFile(path=f"/tmp/{started}.flac", started_at=started, sample_rate=16000,
                  ended_at=ended, duration_s=dur, status=status),
    )


def test_assign_chunk_groups_then_splits_on_gap(conn, settings):
    settings.conversation.max_gap_minutes = 5
    a = _chunk(conn, "2026-06-16T09:00:00.000Z", "2026-06-16T09:01:00.000Z")
    c1 = conversation.assign_chunk(conn, a, settings)
    # 1 minute later → same conversation
    b = _chunk(conn, "2026-06-16T09:02:00.000Z", "2026-06-16T09:03:00.000Z")
    c2 = conversation.assign_chunk(conn, b, settings)
    assert c1 == c2
    # 30 minutes later → new conversation, previous closed + diarize job queued
    d = _chunk(conn, "2026-06-16T09:33:00.000Z", "2026-06-16T09:34:00.000Z")
    c3 = conversation.assign_chunk(conn, d, settings)
    assert c3 != c1
    assert conn.execute(
        "SELECT status FROM conversations WHERE id=?", (c1,)
    ).fetchone()["status"] == "closed"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE type='diarize_conversation'"
    ).fetchone()["n"] == 1


# --- attribution end-to-end (fake audio builder + mock diarizer) -------------


def _fake_builder(conn, chunks, settings):
    return Path("/tmp/concat.wav"), attribution.concat_offsets_from_db(conn, chunks)


def _seeded_conversation(conn):
    """Two 2s chunks in one conversation, each with one transcript segment."""
    conv = conn.execute(
        "INSERT INTO conversations (started_at, status) VALUES ('2026-06-16T09:00:00.000Z','closed')"
    ).lastrowid
    for i in range(2):
        afid = _chunk(conn, f"2026-06-16T09:0{i}:00.000Z", f"2026-06-16T09:0{i}:02.000Z")
        conn.execute("UPDATE audio_files SET conversation_id=? WHERE id=?", (conv, afid))
        tid = models.insert_transcript(conn, afid, "mock", "mock", "en")
        models.insert_segments(
            conn,
            [Segment(tid, afid, 0.0, 2.0, f"sentence from chunk {i}",
                     start_at=f"2026-06-16T09:0{i}:00.000Z")],
        )
    return conv


def test_attribute_conversation_labels_segments(conn, settings):
    conv = _seeded_conversation(conn)
    diar = MockDiarizer(
        turns=[(0.0, 2.0, "S0"), (2.0, 4.0, "S1")],
        embeddings={"S0": deterministic_embedding("p0", settings.diarization.embedding_dim),
                    "S1": deterministic_embedding("p1", settings.diarization.embedding_dim)},
    )
    n = attribution.attribute_conversation(
        conn, conv, diarizer=diar, settings=settings, audio_builder=_fake_builder
    )
    assert n == 2
    rows = conn.execute(
        "SELECT speaker_id FROM transcript_segments ORDER BY audio_file_id"
    ).fetchall()
    speaker_ids = [r["speaker_id"] for r in rows]
    assert all(s is not None for s in speaker_ids)
    assert speaker_ids[0] != speaker_ids[1]        # two distinct global speakers
    # conversation diarized + retention deadline now set on its chunks
    assert conn.execute("SELECT status FROM conversations WHERE id=?", (conv,)).fetchone()["status"] == "diarized"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM audio_files WHERE conversation_id=? AND retention_delete_after IS NOT NULL",
        (conv,),
    ).fetchone()["n"] == 2


# --- clustering + retroactive relabel ---------------------------------------


def test_clustering_merges_similar_unknowns(conn, settings):
    base = deterministic_embedding("person-x", 32)
    near = registry.normalize([v + 0.001 for v in base])   # nearly identical
    s1 = registry.create_unknown_speaker(conn)
    s2 = registry.create_unknown_speaker(conn)
    registry.update_centroid(conn, s1, base)
    registry.update_centroid(conn, s2, near)
    # give each a segment so we can verify retroactive relabel
    af = _chunk(conn, "2026-06-16T10:00:00.000Z", "2026-06-16T10:00:02.000Z")
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 1.0, "x1"), Segment(tid, af, 1.0, 2.0, "x2")])
    segs = conn.execute("SELECT id FROM transcript_segments ORDER BY id").fetchall()
    registry.assign_segment_speaker(conn, segs[0]["id"], s1, 0.9)
    registry.assign_segment_speaker(conn, segs[1]["id"], s2, 0.9)

    merges = cluster.run_clustering(conn, settings)
    assert merges == 1
    canonical = min(s1, s2)
    rows = conn.execute("SELECT DISTINCT speaker_id FROM transcript_segments").fetchall()
    assert {r["speaker_id"] for r in rows} == {canonical}


def test_clustering_keeps_distinct_voices_apart(conn, settings):
    s1 = registry.create_unknown_speaker(conn)
    s2 = registry.create_unknown_speaker(conn)
    registry.update_centroid(conn, s1, deterministic_embedding("alpha", 32))
    registry.update_centroid(conn, s2, deterministic_embedding("omega", 32))
    assert cluster.run_clustering(conn, settings) == 0


# --- naming, merge, opt-out --------------------------------------------------


def test_name_speaker_keeps_history(conn, settings):
    sid = registry.create_unknown_speaker(conn)
    af = _chunk(conn, "2026-06-16T11:00:00.000Z", "2026-06-16T11:00:02.000Z")
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 1.0, "hi there")])
    seg = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    registry.assign_segment_speaker(conn, seg, sid, 0.9)
    registry.name_speaker(conn, sid, "Alice", settings)
    row = conn.execute("SELECT name, kind FROM speakers WHERE id=?", (sid,)).fetchone()
    assert row["name"] == "Alice" and row["kind"] == "known"
    # segment still points at the same (now named) id
    assert conn.execute("SELECT speaker_id FROM transcript_segments WHERE id=?", (seg,)).fetchone()["speaker_id"] == sid


def test_naming_into_optout_redacts_history(conn, settings):
    settings.consent.speaker_opt_out = ["Confidential Person"]
    sid = registry.create_unknown_speaker(conn)
    af = _chunk(conn, "2026-06-16T12:00:00.000Z", "2026-06-16T12:00:02.000Z")
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 1.0, "secret plans here")])
    seg = conn.execute("SELECT id FROM transcript_segments").fetchone()["id"]
    registry.assign_segment_speaker(conn, seg, sid, 0.9)
    redacted = registry.name_speaker(conn, sid, "Confidential Person", settings)
    assert redacted == 1
    text = conn.execute("SELECT text FROM transcript_segments WHERE id=?", (seg,)).fetchone()["text"]
    assert text == registry.REDACTED_TEXT
    # and it's gone from full-text search
    hits = conn.execute(
        "SELECT rowid FROM transcript_segments_fts WHERE transcript_segments_fts MATCH 'secret'"
    ).fetchall()
    assert hits == []


def test_merge_relabels_and_recounts(conn, settings):
    src = registry.create_unknown_speaker(conn)
    dst = registry.create_unknown_speaker(conn)
    af = _chunk(conn, "2026-06-16T13:00:00.000Z", "2026-06-16T13:00:02.000Z")
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [Segment(tid, af, 0.0, 1.0, "a"), Segment(tid, af, 1.0, 2.0, "b")])
    segs = conn.execute("SELECT id FROM transcript_segments ORDER BY id").fetchall()
    registry.assign_segment_speaker(conn, segs[0]["id"], src, 0.9)
    registry.assign_segment_speaker(conn, segs[1]["id"], dst, 0.9)
    n = registry.merge_speakers(conn, src, dst, settings)
    assert n == 1
    assert registry.resolve_speaker_id(conn, src) == dst
    assert conn.execute("SELECT COUNT(*) AS n FROM transcript_segments WHERE speaker_id=?", (dst,)).fetchone()["n"] == 2


def test_set_owner_from_history(conn):
    from secondbrain.query import service

    sid = registry.create_unknown_speaker(conn)
    service.set_owner(conn, sid)
    row = conn.execute("SELECT is_owner, kind FROM speakers WHERE id=?", (sid,)).fetchone()
    assert row["is_owner"] == 1 and row["kind"] == "owner"
    # moving ownership demotes the previous owner to a regular known voice
    # (no stale kind='owner' rows left behind)
    other = registry.create_unknown_speaker(conn)
    service.set_owner(conn, other)
    old = conn.execute("SELECT is_owner, kind FROM speakers WHERE id=?", (sid,)).fetchone()
    new = conn.execute("SELECT is_owner, kind FROM speakers WHERE id=?", (other,)).fetchone()
    assert (new["is_owner"], new["kind"]) == (1, "owner")
    assert (old["is_owner"], old["kind"]) == (0, "known")
