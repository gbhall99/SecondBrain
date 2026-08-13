from secondbrain.pipeline import worker
from secondbrain.pipeline.transcribe import MockTranscriber, TranscribedSegment
from secondbrain.pipeline.vad import MockVad
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile


def _add_audio(conn, tmp_path, started_at="2026-06-16T09:00:00.000Z", name="chunk.flac"):
    # A real file on disk: the worker skips (status 'missing') vanished files.
    audio = tmp_path / name
    audio.write_bytes(b"\x00\x01")
    return models.insert_audio_file(
        conn, AudioFile(path=str(audio), started_at=started_at, sample_rate=16000)
    )


def test_transcription_end_to_end(conn, settings, tmp_path):
    af_id = _add_audio(conn, tmp_path)
    worker.enqueue_transcription(conn, af_id)

    transcriber = MockTranscriber(
        [TranscribedSegment(0.0, 2.0, "we agreed on the pricing", 0.9),
         TranscribedSegment(2.0, 4.0, "ship it on friday", 0.8)]
    )
    n = worker.drain(conn, transcriber=transcriber, vad=MockVad(), settings=settings)
    assert n == 1

    segs = conn.execute("SELECT * FROM transcript_segments ORDER BY id").fetchall()
    assert len(segs) == 2
    # absolute timestamp = file start + offset
    assert segs[0]["start_at"] == "2026-06-16T09:00:00.000Z"
    assert segs[1]["start_at"] == "2026-06-16T09:00:02.000Z"

    af = models.get_audio_file(conn, af_id)
    assert af["status"] == "transcribed"
    assert af["has_speech"] == 1
    assert af["retention_delete_after"] is not None


def test_silence_is_not_transcribed(conn, settings, tmp_path):
    # Enable VAD and give it a real (empty) file so the silence branch runs.
    settings.vad.enabled = True
    audio = tmp_path / "silent.flac"
    audio.write_bytes(b"\x00")
    af_id = models.insert_audio_file(
        conn, AudioFile(path=str(audio), started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    worker.process_audio_file(
        conn, af_id, transcriber=MockTranscriber(), vad=MockVad(has_speech=False), settings=settings
    )
    assert conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"] == 0
    assert models.get_audio_file(conn, af_id)["status"] == "transcribed"


def test_diarization_enabled_defers_retention_and_groups_conversation(conn, settings, tmp_path):
    settings.diarization.enabled = True
    af_id = _add_audio(conn, tmp_path)
    worker.enqueue_transcription(conn, af_id)
    worker.drain(
        conn,
        transcriber=MockTranscriber([TranscribedSegment(0.0, 1.0, "hello", 0.9)]),
        vad=MockVad(),
        settings=settings,
        max_jobs=1,  # only the transcribe job; not the queued diarize job
    )
    row = models.get_audio_file(conn, af_id)
    assert row["status"] == "transcribed"
    assert row["retention_delete_after"] is None      # deferred until diarized
    assert row["conversation_id"] is not None          # grouped into a conversation


def test_diarization_disabled_still_groups_conversation(conn, settings, tmp_path):
    # Segmentation is not a diarization-only concern: extraction consumes
    # conversations, so chunks are grouped even with diarization off.
    settings.diarization.enabled = False
    af_id = _add_audio(conn, tmp_path)
    worker.enqueue_transcription(conn, af_id)
    worker.drain(
        conn,
        transcriber=MockTranscriber([TranscribedSegment(0.0, 1.0, "hello", 0.9)]),
        vad=MockVad(),
        settings=settings,
    )
    row = models.get_audio_file(conn, af_id)
    assert row["conversation_id"] is not None          # grouped into a conversation
    assert row["retention_delete_after"] is not None   # retention NOT deferred


def test_extraction_disabled_enqueues_no_job(conn, settings, tmp_path):
    # diarization on, extraction off → diarize runs, but no extract job is queued
    from secondbrain.pipeline.diarize import MockDiarizer

    settings.diarization.enabled = True
    settings.extraction.enabled = False
    af_id = _add_audio(conn, tmp_path)
    worker.enqueue_transcription(conn, af_id)
    worker.drain(
        conn,
        transcriber=MockTranscriber([TranscribedSegment(0.0, 1.0, "hello", 0.9)]),
        vad=MockVad(),
        diarizer=MockDiarizer(dim=settings.diarization.embedding_dim),
        settings=settings,
    )
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE type='extract_knowledge'"
    ).fetchone()["n"] == 0


def test_worker_dispatches_proactive_job(conn, settings):
    from secondbrain.llm.client import MockLLM
    from secondbrain.pipeline import queue as q
    from secondbrain.proactive.engine import JOB_PROACTIVE

    settings.proactive.enabled = True
    q.enqueue(conn, JOB_PROACTIVE, {"kind": "daily"})
    worker.drain(conn, llm=MockLLM(responses=["brief"]), settings=settings)
    assert conn.execute("SELECT COUNT(*) AS n FROM digests").fetchone()["n"] == 1


def test_worker_dispatches_extract_job(conn, settings):
    # The worker should claim and run an enqueued extract_knowledge job.
    from secondbrain.knowledge.extract import enqueue_extraction
    from secondbrain.llm.client import MockLLM
    from secondbrain.storage.models import AudioFile

    settings.extraction.enabled = True
    conv = conn.execute(
        "INSERT INTO conversations (started_at, status, knowledge_status) "
        "VALUES ('2026-06-16T09:00:00.000Z','diarized','pending')"
    ).lastrowid
    af = models.insert_audio_file(
        conn, AudioFile(path="/tmp/c.flac", started_at="2026-06-16T09:00:00.000Z",
                        sample_rate=16000, status="transcribed"))
    conn.execute("UPDATE audio_files SET conversation_id=? WHERE id=?", (conv, af))
    tid = models.insert_transcript(conn, af, "mock", "mock", "en")
    models.insert_segments(conn, [models.Segment(tid, af, 0.0, 1.0, "about Atlas",
                                                  start_at="2026-06-16T09:00:00.000Z")])
    enqueue_extraction(conn, conv)
    worker.drain(conn, llm=MockLLM(), settings=settings)
    row = conn.execute("SELECT knowledge_status FROM conversations WHERE id=?", (conv,)).fetchone()
    assert row["knowledge_status"] == "extracted"


def test_failed_transcription_marks_audio_failed(conn, settings, tmp_path):
    af_id = _add_audio(conn, tmp_path)
    worker.enqueue_transcription(conn, af_id)

    class Boom(MockTranscriber):
        def transcribe(self, *a, **k):
            raise RuntimeError("model exploded")

    worker.drain(conn, transcriber=Boom(), vad=MockVad(), settings=settings, max_jobs=5)
    assert models.get_audio_file(conn, af_id)["status"] == "failed"


def test_failed_job_logs_full_traceback(conn, settings, tmp_path, caplog):
    import logging

    af_id = _add_audio(conn, tmp_path)
    worker.enqueue_transcription(conn, af_id)

    class Boom(MockTranscriber):
        def transcribe(self, *a, **k):
            raise RuntimeError("model exploded")

    with caplog.at_level(logging.ERROR, logger="secondbrain.worker"):
        worker.drain(conn, transcriber=Boom(), vad=MockVad(), settings=settings, max_jobs=1)
    assert any("model exploded" in (r.exc_text or "") for r in caplog.records)


def test_transcribe_jobs_outrank_heavy_jobs(conn, settings, tmp_path):
    from secondbrain.pipeline import queue as q
    from secondbrain.pipeline.conversation import JOB_DIARIZE

    q.enqueue(conn, JOB_DIARIZE, {"conversation_id": 1})  # heavy job queued FIRST
    af_id = _add_audio(conn, tmp_path)
    worker.enqueue_transcription(conn, af_id)
    job = q.claim_next(conn)
    assert job.type == worker.JOB_TRANSCRIBE  # live transcription is never starved


def test_transcription_is_idempotent_on_rerun(conn, settings, tmp_path):
    af_id = _add_audio(conn, tmp_path)
    transcriber = MockTranscriber([TranscribedSegment(0.0, 2.0, "hello there", 0.9)])
    n1 = worker.process_audio_file(
        conn, af_id, transcriber=transcriber, vad=MockVad(), settings=settings
    )
    # a retried job for the same chunk (worker died between commit and job
    # completion) must not duplicate the transcript/segments
    n2 = worker.process_audio_file(
        conn, af_id, transcriber=transcriber, vad=MockVad(), settings=settings
    )
    assert n1 == n2 == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM transcripts").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"] == 1


def test_vad_failure_falls_through_to_transcription(conn, settings, tmp_path, caplog):
    import logging

    settings.vad.enabled = True
    af_id = _add_audio(conn, tmp_path)

    class BrokenVad(MockVad):
        def detect(self, audio_path):
            raise RuntimeError("onnx runtime exploded")

    with caplog.at_level(logging.WARNING, logger="secondbrain.worker"):
        n = worker.process_audio_file(
            conn, af_id,
            transcriber=MockTranscriber([TranscribedSegment(0.0, 1.0, "still works", 0.9)]),
            vad=BrokenVad(), settings=settings,
        )
    assert n == 1  # transcription still happened
    assert models.get_audio_file(conn, af_id)["status"] == "transcribed"
    assert any("VAD failed" in r.message for r in caplog.records)


def test_missing_audio_file_marked_cleanly(conn, settings, tmp_path):
    from secondbrain.pipeline import queue as q

    af_id = models.insert_audio_file(
        conn, AudioFile(path=str(tmp_path / "vanished.flac"),
                        started_at="2026-06-16T09:00:00.000Z", sample_rate=16000)
    )
    worker.enqueue_transcription(conn, af_id)

    class Boom(MockTranscriber):
        def transcribe(self, *a, **k):  # the backend must never even be called
            raise AssertionError("backend called for a missing file")

    worker.drain(conn, transcriber=Boom(), vad=MockVad(), settings=settings)
    assert models.get_audio_file(conn, af_id)["status"] == "missing"
    assert q.counts(conn).get("done") == 1  # completed, not failed into retries


def test_speech_seconds_persisted_and_min_gate(conn, settings, tmp_path):
    settings.vad.enabled = True
    settings.transcription.min_speech_seconds = 2.0
    af_id = _add_audio(conn, tmp_path)
    n = worker.process_audio_file(
        conn, af_id,
        transcriber=MockTranscriber([TranscribedSegment(0.0, 1.0, "blip", 0.9)]),
        vad=MockVad(has_speech=True, duration_s=0.5),  # 0.5s speech < 2.0s minimum
        settings=settings,
    )
    assert n == 0
    row = models.get_audio_file(conn, af_id)
    assert row["speech_seconds"] == 0.5      # persisted from the VAD spans
    assert row["has_speech"] == 0            # gated: treated like silence
    assert row["status"] == "transcribed"

    # default 0.0 keeps the old behavior: any speech is transcribed
    settings.transcription.min_speech_seconds = 0.0
    af2 = _add_audio(conn, tmp_path, name="chunk2.flac")
    n = worker.process_audio_file(
        conn, af2,
        transcriber=MockTranscriber([TranscribedSegment(0.0, 1.0, "hello", 0.9)]),
        vad=MockVad(has_speech=True, duration_s=0.5),
        settings=settings,
    )
    assert n == 1
    row = models.get_audio_file(conn, af2)
    assert row["speech_seconds"] == 0.5
    assert row["has_speech"] == 1
