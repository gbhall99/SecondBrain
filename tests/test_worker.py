from secondbrain.pipeline import worker
from secondbrain.pipeline.transcribe import MockTranscriber, TranscribedSegment
from secondbrain.pipeline.vad import MockVad
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile


def _add_audio(conn, started_at="2026-06-16T09:00:00.000Z"):
    return models.insert_audio_file(
        conn, AudioFile(path="/tmp/none.flac", started_at=started_at, sample_rate=16000)
    )


def test_transcription_end_to_end(conn, settings):
    af_id = _add_audio(conn)
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


def test_diarization_enabled_defers_retention_and_groups_conversation(conn, settings):
    settings.diarization.enabled = True
    af_id = _add_audio(conn)
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


def test_failed_transcription_marks_audio_failed(conn, settings):
    af_id = _add_audio(conn)
    worker.enqueue_transcription(conn, af_id)

    class Boom(MockTranscriber):
        def transcribe(self, *a, **k):
            raise RuntimeError("model exploded")

    worker.drain(conn, transcriber=Boom(), vad=MockVad(), settings=settings, max_jobs=5)
    assert models.get_audio_file(conn, af_id)["status"] == "failed"
