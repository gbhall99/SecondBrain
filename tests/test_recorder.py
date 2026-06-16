from secondbrain.capture import recorder
from secondbrain.pipeline import queue as q
from secondbrain.storage import models, state


def test_register_chunk_inserts_and_enqueues(conn, settings):
    af_id = recorder.register_chunk(
        conn,
        path=settings.audio_raw_dir / "20260616-090000.flac",
        started_at="2026-06-16T09:00:00.000Z",
        ended_at="2026-06-16T09:01:00.000Z",
        duration_s=60.0,
        settings=settings,
    )
    assert models.get_audio_file(conn, af_id)["status"] == "recorded"
    # a transcription job should be queued for it
    job = q.claim_next(conn, "transcribe")
    assert job is not None and job.payload["audio_file_id"] == af_id


def test_should_record_respects_consent_and_pause(conn, settings):
    ok, _ = recorder.should_record(settings, conn)
    assert ok is True

    state.set_paused(conn, True)
    ok, reason = recorder.should_record(settings, conn)
    assert ok is False and "paused" in reason
    state.set_paused(conn, False)

    settings.consent.recording_enabled = False
    ok, reason = recorder.should_record(settings, conn)
    assert ok is False and "disabled" in reason


def test_should_record_disk_guardrail(conn, settings):
    settings.consent.recording_enabled = True
    settings.capture.min_free_disk_gb = 10**9  # impossibly high -> guardrail trips
    ok, reason = recorder.should_record(settings, conn)
    assert ok is False and "disk" in reason
