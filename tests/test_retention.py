from secondbrain.storage import models, retention
from secondbrain.storage.models import AudioFile


def test_compute_delete_after_modes(settings):
    settings.consent.raw_audio_retention_hours = -1
    assert retention.compute_delete_after(settings) is None  # keep forever

    settings.consent.raw_audio_retention_hours = 24
    assert retention.compute_delete_after(settings) is not None


def test_sweep_deletes_expired_transcribed_audio(conn, settings, tmp_path):
    audio = tmp_path / "old.flac"
    audio.write_bytes(b"\x00\x01")
    af_id = models.insert_audio_file(
        conn,
        AudioFile(
            path=str(audio),
            started_at="2026-06-10T09:00:00.000Z",
            sample_rate=16000,
            status="transcribed",
            retention_delete_after="2000-01-01T00:00:00.000Z",  # long past
        ),
    )
    deleted = retention.sweep_expired_audio(conn, settings)
    assert deleted == 1
    assert not audio.exists()
    assert models.get_audio_file(conn, af_id)["status"] == "deleted"


def test_sweep_keeps_unexpired_audio(conn, settings, tmp_path):
    audio = tmp_path / "fresh.flac"
    audio.write_bytes(b"\x00")
    models.insert_audio_file(
        conn,
        AudioFile(
            path=str(audio),
            started_at="2026-06-16T09:00:00.000Z",
            sample_rate=16000,
            status="transcribed",
            retention_delete_after="2999-01-01T00:00:00.000Z",
        ),
    )
    assert retention.sweep_expired_audio(conn, settings) == 0
    assert audio.exists()
