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


def test_sweep_force_expires_orphan_deferred_chunk(conn, settings, tmp_path):
    """A diarization-deferred chunk (NULL deadline) stuck past the grace is swept."""
    settings.consent.raw_audio_retention_hours = 168
    audio = tmp_path / "orphan.flac"
    audio.write_bytes(b"\x00")
    af_id = models.insert_audio_file(
        conn,
        AudioFile(
            path=str(audio),
            started_at="2000-01-01T00:00:00.000Z",  # long past the retention+grace cutoff
            sample_rate=16000,
            status="transcribed",
            retention_delete_after=None,  # deferred, never finalized
        ),
    )
    assert retention.sweep_expired_audio(conn, settings) == 1
    assert not audio.exists()
    assert models.get_audio_file(conn, af_id)["status"] == "deleted"


def test_sweep_keeps_recent_deferred_chunk(conn, settings, tmp_path):
    """A recently-deferred chunk (NULL deadline) is kept until diarization finalizes."""
    settings.consent.raw_audio_retention_hours = 168
    audio = tmp_path / "recent.flac"
    audio.write_bytes(b"\x00")
    models.insert_audio_file(
        conn,
        AudioFile(
            path=str(audio),
            started_at=models.utcnow_iso(),  # just captured
            sample_rate=16000,
            status="transcribed",
            retention_delete_after=None,
        ),
    )
    assert retention.sweep_expired_audio(conn, settings) == 0
    assert audio.exists()


def test_sweep_keeps_deferred_chunk_when_retention_forever(conn, settings, tmp_path):
    """NULL deadline + keep-forever retention means keep, even for old chunks."""
    settings.consent.raw_audio_retention_hours = -1
    audio = tmp_path / "keep.flac"
    audio.write_bytes(b"\x00")
    models.insert_audio_file(
        conn,
        AudioFile(
            path=str(audio),
            started_at="2000-01-01T00:00:00.000Z",
            sample_rate=16000,
            status="transcribed",
            retention_delete_after=None,
        ),
    )
    assert retention.sweep_expired_audio(conn, settings) == 0
    assert audio.exists()


def test_sweep_removes_derived_sample_clips(conn, settings, tmp_path):
    """Cached voice-sample clips are derived raw audio: they follow retention."""
    from secondbrain.speaker import registry

    def _obs(af_id):
        sid = registry.create_unknown_speaker(conn)
        return registry.record_observation(
            conn, speaker_id=sid, audio_file_id=af_id, conversation_id=None,
            start_offset_s=0.0, end_offset_s=1.0, start_at=None,
            confidence=0.9, embedding=[1.0],
        )

    expired = tmp_path / "old.flac"
    expired.write_bytes(b"\x00\x01")
    af_old = models.insert_audio_file(
        conn,
        AudioFile(path=str(expired), started_at="2026-06-10T09:00:00.000Z", sample_rate=16000,
                  status="transcribed", retention_delete_after="2000-01-01T00:00:00.000Z"),
    )
    fresh = tmp_path / "fresh.flac"
    fresh.write_bytes(b"\x00")
    af_new = models.insert_audio_file(
        conn,
        AudioFile(path=str(fresh), started_at="2026-06-16T09:00:00.000Z", sample_rate=16000,
                  status="transcribed", retention_delete_after="2999-01-01T00:00:00.000Z"),
    )
    clip_dir = settings.audio_processed_dir
    clip_dir.mkdir(parents=True, exist_ok=True)
    obs_old, obs_new = _obs(af_old), _obs(af_new)
    swept_clip = clip_dir / f"sample_{obs_old}.wav"
    swept_clip.write_bytes(b"RIFF")
    # window-stamped cache names (sample_{id}_{start}-{end}.wav) follow suit
    swept_windowed = clip_dir / f"sample_{obs_old}_0-100.wav"
    swept_windowed.write_bytes(b"RIFF")
    kept_clip = clip_dir / f"sample_{obs_new}.wav"
    kept_clip.write_bytes(b"RIFF")
    kept_windowed = clip_dir / f"sample_{obs_new}_50-1050.wav"
    kept_windowed.write_bytes(b"RIFF")
    stray_clip = clip_dir / "sample_99999.wav"  # observation no longer exists
    stray_clip.write_bytes(b"RIFF")
    unrelated = clip_dir / "conv_concat.wav"  # diarization scratch file: not ours
    unrelated.write_bytes(b"RIFF")

    assert retention.sweep_expired_audio(conn, settings) == 1
    assert not swept_clip.exists()      # source swept -> derived clip removed
    assert not swept_windowed.exists()  # stamped variant swept the same way
    assert not stray_clip.exists()      # orphaned cache entries cleaned up
    assert kept_clip.exists()           # live source -> cache stays valid
    assert kept_windowed.exists()
    assert unrelated.exists()


def test_sweep_removes_derived_segment_clips(conn, settings, tmp_path):
    """Day-view per-line clips (segclip_*.wav) follow retention like samples."""
    expired = tmp_path / "old.flac"
    expired.write_bytes(b"\x00\x01")
    af_old = models.insert_audio_file(
        conn,
        AudioFile(path=str(expired), started_at="2026-06-10T09:00:00.000Z", sample_rate=16000,
                  status="transcribed", retention_delete_after="2000-01-01T00:00:00.000Z"),
    )
    fresh = tmp_path / "fresh.flac"
    fresh.write_bytes(b"\x00")
    af_new = models.insert_audio_file(
        conn,
        AudioFile(path=str(fresh), started_at="2026-06-16T09:00:00.000Z", sample_rate=16000,
                  status="transcribed", retention_delete_after="2999-01-01T00:00:00.000Z"),
    )
    t_old = models.insert_transcript(conn, af_old, "mock", "mock", "en")
    t_new = models.insert_transcript(conn, af_new, "mock", "mock", "en")
    models.insert_segments(
        conn,
        [
            models.Segment(t_old, af_old, 0.0, 1.0, "old line",
                           start_at="2026-06-10T09:00:00.000Z"),
            models.Segment(t_new, af_new, 0.0, 1.0, "fresh line",
                           start_at="2026-06-16T09:00:00.000Z"),
        ],
    )
    seg_old, seg_new = (
        r["id"]
        for r in conn.execute("SELECT id FROM transcript_segments ORDER BY id").fetchall()
    )
    clip_dir = settings.audio_processed_dir
    clip_dir.mkdir(parents=True, exist_ok=True)
    swept_clip = clip_dir / f"segclip_{seg_old}.wav"
    swept_clip.write_bytes(b"RIFF")
    kept_clip = clip_dir / f"segclip_{seg_new}.wav"
    kept_clip.write_bytes(b"RIFF")
    stray_clip = clip_dir / "segclip_99999.wav"  # segment no longer exists
    stray_clip.write_bytes(b"RIFF")

    assert retention.sweep_expired_audio(conn, settings) == 1
    assert not swept_clip.exists()  # source swept -> derived clip removed
    assert not stray_clip.exists()  # orphaned cache entries cleaned up
    assert kept_clip.exists()       # live source -> cache stays valid


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
