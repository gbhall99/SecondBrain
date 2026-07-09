"""Memory timeline — day grouped into conversations with inline extractions."""

from __future__ import annotations

from datetime import UTC, datetime

from secondbrain.query import service


def _local_hhmm(ts: str) -> str:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%H:%M")


def _utc_at_local(day: str, hh: int, mm: int) -> str:
    """UTC storage timestamp for a *local* wall-clock time (tz-robust tests)."""
    naive = datetime.strptime(f"{day} {hh:02d}:{mm:02d}", "%Y-%m-%d %H:%M")
    return naive.astimezone().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _conv(conn, cid, day):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (?, ?, 'diarized')",
        (cid, f"{day}T09:00:00.000Z"),
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (?, ?, ?, 16000, 'transcribed', ?)",
        (cid, f"/tmp/{cid}.flac", f"{day}T09:00:00.000Z", cid),
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')", (cid, cid)
    )


def _seg(conn, sid, aid, day, sec, text, speaker_id):
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
        " speaker_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, aid, aid, sec, sec + 2.0, f"{day}T09:00:{sec:02d}.000Z", text, speaker_id),
    )


def _seed(conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Dana', 'known', 0)")
    _conv(conn, 1, "2026-06-16")
    _seg(conn, 1, 1, "2026-06-16", 0, "kick off the project", 1)
    _seg(conn, 2, 1, "2026-06-16", 2, "sounds good", 2)
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (1, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, conversation_id, valid, "
        " source_segment_ids) VALUES (1, 1, 'decision', 'start Atlas', 1, 1, '[1]')"
    )


def test_timeline_groups_conversation_with_extractions(conn, settings):
    _seed(conn)
    tl = service.timeline(conn, "2026-06-16", settings)
    assert len(tl) == 1
    block = tl[0]
    assert block["conversation_id"] == 1
    assert block["participants"] == ["Dana", "Me"]
    assert [s["text"] for s in block["segments"]] == ["kick off the project", "sounds good"]
    assert block["extractions"]["decision"][0]["object_text"] == "start Atlas"
    assert block["extractions"]["decision"][0]["segment_ids"] == [1]


def test_timeline_empty_day(conn, settings):
    _seed(conn)
    assert service.timeline(conn, "2026-01-01", settings) == []


def test_timeline_opt_out_filtered(conn, settings):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    tl = service.timeline(conn, "2026-06-16", settings)
    texts = [s["text"] for s in tl[0]["segments"]]
    assert "sounds good" not in texts  # Dana opted out
    assert "kick off the project" in texts
    assert "Dana" not in tl[0]["participants"]


def test_timeline_block_metadata(conn, settings):
    _seed(conn)
    b = service.timeline(conn, "2026-06-16", settings)[0]
    # seg 1: 09:00:00 + 2 s, seg 2: 09:00:02 + 2 s -> block spans 09:00:00–09:00:04 UTC
    assert b["started_at"] == "2026-06-16T09:00:00.000Z"
    assert b["ended_at"] == "2026-06-16T09:00:04.000Z"
    assert b["duration_seconds"] == 4.0
    assert b["duration_minutes"] == 0
    assert b["duration_label"] == "under 1 min"
    assert b["segment_count"] == 2
    # display times are local wall-clock, minute precision
    assert b["start_time"] == _local_hhmm("2026-06-16T09:00:00.000Z")
    assert b["end_time"] == _local_hhmm("2026-06-16T09:00:04.000Z")
    assert b["segments"][0]["time"] == _local_hhmm("2026-06-16T09:00:00.000Z")


def test_duration_label_scales():
    assert service.duration_label(30) == "under 1 min"
    assert service.duration_label(21 * 60) == "21 min"
    assert service.duration_label(125 * 60) == "2 h 05 min"
    assert service.duration_label(120 * 60) == "2 h"


def test_timeline_invalid_day_is_empty_not_error(conn, settings):
    _seed(conn)
    assert service.timeline(conn, "not-a-date", settings) == []


def test_timeline_buckets_by_local_day(conn, settings):
    _seed(conn)
    # A 23:30 UTC conversation belongs to whatever *local* day that instant
    # falls on (the next day on machines east of UTC), not the UTC date.
    _conv(conn, 2, "2026-06-20")
    conn.execute(
        "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, start_offset_s,"
        " end_offset_s, start_at, text) VALUES (10, 2, 2, 0, 2, '2026-06-20T23:30:00.000Z',"
        " 'late night')"
    )
    local_day = datetime(2026, 6, 20, 23, 30, tzinfo=UTC).astimezone().strftime("%Y-%m-%d")
    tl = service.timeline(conn, local_day, settings)
    assert any(s["text"] == "late night" for b in tl for s in b["segments"])
    if local_day != "2026-06-20":  # machine is offset from UTC: the UTC date must NOT have it
        tl_utc = service.timeline(conn, "2026-06-20", settings)
        assert not any(s["text"] == "late night" for b in tl_utc for s in b["segments"])


def test_timeline_splits_untracked_segments_on_silence(conn, settings):
    _seed(conn)
    # segments with no conversation id (mid-transcription / legacy) bucket by gaps
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (5, '/tmp/5.flac', '2026-06-18T09:00:00.000Z', 16000, 'transcribed')"
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (5, 5, 'mock')")
    for sid, ts in ((20, "2026-06-18T09:00:00.000Z"), (21, "2026-06-18T09:01:00.000Z"),
                    (22, "2026-06-18T11:00:00.000Z")):
        conn.execute(
            "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, start_offset_s,"
            " end_offset_s, start_at, text) VALUES (?, 5, 5, 0, 2, ?, 'x')", (sid, ts))
    tl = service.timeline(conn, "2026-06-18", settings)
    assert len(tl) == 2  # the two-hour silence starts a new block
    assert [b["segment_count"] for b in tl] == [2, 1]
    assert all(b["conversation_id"] is None for b in tl)


def test_timeline_interleaved_conversations_stay_grouped(conn, settings):
    _seed(conn)
    # two capture sources overlapping: segments interleave but stay in their block
    _conv(conn, 3, "2026-06-19")
    _conv(conn, 4, "2026-06-19")
    for sid, cid, ts in ((30, 3, "2026-06-19T09:00:00.000Z"),
                         (31, 4, "2026-06-19T09:00:10.000Z"),
                         (32, 3, "2026-06-19T09:00:20.000Z")):
        conn.execute(
            "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, start_offset_s,"
            " end_offset_s, start_at, text) VALUES (?, ?, ?, 0, 2, ?, 'x')", (sid, cid, cid, ts))
    tl = service.timeline(conn, "2026-06-19", settings)
    assert [b["conversation_id"] for b in tl] == [3, 4]
    assert tl[0]["segment_count"] == 2


def test_timeline_strip_geometry_and_lanes():
    day = "2026-06-16"
    mk = _utc_at_local
    blocks = [
        {"started_at": mk(day, 6, 0), "ended_at": mk(day, 7, 0), "start_time": "06:00",
         "end_time": "07:00", "duration_label": "1 h", "segment_count": 3,
         "participants": ["Me"], "segments": []},
        {"started_at": mk(day, 6, 30), "ended_at": mk(day, 6, 45), "start_time": "06:30",
         "end_time": "06:45", "duration_label": "15 min", "segment_count": 1,
         "participants": ["Dana"], "segments": []},
    ]
    strip = service.timeline_strip(blocks, day)
    assert strip["lanes"] == 2  # overlapping conversations sit in separate lanes
    a, b = strip["spans"]
    assert a["lane"] == 0 and b["lane"] == 1
    assert a["left"] == 25.0  # 06:00 local = a quarter of the day in
    assert 4.0 < a["width"] < 4.5  # one hour ≈ 4.17% of the day
    assert b["left"] > a["left"]
    assert "3 lines" in a["label"] and "Me" in a["label"]
    # malformed day / no blocks -> harmless empty geometry
    assert service.timeline_strip(blocks, "bogus") == {"lanes": 1, "spans": [], "zoom": None}
    assert service.timeline_strip([], day) == {"lanes": 1, "spans": [], "zoom": None}


def test_timeline_strip_zooms_narrow_evening_window():
    # An evening-only day renders as slivers on the 24 h axis, so the strip
    # also returns a magnified whole-hour window with its own geometry.
    day = "2026-06-16"
    mk = _utc_at_local
    blocks = [
        {"started_at": mk(day, 22, 10), "ended_at": mk(day, 23, 40), "start_time": "22:10",
         "end_time": "23:40", "duration_label": "1 h 30 min", "segment_count": 4,
         "participants": ["Me"], "segments": []},
    ]
    strip = service.timeline_strip(blocks, day)
    z = strip["zoom"]
    assert z is not None
    assert z["start_label"] == "22:00" and z["end_label"] == "24:00"
    assert [lb["text"] for lb in z["labels"]] == ["22:00", "23:00", "24:00"]
    assert [lb["pct"] for lb in z["labels"]] == [0.0, 50.0, 100.0]
    assert z["start_pct"] + z["width_pct"] == z["end_pct"] == 100.0
    sp = strip["spans"][0]
    # 22:10 is 10 min into the 2 h window; 1.5 h fills three quarters of it
    assert 8.0 < sp["zoom_left"] < 8.7
    assert 74.5 <= sp["zoom_width"] <= 75.5
    # the 24 h axis keeps its own sliver geometry for orientation
    assert sp["left"] > 90.0 and sp["width"] < 7.0


def test_day_nav_skips_days_whose_speech_is_all_opted_out(conn, settings):
    _seed(conn)  # 2026-06-16 has Me (1) + Dana (2)
    # 2026-06-18: Dana only
    _conv(conn, 8, "2026-06-18")
    conn.execute(
        "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, start_offset_s,"
        " end_offset_s, start_at, text, speaker_id) VALUES (80, 8, 8, 0, 2, ?, 'private', 2)",
        (_utc_at_local("2026-06-18", 9, 0),),
    )
    assert service.day_nav(conn, "2026-06-20", settings)["prev_day_with_data"] == "2026-06-18"
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    # Dana opted out -> 06-18 would render as "Nothing recorded"; skip past it
    assert service.day_nav(conn, "2026-06-20", settings)["prev_day_with_data"] == "2026-06-16"
    assert service.day_nav(conn, "2026-06-17", settings)["next_day_with_data"] is None
    # 06-16 still counts (Me spoke there too), from either direction
    assert service.day_nav(conn, "2026-06-15", settings)["next_day_with_data"] == "2026-06-16"


def test_day_nav_counts_unattributed_speech(conn, settings):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    # 2026-06-21: a single line nobody is attributed to yet (speaker NULL)
    _conv(conn, 9, "2026-06-21")
    conn.execute(
        "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, start_offset_s,"
        " end_offset_s, start_at, text) VALUES (90, 9, 9, 0, 2, ?, 'who was that')",
        (_utc_at_local("2026-06-21", 9, 0),),
    )
    assert service.day_nav(conn, "2026-06-23", settings)["prev_day_with_data"] == "2026-06-21"


def test_timeline_strip_no_zoom_for_spread_days():
    day = "2026-06-16"
    mk = _utc_at_local
    blocks = [
        {"started_at": mk(day, 8, 0), "ended_at": mk(day, 9, 0), "start_time": "08:00",
         "end_time": "09:00", "duration_label": "1 h", "segment_count": 1,
         "participants": ["Me"], "segments": []},
        {"started_at": mk(day, 20, 0), "ended_at": mk(day, 21, 0), "start_time": "20:00",
         "end_time": "21:00", "duration_label": "1 h", "segment_count": 1,
         "participants": ["Me"], "segments": []},
    ]
    strip = service.timeline_strip(blocks, day)
    # 08:00 -> 21:00 already reads fine on the 24 h axis: no zoom, no zoom keys
    assert strip["zoom"] is None
    assert all("zoom_left" not in sp for sp in strip["spans"])


def test_timeline_strip_zoom_widens_odd_windows_for_even_labels():
    # 08:50–14:40 is under the quarter-day threshold, but its whole-hour hull
    # (08:00–15:00) is 7 h — odd, so 2-hour labels wouldn't divide evenly.
    # The zoom widens by one hour and labels every other whole hour.
    day = "2026-06-16"
    mk = _utc_at_local
    blocks = [
        {"started_at": mk(day, 8, 50), "ended_at": mk(day, 14, 40), "start_time": "08:50",
         "end_time": "14:40", "duration_label": "5 h 50 min", "segment_count": 2,
         "participants": ["Me"], "segments": []},
    ]
    z = service.timeline_strip(blocks, day)["zoom"]
    assert z is not None
    assert z["start_label"] == "07:00" and z["end_label"] == "15:00"
    assert [lb["text"] for lb in z["labels"]] == ["07:00", "09:00", "11:00", "13:00", "15:00"]
    assert [lb["pct"] for lb in z["labels"]] == [0.0, 25.0, 50.0, 75.0, 100.0]
