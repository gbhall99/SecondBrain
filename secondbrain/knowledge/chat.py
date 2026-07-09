"""Graph-RAG grounded Q&A.

Retrieve cited transcript spans (FTS + optional semantic) and a small relevant
subgraph, assemble a context block, and ask the local LLM. Grounded claims must
cite [seg_id]; general knowledge is allowed but must be clearly labeled (per the
user's "grounded + general" choice).

Temporal questions ("What did I talk about today?") additionally pull the
segments from the asked-about local date window into the context, since pure
similarity retrieval has no notion of *when* something was said.

``prepare`` + ``finalize`` split the work around the LLM call so the streaming
endpoint can retrieve, stream tokens, then resolve citations — ``answer`` is
the one-shot composition of the two.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.knowledge import graph
from secondbrain.llm.client import LLM, get_llm
from secondbrain.search import combined
from secondbrain.speaker import registry

_CITE = re.compile(r"\[(?:seg_id=)?(\d+)\]")
_GENERAL_TAG = "(general knowledge"

# History passed to answer() is clamped so a long chat can't starve the
# retrieval context out of the prompt window.
_MAX_HISTORY_TURNS = 3
_MAX_HISTORY_ANSWER_CHARS = 1200

# Bounds both latency and the cost of a generation the user has walked away
# from (Ollama keeps generating after a client disconnect on the non-streaming
# endpoint; ~1024 tokens is a generous ceiling for a cited chat answer).
MAX_ANSWER_TOKENS = 1024

# Most recent segments merged into the context for a temporal question.
_MAX_WINDOW_SEGMENTS = 20

_SYSTEM = (
    "You are the user's second brain. Answer the question using the provided context "
    "(transcript excerpts and known facts). Cite every claim drawn from the context "
    "with its [seg_id]. You MAY add helpful general knowledge to fill gaps, but you "
    "MUST prefix any such sentence with '(general knowledge — not from your data)'. "
    "If the context doesn't cover something and you don't know, say so. "
    "When a previous conversation is provided, resolve pronouns and follow-up "
    "questions against it. "
    "Write plain conversational prose — no markdown formatting (no **bold**, no "
    "headings, no bullet syntax)."
)


def _seg_info(conn: sqlite3.Connection, seg_ids: list[int]) -> dict[int, dict]:
    if not seg_ids:
        return {}
    ph = ",".join("?" * len(seg_ids))
    rows = conn.execute(
        f"""
        SELECT ts.id, ts.text, ts.start_at, ts.speaker_id, af.conversation_id,
               COALESCE(sp.name, sp.display_label) AS speaker,
               CASE WHEN sp.is_owner THEN 1 ELSE 0 END AS is_owner
        FROM transcript_segments ts
        JOIN audio_files af ON af.id = ts.audio_file_id
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE ts.id IN ({ph})
        """,
        seg_ids,
    ).fetchall()
    opted = registry.opted_out_speaker_ids(conn)
    out = {}
    for r in rows:
        if r["speaker_id"] in opted:  # never surface opted-out speech in answers
            continue
        d = dict(r)
        d["speaker"] = "Me" if r["is_owner"] else (r["speaker"] or "Unknown")
        out[r["id"]] = d
    return out


def _seed_nodes(conn: sqlite3.Connection, seg_ids: list[int], question: str) -> list[int]:
    nodes: set[int] = set()
    # Nodes whose edges cite the retrieved segments. json_each unpacks the
    # source_segment_ids JSON array inside SQLite so we never materialize the
    # whole edge table in Python on this hot path.
    if seg_ids:
        ph = ",".join("?" * len(seg_ids))
        rows = conn.execute(
            f"""
            SELECT DISTINCT e.src_node_id, e.dst_node_id
            FROM kg_edges e, json_each(COALESCE(e.source_segment_ids, '[]')) js
            WHERE e.valid = 1 AND js.value IN ({ph})
            """,
            list(seg_ids),
        ).fetchall()
        for r in rows:
            nodes.add(graph.resolve_node_id(conn, r["src_node_id"]))
            if r["dst_node_id"]:
                nodes.add(graph.resolve_node_id(conn, r["dst_node_id"]))
    # Nodes whose normalized name appears in the question (matched in SQL for
    # the same reason — no full-table scan into Python).
    qnorm = graph.normalize_name(question)
    if qnorm:
        rows = conn.execute(
            """
            SELECT id FROM kg_nodes
            WHERE merged_into IS NULL AND normalized_name <> ''
              AND instr(?, normalized_name) > 0
            """,
            (qnorm,),
        ).fetchall()
        nodes.update(int(r["id"]) for r in rows)
    return list(nodes)


def _subgraph_facts(
    conn: sqlite3.Connection, node_ids: list[int], settings: Settings
) -> list[dict]:
    if not node_ids:
        return []
    ph = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"""
        SELECT e.predicate, e.kind, e.object_text, e.due_date, e.source_segment_ids,
               s.name AS src_name, d.name AS dst_name
        FROM kg_edges e
        JOIN kg_nodes s ON s.id = e.src_node_id
        LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
        WHERE e.valid=1 AND (e.src_node_id IN ({ph}) OR e.dst_node_id IN ({ph}))
        ORDER BY e.confidence DESC
        LIMIT ?
        """,
        [*node_ids, *node_ids, settings.extraction.chat_max_facts],
    ).fetchall()
    return [dict(r) for r in rows]


def _fact_line(f: dict) -> str:
    obj = f["dst_name"] or f["object_text"] or ""
    due = f" (due {f['due_date']})" if f.get("due_date") else ""
    cites = json.loads(f["source_segment_ids"] or "[]")
    cite = " " + " ".join(f"[{c}]" for c in cites) if cites else ""
    return f"- {f['src_name']} {f['predicate'] or f['kind']} {obj}{due}{cite}"


def _history_block(history: list[dict] | None) -> tuple[str, list[int]]:
    """Prompt block for prior turns + the segment ids those answers cited.

    Only the most recent ``_MAX_HISTORY_TURNS`` turns are spelled out as prose
    (bounding prompt cost), but citation ids are harvested from *every* provided
    turn — so a follow-up that reaches back to an early turn ("what did I mean
    earlier about X?") keeps those sources resolvable even though the older
    turn's text isn't re-sent. Older turns whose only value was a citation thus
    stay grounded; their prose context is what drops.
    """
    if not history:
        return "", []
    valid = [
        t for t in history
        if isinstance(t, dict) and str(t.get("question") or "").strip()
        and str(t.get("answer") or "").strip()
    ]
    if not valid:
        return "", []
    # Harvest cited ids across the whole (validated) history for id carryover.
    cited: list[int] = []
    for t in valid:
        cited.extend(int(m) for m in _CITE.findall(str(t.get("answer") or "")))
    # Spell out only the most recent turns to keep the prompt bounded.
    lines: list[str] = []
    for turn in valid[-_MAX_HISTORY_TURNS:]:
        q = str(turn.get("question") or "").strip()
        a = str(turn.get("answer") or "").strip()
        if len(a) > _MAX_HISTORY_ANSWER_CHARS:
            a = a[:_MAX_HISTORY_ANSWER_CHARS] + " …"
        lines.append(f"User asked: {q}\nYou answered: {a}")
    return "Previous conversation (context for follow-ups):\n" + "\n\n".join(lines) + "\n\n", cited


# --- time-aware retrieval ------------------------------------------------------

# Ordered: first match wins, so specific phrases beat the generic "recently".
_TEMPORAL_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:last|past)\s+(\d{1,2})\s+days?\b", re.I), "lastn"),
    (re.compile(r"\b(?:today|this\s+(?:morning|afternoon|evening)|tonight)\b", re.I), "today"),
    (re.compile(r"\b(?:yesterday|last\s+night)\b", re.I), "yesterday"),
    (re.compile(r"\bthis\s+week\b", re.I), "this_week"),
    (re.compile(r"\blast\s+week\b", re.I), "last_week"),
    (re.compile(r"\bpast\s+week\b", re.I), "past7"),
    (re.compile(r"\bthis\s+month\b", re.I), "this_month"),
    (re.compile(r"\b(?:recent(?:ly)?|lately)\b", re.I), "past7"),
]

# Month names / common abbreviations → month number (1-12). Abbreviations are
# accepted with or without a trailing dot ("Jul", "Jul.", "Sept").
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))  # longest-first: 'june' before 'jun'
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"  # day-of-month, tolerating an ordinal suffix
_YEAR = r"(\d{4})"

# Explicit-date forms, tried before the relative phrases so "on July 2" beats a
# stray "recently" and resolves to a single day rather than the last-7 fallback.
# Each yields (year|None, month, day); a missing year is resolved to the most
# recent past-or-today occurrence (the natural reading for a retrospective).
_ABSOLUTE_DATES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{_YEAR}-(\d{{2}})-(\d{{2}})\b"), "iso"),  # 2026-07-02
    # 'July 2', 'Jul 2nd 2026'  /  '2 July', '2nd Jul 2026'
    (re.compile(rf"\b(?:{_MONTH_ALT})\.?\s+{_DAY}(?:,?\s+{_YEAR})?\b", re.I), "mdy"),
    (re.compile(rf"\b{_DAY}\s+(?:{_MONTH_ALT})\.?(?:,?\s+{_YEAR})?\b", re.I), "dmy"),
    (re.compile(rf"\b(\d{{1,2}})/(\d{{1,2}})/{_YEAR}\b"), "slash_us"),  # 07/02/2026 (M/D/Y)
]


def _resolve_year(month: int, day: int, today: date) -> int:
    """Year for a bare month/day: the most recent occurrence at or before today.

    "What did I talk about on July 2?" asked on July 9 means *this* year's July 2;
    asked on Jan 3 it means *last* year's July 2. Keeps a year-less date from
    silently pointing at a day that hasn't happened yet.
    """
    year = today.year
    try:
        candidate = date(year, month, day)
    except ValueError:  # e.g. Feb 29 on a non-leap year — step back a year and retry
        return year - 1
    return year if candidate <= today else year - 1


def _absolute_day(question: str, today: date) -> date | None:
    """The single calendar date an explicit-date question names, or None.

    Recognizes ISO (2026-07-02), month-name forms ("on July 2", "Jul 2 2026",
    "2 July 2026"), and slashed M/D/Y (07/02/2026). Returns None for bare month
    names with no day, or day/month values that don't form a real date.
    """
    for pattern, kind in _ABSOLUTE_DATES:
        m = pattern.search(question)
        if not m:
            continue
        if kind == "iso":
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        elif kind == "slash_us":
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:  # month-name forms: locate the month token and the numeric groups
            month, day, year = _month_name_parts(m, kind)
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        if year is None:  # bare 'July 2' → most recent past-or-today July 2
            year = _resolve_year(month, day, today)
        try:
            return date(year, month, day)
        except ValueError:
            continue  # not a real date (e.g. Feb 30, Jun 31) — keep scanning
    return None


def _month_name_parts(m: re.Match[str], kind: str) -> tuple[int, int, int | None]:
    """Extract (month, day, year|None) from a month-name date match.

    Both 'July 2[, 2026]' and '2 July[ 2026]' expose the day in group 1 and the
    optional year in group 2; the month is the name token within the match.
    """
    name = re.search(rf"(?i)\b({_MONTH_ALT})\b", m.group(0))
    month = _MONTHS[name.group(1).lower()] if name else 0
    day = int(m.group(1))
    year = int(m.group(2)) if m.group(2) else None
    return month, day, year


def _single_day_window(d: date, today: date) -> dict:
    """A one-day window keyed by an explicit date, labelled for the badge.

    Reads relative ("today"/"yesterday") when the date is one of those — matching
    how the user would say it back — otherwise a plain absolute label
    ("Jul 2, 2026"). ``start_day == end_day`` marks it a single day to callers.
    """
    iso = str(d)
    if d == today:
        label = "today"
    elif d == today - timedelta(days=1):
        label = "yesterday"
    else:
        label = f"{d.strftime('%b')} {d.day}, {d.year}"  # 'Jul 2, 2026'
    return {"label": label, "start_day": iso, "end_day": iso}


def _temporal_window(question: str, now: datetime | None = None) -> dict | None:
    """Local calendar-day window a temporal question refers to, or None.

    Returns ``{"label", "start_day", "end_day"}`` with inclusive local dates
    (YYYY-MM-DD). Retrieval alone can't serve "What did I talk about today?" —
    similarity has no clock — so callers merge this window's segments into the
    context. Explicit dates ("on July 2nd", "2026-07-02") resolve to a single
    day and are matched before the relative phrases below.
    """
    now = now or datetime.now().astimezone()
    today = now.date()
    explicit = _absolute_day(question, today)
    if explicit is not None:
        return _single_day_window(explicit, today)
    for pattern, kind in _TEMPORAL_PHRASES:
        m = pattern.search(question)
        if not m:
            continue
        if kind == "lastn":
            n = max(1, min(int(m.group(1)), 90))
            return {
                "label": f"the last {n} day{'s' if n != 1 else ''}",
                "start_day": str(today - timedelta(days=n - 1)),
                "end_day": str(today),
            }
        if kind == "today":
            return {"label": "today", "start_day": str(today), "end_day": str(today)}
        if kind == "yesterday":
            d = today - timedelta(days=1)
            return {"label": "yesterday", "start_day": str(d), "end_day": str(d)}
        if kind == "this_week":
            monday = today - timedelta(days=today.weekday())
            return {"label": "this week", "start_day": str(monday), "end_day": str(today)}
        if kind == "last_week":
            monday = today - timedelta(days=today.weekday() + 7)
            return {
                "label": "last week",
                "start_day": str(monday),
                "end_day": str(monday + timedelta(days=6)),
            }
        if kind == "this_month":
            return {
                "label": "this month",
                "start_day": str(today.replace(day=1)),
                "end_day": str(today),
            }
        if kind == "past7":
            return {
                "label": "the last 7 days",
                "start_day": str(today - timedelta(days=6)),
                "end_day": str(today),
            }
    return None


def _window_segment_ids(conn: sqlite3.Connection, window: dict) -> list[int]:
    """Most recent segment ids inside the window, returned oldest-first.

    Reuses the service's local-day → UTC bounds so "today" means the owner's
    calendar day, not the UTC one (a 23:30 chat belongs to that evening).
    """
    from secondbrain.query import service  # lazy: service imports search modules

    start_utc, _ = service._local_day_utc_bounds(window["start_day"])
    _, end_utc = service._local_day_utc_bounds(window["end_day"])
    rows = conn.execute(
        """
        SELECT id FROM transcript_segments
        WHERE start_at >= ? AND start_at < ?
        ORDER BY start_at DESC, id DESC
        LIMIT ?
        """,
        (start_utc, end_utc, _MAX_WINDOW_SEGMENTS),
    ).fetchall()
    return [r["id"] for r in reversed(rows)]


def _window_note(window: dict, has_segments: bool) -> str:
    span = (
        window["start_day"]
        if window["start_day"] == window["end_day"]
        else f"{window['start_day']} to {window['end_day']}"
    )
    if has_segments:
        return (
            f"The question refers to {window['label']} (local date{'s' if ' to ' in span else ''}"
            f" {span}). The excerpts below include the most recent conversations from that"
            " period — describe what was actually said there.\n\n"
        )
    return (
        f"The question refers to {window['label']} (local date{'s' if ' to ' in span else ''}"
        f" {span}), but nothing was captured in that period. Say so plainly; do not"
        " substitute other days' conversations.\n\n"
    )


# --- prompt assembly / answer resolution ---------------------------------------


@dataclass
class PreparedAsk:
    """Everything retrieval produced, frozen before the (slow) LLM call.

    ``info`` carries the full segment rows for every id the model is allowed to
    cite, so citations can be resolved after streaming without reopening the DB.
    """

    question: str
    system: str
    prompt: str
    info: dict[int, dict] = field(default_factory=dict)
    time_window: dict | None = None


def prepare(
    conn: sqlite3.Connection,
    question: str,
    *,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> PreparedAsk:
    """Retrieve context and build the prompt (everything except the LLM call)."""
    settings = settings or get_settings()

    convo, history_cited = _history_block(history)

    hits = combined.search(conn, question, limit=8, settings=settings)
    seg_ids = [h.segment_id for h in hits]

    window = _temporal_window(question)
    window_ids = _window_segment_ids(conn, window) if window else []
    seg_ids.extend(window_ids)

    seed = _seed_nodes(conn, seg_ids, question)
    facts = _subgraph_facts(conn, seed, settings)
    for f in facts:  # ensure fact-cited segments are resolvable too
        for c in json.loads(f["source_segment_ids"] or "[]"):
            seg_ids.append(c)
    seg_ids.extend(history_cited)  # …and sources carried over from prior turns
    info = _seg_info(conn, sorted(set(seg_ids)))

    def _excerpt(sid: int) -> str:
        s = info[sid]
        return f"[{sid}] {s['speaker']} ({(s['start_at'] or '')[:19]}): {s['text']}"

    ordered: list[int] = []
    seen: set[int] = set()
    for sid in seg_ids:
        if sid in info and sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    excerpts = "\n".join(_excerpt(sid) for sid in ordered)
    fact_block = "\n".join(_fact_line(f) for f in facts)

    now = datetime.now().astimezone()
    context = f"Today's date: {now.strftime('%A %Y-%m-%d')}.\n"
    if window:
        window = {**window, "segment_count": sum(1 for sid in window_ids if sid in info)}
        context += _window_note(window, window["segment_count"] > 0)
    else:
        context += "\n"
    if excerpts:
        context += f"Transcript excerpts:\n{excerpts}\n\n"
    if fact_block:
        context += f"Known facts:\n{fact_block}\n\n"
    context = context[: settings.extraction.chat_max_context_chars]

    return PreparedAsk(
        question=question,
        system=_SYSTEM,
        prompt=f"{convo}{context}Question: {question}",
        info=info,
        time_window=window,
    )


def finalize(prep: PreparedAsk, text: str) -> dict:
    """Resolve the model's citations against the prepared context."""
    cited_ids = {int(m) for m in _CITE.findall(text)}
    citations = [
        {
            "segment_id": sid,
            "conversation_id": prep.info[sid]["conversation_id"],
            "start_at": prep.info[sid]["start_at"],
            "speaker": prep.info[sid]["speaker"],
            "text": prep.info[sid]["text"],
        }
        for sid in sorted(cited_ids)
        if sid in prep.info
    ]
    return {
        "question": prep.question,
        "answer": text,
        "citations": citations,
        "general_used": _GENERAL_TAG in text,
        "grounded": bool(citations),
        "time_window": prep.time_window,
    }


def answer(
    conn: sqlite3.Connection,
    question: str,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> dict:
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    prep = prepare(conn, question, settings=settings, history=history)
    resp = llm.complete(system=prep.system, prompt=prep.prompt, max_tokens=MAX_ANSWER_TOKENS)
    return finalize(prep, resp.text)
