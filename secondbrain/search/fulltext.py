"""Full-text search over transcript segments via FTS5 (always available)."""

from __future__ import annotations

import re
import sqlite3

from secondbrain.storage.models import SearchHit

# Snippet match markers. Private-Use-Area sentinels survive HTML escaping, so a
# web client can escape the whole snippet and then swap these for <mark>…</mark>
# without ever trusting raw HTML; the CLI swaps them for [ ] instead.
MARK_START = "\ue000"
MARK_END = "\ue001"

# Snippet window width in tokens (the FTS5 snippet() budget around matches).
SNIPPET_TOKENS = 20

# Filler words dropped when relaxing a natural-language question into an OR
# query ("What was the plan for the Canada pilot?" -> "plan" OR "canada" OR
# "pilot"). Includes question/aux words plus temporal words that chat resolves
# to real date windows instead (searching for the literal word "today" only
# adds noise). Meaning-bearing words (no/not/will/should/said/say/tell/own/
# same) are deliberately NOT stripped — "Dana said no to the deal" must keep
# its negation.
_STOPWORDS = frozenset(
    "a about after again all also am an and any anything are as at be because been before "  # noqa: SIM905 - a 100-word list literal would be far noisier
    "being between both but by can could did do does doing down during each else few for "
    "from further had has have having he her here hers him his how i if in into is it its "
    "itself just like me mine more most much my myself nor now of off on once only "
    "or other our ours out over re s she so some such t than "
    "that the their theirs them then there these they this those through to too under until "
    "up us very was we were what when where which while who whom why with would you "
    "your yours "
    "today tonight yesterday tomorrow week month year recent recently lately days".split()
)

_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

# "quoted phrase" spans in the raw query (kept together as FTS5 phrase matches).
_PHRASE = re.compile(r'"([^"]*)"')


def _fts_query(raw: str) -> str:
    """Turn a user phrase into a safe FTS5 MATCH query.

    - "quoted phrases" pass through as FTS5 phrase matches (words must be
      adjacent, in order).
    - a leading ``-`` excludes a term/phrase (translated to FTS5 NOT).
    - everything else is double-quoted per token (so punctuation/operators in
      user input can't break the query) and combined with implicit AND.
    """
    include: list[str] = []
    exclude: list[str] = []

    def add(term: str, negate: bool) -> None:
        words = term.split()
        if not words:
            return
        quoted = '"' + " ".join(words) + '"'  # multi-word → FTS5 phrase
        (exclude if negate else include).append(quoted)

    pos = 0
    for m in _PHRASE.finditer(raw):
        for tok in raw[pos:m.start()].split():
            neg = tok.startswith("-")
            add(tok.lstrip("-").replace('"', " "), neg)
        neg = raw[max(m.start() - 1, 0):m.start()] == "-"
        add(m.group(1).replace('"', " "), neg)
        pos = m.end()
    for tok in raw[pos:].split():
        neg = tok.startswith("-")
        add(tok.lstrip("-").replace('"', " "), neg)

    if not include:
        return '""'  # nothing (or exclusions only) — matches no rows
    q = " ".join(include)
    for ex in exclude:
        q += f" NOT {ex}"
    return q


def _relaxed_fts_query(raw: str) -> str:
    """Recall-oriented MATCH query for natural-language questions.

    The strict query ANDs every token, so "What was the plan for the Canada
    pilot?" matches nothing unless a segment contains *all* of those words.
    Here we keep only meaningful tokens and OR them together (bm25 still ranks
    segments matching more of them higher). Falls back to OR-ing all tokens
    when the question is nothing but stopwords.
    """
    tokens = [t.lower() for t in _TOKEN.findall(raw)]
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    if not kept:
        kept = tokens
    seen: set[str] = set()
    uniq = [t for t in kept if not (t in seen or seen.add(t))]
    if not uniq:
        return '""'
    return " OR ".join(f'"{t}"' for t in uniq)


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    *,
    relaxed: bool = False,
    since_utc: str | None = None,
    until_utc: str | None = None,
    speaker_id: int | None = None,
    exclude_speaker_ids: set[int] | frozenset[int] | None = None,
) -> list[SearchHit]:
    """FTS5 search, optionally constrained by time window / speaker in SQL.

    Filters run *before* LIMIT, so a filtered page is exact — never a
    post-filter over a capped candidate pool that silently drops matches at
    corpus scale. Bounds are UTC ISO strings compared against ``start_at``
    ([since_utc, until_utc)); ``speaker_id`` must already be merge-resolved.
    ``exclude_speaker_ids`` drops those voices while keeping unattributed
    (NULL-speaker) lines.
    """
    match = _relaxed_fts_query(query) if relaxed else _fts_query(query)
    where = ["transcript_segments_fts MATCH ?"]
    params: list[object] = [MARK_START, MARK_END, match]
    if since_utc:
        where.append("s.start_at >= ?")
        params.append(since_utc)
    if until_utc:
        where.append("s.start_at < ?")
        params.append(until_utc)
    if speaker_id is not None:
        where.append("s.speaker_id = ?")
        params.append(speaker_id)
    exclude = sorted(exclude_speaker_ids or ())
    if exclude:
        ph = ",".join("?" * len(exclude))
        where.append(f"(s.speaker_id IS NULL OR s.speaker_id NOT IN ({ph}))")
        params.extend(exclude)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT s.id, s.audio_file_id, s.text, s.start_offset_s, s.end_offset_s,
               s.start_at,
               bm25(transcript_segments_fts) AS score,
               snippet(transcript_segments_fts, 0, ?, ?, ' … ', {SNIPPET_TOKENS}) AS snip
        FROM transcript_segments_fts
        JOIN transcript_segments s ON s.id = transcript_segments_fts.rowid
        WHERE {" AND ".join(where)}
        ORDER BY score
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        SearchHit(
            segment_id=r["id"],
            audio_file_id=r["audio_file_id"],
            text=r["text"],
            start_offset_s=r["start_offset_s"],
            end_offset_s=r["end_offset_s"],
            start_at=r["start_at"],
            score=float(r["score"]),  # bm25: lower is better
            snippet=r["snip"] or "",
        )
        for r in rows
    ]


def count(
    conn: sqlite3.Connection,
    query: str,
    *,
    cap: int = 1000,
    relaxed: bool = False,
    since_utc: str | None = None,
    until_utc: str | None = None,
    speaker_id: int | None = None,
    exclude_speaker_ids: set[int] | frozenset[int] | None = None,
) -> int:
    """Corpus-wide match count for ``query``, capped at ``cap``.

    Lets the UI say "Showing 50 of ~340" without ever scanning an unbounded
    match set — the inner SELECT stops at ``cap`` rows.
    """
    match = _relaxed_fts_query(query) if relaxed else _fts_query(query)
    where = ["transcript_segments_fts MATCH ?"]
    params: list[object] = [match]
    if since_utc:
        where.append("s.start_at >= ?")
        params.append(since_utc)
    if until_utc:
        where.append("s.start_at < ?")
        params.append(until_utc)
    if speaker_id is not None:
        where.append("s.speaker_id = ?")
        params.append(speaker_id)
    exclude = sorted(exclude_speaker_ids or ())
    if exclude:
        ph = ",".join("?" * len(exclude))
        where.append(f"(s.speaker_id IS NULL OR s.speaker_id NOT IN ({ph}))")
        params.extend(exclude)
    params.append(cap)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS n FROM (
            SELECT 1 FROM transcript_segments_fts
            JOIN transcript_segments s ON s.id = transcript_segments_fts.rowid
            WHERE {" AND ".join(where)}
            LIMIT ?
        )
        """,
        params,
    ).fetchone()
    return int(row["n"])
