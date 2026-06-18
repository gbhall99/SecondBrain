# SecondBrain — Roadmap

The long-term vision: an always-on, **fully local** second brain that records your
day, builds a personal knowledge graph (people, projects, ideas, decisions,
commitments, goals), and **proactively** helps you be a better team member —
surfacing connections, tracking commitments, and answering questions grounded in
everything it has heard. Everything runs on your Mac mini; nothing leaves it.

We build it in value-delivering phases.

## ✅ Phase 0–1 — Capture + transcript (this release)
Always-on recorder → VAD → on-device transcription → searchable transcript log,
with consent controls (recording indicator, pause, raw-audio retention, disk
guardrails) and three interfaces (CLI, local web UI, menu bar).

**Outcome:** a private, searchable record of everything said in your day.

## ✅ Phase 2 — Diarization & speaker identity (shipped)
- `pyannote.audio` 3.1 **conversation-level** diarization; align speaker turns
  onto transcript segments by max overlap (populate `transcript_segments.speaker_id`
  + `speaker_confidence`).
- Owner enrollment (guided recording **and** label-from-history); speaker
  embeddings stored as BLOBs, matched with cosine to a global registry.
- Auto-match recurring voices above a threshold; nightly clustering of unknown
  speakers; a web "Who is this?" labeling queue with sample-clip playback and
  retroactive relabeling; `sb speaker` CLI.
- Enforce per-speaker opt-out (redaction) from `[consent].speaker_opt_out`.
- Raw-audio retention deferred until a conversation is diarized.

**Outcome:** transcripts attributed to named people; voices learned over time.

## ✅ Phase 3 — Knowledge extraction & graph (local LLM) (shipped)
- Per-conversation extraction (after diarization) via a local LLM (**Ollama**,
  schema-constrained JSON), behind an interface with a deterministic mock for CI.
- Structured extraction: people, projects, orgs, topics, facts about people,
  action items/commitments, decisions, ideas — each with `source_segment_ids`
  provenance + confidence; low-confidence speaker attributions are downgraded to
  'mention' (never asserted as hard facts).
- Entity resolution (normalized-name + embedding cosine + LLM disambiguation,
  reusing the speaker-registry helpers); Person nodes linked to `speakers`.
- **Pure-SQLite** graph (`kg_nodes`/`kg_aliases`/`kg_edges`) with provenance and
  **fact versioning** (superseded, not overwritten); merge mirrors speaker merge.
- Graph-RAG chat: `sb ask` + web `/chat` answer **grounded with citations**, and
  may add clearly-labeled general knowledge ("grounded + general"); web `/graph`
  browser. Opted-out/redacted speech never enters the graph.

**Outcome:** the actual "second brain" — ask grounded questions; browse your world.

## ✅ Phase 4 — Proactivity & goals (shipped)
- Goals subsystem (CRUD + auto-linking to the graph by embedding/keyword);
  goal-aware engine.
- Deterministic detectors (commitments both directions, connections, goal
  alignment, staleness) + opt-in **candid** LLM coaching; the LLM only synthesizes
  the brief prose.
- Nightly **morning brief** + **weekly review**; sparse real-time nudges for
  urgent commitments.
- Noise control: importance ranking, daily/per-kind caps, confidence floor,
  snooze-a-kind, thumbs up/down with local per-kind weighting, cross-day dedupe.
- Surfaces: web `/brief` + `/goals`, menu-bar digest count, `sb digest`/`sb goals`.

**Outcome:** a morning brief and proactive nudges that make you more effective.

## Phase 5 — Hardening & remote access (next)
- Better overlap handling; profile-quality improvements; backup/export.
- **Tailscale-secured** remote access to the local web UI as the
  privacy-preserving "from anywhere" path. (Native Telegram chat would route data
  through Telegram's cloud, breaking the local-only guarantee — explicit opt-in
  only.)

## Cross-cutting principles
- **Local-first / offline** at every phase.
- **Provenance & confidence** on every fact, so we can re-extract as models
  improve and always cite sources.
- **Pluggable, testable backends** — Apple/ML-specific code behind interfaces
  with CI mocks.
- **Consent by design** — visible indicator, instant pause, short raw-audio
  retention, per-speaker opt-out.
