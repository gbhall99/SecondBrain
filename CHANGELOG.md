# Changelog

All notable changes to SecondBrain. This project is built in phases; each phase
is fully local/offline and ships behind defaults that keep earlier behaviour
unchanged.

## Phases

### Phase 1 — Ambient capture + transcription
- Always-on capture daemon (`sounddevice` ring buffer → rolling FLAC chunks),
  Silero VAD to drop silence, on-device transcription (parakeet-mlx / whisper-mlx,
  behind a mockable `Transcriber` interface), SQLite system-of-record with FTS5
  full-text and optional `sqlite-vec` semantic search.
- Durable SQLite job queue decouples cheap capture from heavy transcription.
- Consent from day one: recording indicator + pause, raw-audio retention sweep,
  disk guardrails. Menu bar app, local web UI, and `sb` CLI.

### Phase 2 — Diarization & speaker identity
- Conversation-level pyannote 3.1 diarization (mockable), global speaker registry
  with cosine matching, owner enrollment, unknown-cluster discovery, retroactive
  relabelling, and per-speaker opt-out enforced on read and write paths.

### Phase 3 — Knowledge graph + grounded Q&A
- Per-conversation local-LLM extraction (Ollama, mockable) of entities, facts,
  action items, decisions, ideas into a pure-SQLite knowledge graph with
  provenance and fact versioning; entity resolution; graph-RAG `sb ask` with
  citations.

### Phase 4 — Proactivity & goals
- Goals subsystem, deterministic detectors (commitments both directions,
  connections, goal alignment, staleness), ranking with noise control, nightly
  brief + weekly review, opt-in candid coaching.

### Phase 5 — Hardening
- Username/password auth + signed session for safe remote access (Tailscale),
  optional SQLCipher at-rest encryption, `/health` + `sb doctor`, structured
  logging, fail-closed binding.

### Phase 6 — Goal decomposition + tasks + daily planning
- LLM goal→milestone→task decomposition (propose/approve), Eisenhower quadrants
  + weighted prioritisation, capacity-fitted "Today" plan, local-first task
  research with opt-in gated web research.

### Phase 7 — Diarization & voice-profile quality
- Exemplar-aware matching, on-demand + nightly re-attribution (high bar, never
  touches confirmed labels), correction loop that feeds learning, overlap
  flagging, quality metrics.

## QA & hardening
- **QA remediation (#8):** privacy read-path filtering, `Secure` cookie + login
  rate-limiting, job-retry backoff + stuck-job reclaim, daemon heartbeat,
  merge-cycle guards, cross-phase e2e test.
- **Audit follow-ups (#9):** pinned dependency `constraints.txt`, nightly
  real-backend CI (macOS), at-rest hygiene pragmas.

## Continuous self-improvement loop
Each item shipped behind the green gate (`ruff` + full `pytest` +
`alembic upgrade head`), auto-merged on green.

- **#10** Local backup & export — consistent DB snapshot (online backup API) +
  JSON/Markdown dumps; opted-out speakers excluded.
- **#11** Data "forget" — purge person / day / range + VACUUM; FK cascades,
  FTS/vectors/raw-audio cleaned; owner protected.
- **#12** Secret-in-committed-config guard — `sb doctor` / `/health` flag secrets
  that belong in `config.local.toml` or env.
- **#13** Config validation — backend enums, thresholds, port/hour/weekday ranges
  fail fast with clear messages.
- **#14** Daemon maintenance tests — cluster/reattribute/proactive enqueue +
  date-gating + best-effort error handling.
- **#15** Database restore — reversible, validated snapshot replacement.
- **#16** Backup retention — prune to the newest N snapshots.
- **#17** Lint — enable ruff `C4`/`SIM`/`PIE`/`RET` and apply fixes.
