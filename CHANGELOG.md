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
- **#18** CHANGELOG / loop-log.
- **#19** `sb backups` — list snapshots.
- **#20** `sb stats` — corpus overview (CLI).
- **#21** Forget — prune knowledge-graph edge citations; drop ungrounded edges.
- **#22** Forget — purge extraction provenance of fully-forgotten conversations.
- **#23** `sb config show` / `check` — effective config with secrets redacted.
- **#24** `/api/stats` — web parity for the corpus overview.
- **#25** `sb queue` — job-queue inspection + stuck-job reclaim.
- **#26** Export — optional `--since`/`--until` date range.
- **#27** Search — optional `--since`/`--until` date range.
- **#28** Backup-freshness check in `sb doctor` / `/health`.

## Phase 8 — People & Memory Intelligence
Larger net-new features turning the captured graph into relationship & memory
intelligence. All local, default-safe, opt-out-filtered, green-gated.

- **8A Person dossier (#29, #30):** `service.person_dossier` aggregates identity,
  interactions/talk-time, known facts, commitments (owed by/to), recent quotes,
  and connections; `GET /api/person/{id}`, `/person/{id}` page, `sb person`.
  Opted-out people show identity/interaction shape only.
- **8B Relationship intelligence (#31, #32):** `service.relationships` ranks
  people by interaction; `detect_stale_relationships` reconnect nudge
  (`reconnect_days`, gated); `/api/relationships`, `/relationships`, `sb relationships`.
- **8C Memory timeline (#33, #34):** `service.timeline` renders a day as
  conversations with attributed segments + inline extracted knowledge;
  `/api/timeline/{day}`, `/timeline[/{day}]`, `sb timeline`.
- **8D Unified dashboard (#35):** shared `base.html` nav; `index.html` reworked as
  a home dashboard with corpus stats deep-linking to every section; person/
  relationships/timeline pages share the nav.

## Mac deploy automation
One-command, no-hand-editing deployment for an always-on Mac mini.
- launchd templates for the **web** and **menu bar** agents (alongside the existing
  daemon); `python -m secondbrain` entrypoint; `sb deploy launchd [--load
  --include-menubar --unload]` fills the templates with this venv's Python + repo
  path, installs to `~/Library/LaunchAgents`, and (un)loads via `launchctl` after a
  `sb doctor` preflight (blocks only on migrations/disk/database).
- `deploy/install.sh` idempotent bootstrap; `docs/DEPLOY.md` canonical guide
  (install, config, Ollama/pyannote, Tailscale, verify, troubleshooting).

## Phase 9 — Project Intelligence
Project surfaces mirroring the Phase 8 people pattern (projects are first-class KG
nodes). No migration; local, opt-out-filtered, green-gated.
- **9A service:** `service.list_projects` ranks projects by activity (conversations
  then mention volume, with linked-goal and open-action-item counts);
  `service.project_dossier` aggregates identity/aliases, activity, linked goals,
  associated people (opt-out filtered), decisions, facts, open commitments, and
  recent cited quotes. `corpus_stats` gains a `projects` count.
- **9B API:** `GET /api/projects`, `GET /api/project/{node_id}` (404 if unknown).
- **9C web:** `projects.html` (ranked list) + `project.html` (dossier) extending
  `base.html`; **Projects** added to the nav and a projects deep-link on the home
  dashboard.
- **9D CLI:** `sb projects`, `sb project <node_id>`.

## Deployment hardening
Closes the gaps that could make a fresh Mac mini look "broken" silently.
- **Microphone health check:** `sb doctor` / `/health` now include a `microphone`
  check (best-effort, degradable) that flags a missing input device or a configured
  device that can't be found — surfacing the case where capture silently
  crash-loops because no mic is available or macOS Microphone permission was denied.
- **Semantic search out of the box:** added `sentence-transformers` to the `ml`
  extra so semantic search activates on install instead of silently falling back to
  full-text only.
- **First-run UX:** added `config.local.toml.example` (seeded by `install.sh`); and
  `docs/DEPLOY.md` now calls out the Microphone-permission step, the one-time
  first-run model downloads, and an on/off-by-default feature matrix.

## AI features on by default (turnkey local setup)
The seeded `config.local.toml` now enables diarization, knowledge extraction, and
the proactive brief (with `[llm].backend = "ollama"`), and `deploy/install.sh` sets
up their prerequisites so first run is turnkey:
- Installs Ollama (Homebrew), starts it as a service, and pulls the model
  (`SB_OLLAMA_MODEL`, default `llama3.1:8b-instruct`).
- Prompts for a HuggingFace token, writes it into `config.local.toml`, and runs
  `sb speaker setup` to fetch the gated pyannote models.
- `SB_SKIP_AI=1 ./deploy/install.sh` opts out (capture-only); committed defaults
  (`config.toml`) and CI are unchanged — these flags live only in the gitignored
  local override, so fresh clones stay capture-only and green.
- DEPLOY.md documents the diarization → extraction → proactive chain (a HF token is
  required for the knowledge graph to populate).

## Deploy hardening pass
Fixes found by re-auditing the install.sh AI block, plus guardrails:
- **Ollama start/pull race fixed:** wait (up to ~30s) for the server to accept
  connections before `ollama pull`; clean `brew services` vs `ollama serve` start
  (no more backgrounding the whole `||` list); idempotent — skips the HF-token
  prompt when one is already set.
- **Tested token write:** new `sb config set-hf-token` (backed by
  `secondbrain/config_edit.py`, `json`-escaped, re-parseable) replaces the fragile
  inline string substitution in install.sh.
- **Tests:** `config_edit` (placeholder present/absent/append, special chars,
  file round-trip) and a check that `config.local.toml.example` is valid TOML →
  valid `Settings` with the AI flags on.
- **CI:** added a `shellcheck` job for `deploy/*.sh`.
- DEPLOY.md: explicit microphone-check (TCC) caveat + an on-device verification
  checklist, since the macOS-only paths can't be exercised in CI.

## Pipeline reliability hardening
Queue, worker, capture, daemon, backup, retention, and repair all get sturdier so
transient failures degrade gracefully instead of silently losing work.
- **Queue:** `sb queue --retry-failed` re-queues dead-lettered jobs
  (`queue.requeue_failed`); longer backoff schedule (1m → 5m → 30m) and extra
  attempts for transient-prone job types (diarize/extract) so an Ollama/HF outage
  doesn't dead-letter meetings; `prune_done_jobs` sweeps old completed rows from
  the daemon (new `jobs(state, finished_at)` index, migration `0008_reliability`);
  transcribe jobs enqueue at priority 10 so a slow diarization can't starve live
  transcription.
- **Worker:** transcription is idempotent (an existing transcript short-circuits a
  retried job) and transactional (transcript + segments + status commit atomically);
  a VAD failure falls through to transcription instead of failing the chunk; a
  vanished audio file is marked `missing` cleanly; VAD speech totals persist to
  `audio_files.speech_seconds` with a configurable
  `[transcription].min_speech_seconds` gate (default 0.0 = old behavior); job
  failures log the full traceback, not just `repr(exc)`.
- **Capture:** a configured input device that can't be found now *refuses* to
  record (never silently falls back to the default mic), raising an
  `alarm:input_device` app_state alarm surfaced by health checks; pause takes
  effect within ~1s mid-chunk (partial buffers are discarded); input-stream
  overflows are counted and stored (`overflow_count`); per-chunk RMS is stored
  (`rms_level`) for dead/muted-mic detection; blocked-capture reasons are logged
  (rate-limited to once/minute/reason); the recorder writes a `heartbeat:capture`.
- **Daemon:** a watchdog restarts any dead capture/worker/maintenance thread with
  a growing backoff; scheduled daily DB backup + prune (`[backup] auto_enabled`,
  `keep`); conversation-stale-closing now runs on a ~60s cadence, separate from
  the hourly retention sweep.
- **Health/doctor:** checks carry a severity (`error`/`warn`); advisory checks
  (stale backups, unreachable LLM, backlog, failed jobs) are warnings and no
  longer fail `sb doctor` (exit 1 only on errors); new checks for capture
  freshness, failed-job count, queue backlog/oldest-pending age, tiered heartbeat
  staleness (warn 15m / error 2h, now including capture), the input-device alarm,
  and near-zero-RMS mic detection; "no backups yet" degrades to a warning once
  transcripts are older than a week.
- **Backups:** snapshots are verified with `PRAGMA quick_check` after writing (a
  corrupt snapshot is deleted and raises); restore writes to a temp file next to
  the live DB and `os.replace()`s it into place (atomic); `prune_backups` never
  deletes `*-pre-restore.db` safety snapshots.
- **Retention:** raw audio for dead-lettered chunks is eventually swept ('failed'
  past retention + grace; 'recorded' past a generous 7-day grace); stale
  diarization scratch files (`conv_concat*.wav` older than a day) are cleaned up.
- **Repair:** re-enqueues 'recorded' chunks with no live job and conversations
  stuck in 'closed'/'diarizing' with no live diarize job; `config.local.toml`
  seeding resolves against the repo root (not the CWD); the corruption message
  names the newest backup and its age.
