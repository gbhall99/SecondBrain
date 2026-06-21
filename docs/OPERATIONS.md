# Operations, Security Caveats & Tuning

## Health & troubleshooting
- `sb doctor` runs preflight checks (migrations, disk, DB, Ollama, encryption,
  **microphone/input device**, recording, **daemon heartbeat**). `GET /health`
  returns the same as JSON. The `microphone` check flags a missing input device or
  a configured device that isn't found — the usual sign that capture is silently
  failing because no mic is available or macOS Microphone permission was denied.
- **Stuck pipeline?** Check `sb status` (queue depth) and `sb doctor`. The daemon
  writes `heartbeat:worker` / `heartbeat:maintenance` to `app_state`; if `/health`
  shows the `daemon` check **stale**, a loop has died — restart the daemon
  (launchd will relaunch it). Jobs stuck in `running` (worker died mid-job) are
  auto-reclaimed to `pending` after 30 min by the maintenance loop; failed jobs
  retry with exponential backoff and dead-letter after `max_attempts`.
- To process one job in the foreground with a full traceback: `sb drain --max-jobs 1`.

## Security / privacy caveats
- **At-rest encryption scope.** `[security].encrypt_db` (SQLCipher) encrypts the
  SQLite database, **but not**: the WAL/`-shm` sidecar files, the sqlite-vec data,
  or the **raw audio FLAC files on disk**. For true at-rest protection use
  **macOS FileVault** (full-disk) — that is the recommended control; SQLCipher is
  defense-in-depth for the structured data.
- **Remote access.** Auth is off by default and only enforced for non-loopback
  clients. The server **refuses to bind off-loopback without `require_auth` + a
  password**. Prefer `tailscale serve https / 127.0.0.1:<port>` (TLS, tailnet
  only); the session cookie is marked `Secure` when served over HTTPS. Login is
  rate-limited (5 failures / 5 min / IP). Keep `session_max_age_days` short for
  remote use.
- **Opt-out / redaction** is enforced on every read path (search, `/day`, chat,
  graph) in addition to the write-time redaction — opted-out speakers' words do
  not surface in answers or search.
- **Secrets** (`db_passphrase`, `hf_token`) belong in `config.local.toml`
  (gitignored) or env vars (`SB_SECURITY__DB_PASSPHRASE`, `HF_TOKEN`) — never in
  the committed `config.toml`.

## Reproducible installs
Dependencies are floating in `pyproject.toml`. For reproducible CI/deploys, pin
transitive versions with a lockfile (e.g. `uv pip compile pyproject.toml -o
requirements.lock`) and install from it.

## Key tuning thresholds (`config.toml`)
| Setting | Meaning |
|---|---|
| `diarization.match_threshold` (0.70) | cosine sim to auto-label a known voice |
| `diarization.owner_match_threshold` (0.65) | owner checked first, looser |
| `diarization.exemplar_k` (3) | match vs k nearest stored voice samples |
| `diarization.reattribute_threshold` (0.80) | HIGH bar to relabel past low-conf lines |
| `diarization.cluster_distance_threshold` (0.30) | nightly unknown-voice merge distance |
| `diarization.low_confidence_threshold` (0.5) | below ⇒ a label is flagged/excluded from facts |
| `extraction.entity_match_threshold` (0.82) | auto-link an extracted entity to a node |
| `proactive.confidence_floor` (0.4) | drop weak suggestions |
| `tasks.daily_capacity_minutes` (240) | capacity the day-planner fits tasks into |

Relationships to keep consistent: `reattribute_threshold` > `match_threshold` >
`owner_match_threshold` > `low_confidence_threshold`; raise thresholds to reduce
false positives, lower them to surface more (and review via the `/day` fix-speaker
and `/speakers` queues).
