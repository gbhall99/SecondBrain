"""Nightly clustering of unknown speakers.

Recurring unknown voices show up as separate provisional "Unknown #N" speakers
(one per conversation where they first appear). This job groups unknown-speaker
centroids by acoustic similarity (single-linkage / connected components under a
cosine-distance threshold) and merges each group into one canonical speaker —
turning "a different stranger every day" into "this person you've spoken to N
times". Pure-Python; no sklearn dependency, so it runs the same on CI.
"""

from __future__ import annotations

import logging
import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.speaker import registry
from secondbrain.storage import state
from secondbrain.storage.models import utcnow_iso

log = logging.getLogger("secondbrain.cluster")

LAST_RUN_KEY = "last_cluster_run"


def _unknown_speakers(conn: sqlite3.Connection) -> list[tuple[int, list[float]]]:
    rows = conn.execute(
        "SELECT id, centroid FROM speakers "
        "WHERE kind='unknown' AND merged_into IS NULL AND centroid IS NOT NULL"
    ).fetchall()
    out = []
    for r in rows:
        vec = registry.deserialize_embedding(r["centroid"])
        if vec:
            out.append((int(r["id"]), vec))
    return out


def _connected_components(
    items: list[tuple[int, list[float]]], distance_threshold: float
) -> list[list[int]]:
    """Single-linkage clusters: ids connected if cosine distance < threshold."""
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            dist = 1.0 - registry.cosine(items[i][1], items[j][1])
            if dist < distance_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(items[idx][0])
    return [g for g in groups.values() if len(g) > 1]


def _cohesive(group: list[int], vec_by_id: dict[int, list[float]], threshold: float) -> bool:
    """True when every pair in the group is within the distance threshold.

    Single-linkage happily chains A~B~C into one group even when A and C are
    different people (each link is close, the ends are not). Requiring
    complete-linkage cohesion before merging keeps a chain of two distinct
    voices from collapsing into one person.
    """
    vecs = [vec_by_id[i] for i in group]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if 1.0 - registry.cosine(vecs[i], vecs[j]) >= threshold:
                return False
    return True


def run_clustering(conn: sqlite3.Connection, settings: Settings | None = None) -> int:
    """Merge similar unknown speakers. Returns the number of merges performed."""
    settings = settings or get_settings()
    threshold = settings.diarization.cluster_distance_threshold
    items = _unknown_speakers(conn)
    vec_by_id = dict(items)
    merges = 0
    for group in _connected_components(items, threshold):
        if not _cohesive(group, vec_by_id, threshold):
            log.warning(
                "skipping non-cohesive unknown-speaker group %s "
                "(chained links between distinct voices)",
                sorted(group),
            )
            continue
        canonical = min(group)  # stable, deterministic
        for other in group:
            if other != canonical:
                registry.merge_speakers(conn, other, canonical, settings)
                merges += 1
    state.set_state(conn, LAST_RUN_KEY, utcnow_iso())
    return merges
