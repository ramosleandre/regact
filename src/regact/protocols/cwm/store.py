"""Deduplicated real experience plus ordered occurrences; one trusted DB per task."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, cast


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _link(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def info_trace(db: sqlite3.Connection) -> list[tuple[int, dict[str, Any]]]:
    """``(real actions so far, observation info)`` after every real step of the task, in order.
    Explicit resets count as one action each, as in the task's action budget."""
    rows = db.execute(
        """SELECT json_extract(o.payload,'$.info') info,
            (SELECT COUNT(*) FROM episodes e WHERE e.id<=s.episode_id
                AND e.purpose LIKE 'reset\\_%' ESCAPE '\\') resets
        FROM step_events s JOIN transitions t ON t.id=s.transition_id
        JOIN observations o ON o.id=t.after_id ORDER BY s.id"""
    ).fetchall()
    return [(index + 1 + row[1], json.loads(row[0] or "{}")) for index, row in enumerate(rows)]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class ExperienceStore:
    """Calls are serialized by the task coordinator, including real action execution.

    DELETE journaling avoids WAL's shared-memory filesystem assumptions. Use a
    local filesystem with working SQLite locks. No pruning of validation evidence.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY, hash TEXT UNIQUE NOT
                NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS actions(id INTEGER PRIMARY KEY, hash TEXT UNIQUE NOT
                NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS transitions(id INTEGER PRIMARY KEY, before_id INTEGER
                REFERENCES observations(id), action_id INTEGER REFERENCES actions(id), after_id
                INTEGER REFERENCES observations(id), UNIQUE(before_id,action_id,after_id));
            CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY, initial_obs_id INTEGER
                REFERENCES observations(id), purpose TEXT, metadata TEXT, status TEXT,
                stop_reason TEXT, result TEXT, start_hash TEXT);
            CREATE TABLE IF NOT EXISTS step_events(id INTEGER PRIMARY KEY, episode_id INTEGER
                REFERENCES episodes(id), step_index INTEGER, transition_id INTEGER REFERENCES
                transitions(id), prev_hash TEXT, history_hash TEXT,
                UNIQUE(episode_id,step_index));
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, timestamp REAL, kind TEXT,
                phase TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
                status TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS diagnostics(id INTEGER PRIMARY KEY, payload TEXT NOT
                NULL);
            CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO meta VALUES('dataset_version','0');
            CREATE INDEX IF NOT EXISTS event_episode ON step_events(episode_id,step_index);
            CREATE INDEX IF NOT EXISTS event_history ON step_events(prev_hash);
        """)

    def _intern(self, table: str, value: Any) -> int:
        payload = canonical(value)
        h = hashlib.sha256(payload.encode()).hexdigest()
        self.db.execute(f"INSERT OR IGNORE INTO {table}(hash,payload) VALUES(?,?)", (h, payload))
        row = self.db.execute(f"SELECT id,payload FROM {table} WHERE hash=?", (h,)).fetchone()
        if row["payload"] != payload:
            raise RuntimeError("canonical hash collision")
        return int(row["id"])

    @property
    def version(self) -> int:
        return int(
            self.db.execute("SELECT value FROM meta WHERE key='dataset_version'").fetchone()[0]
        )

    def _bump(self) -> None:
        self.db.execute(
            "UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='dataset_version'"
        )

    def _hash_of(self, table: str, row_id: int) -> str:
        return str(self.db.execute(f"SELECT hash FROM {table} WHERE id=?", (row_id,)).fetchone()[0])

    def last_hash(self, episode: int) -> str:
        """The history hash after the episode's latest step: its first observation and every
        action and observation since."""
        row = self.db.execute(
            "SELECT history_hash FROM step_events WHERE episode_id=? "
            "ORDER BY step_index DESC LIMIT 1",
            (episode,),
        ).fetchone()
        if row is not None:
            return str(row[0])
        return str(
            self.db.execute("SELECT start_hash FROM episodes WHERE id=?", (episode,)).fetchone()[0]
        )

    def start_episode(
        self, obs: dict[str, Any], purpose: str, metadata: dict[str, Any]
    ) -> tuple[int, int]:
        with self.db:
            oid = self._intern("observations", obs)
            start = _link("start", self._hash_of("observations", oid))
            cursor = self.db.execute(
                (
                    "INSERT INTO episodes(initial_obs_id,purpose,metadata,status,start_hash) "
                    "VALUES(?,?,?,'running',?)"
                ),
                (oid, purpose, canonical(metadata), start),
            )
            episode = int(cursor.lastrowid or 0)
            self._bump()  # resets, including repeated ones, are part of the evidence boundary
            return episode, oid

    def record_step(
        self, episode: int, before: dict[str, Any], action: Any, after: dict[str, Any]
    ) -> dict[str, Any]:
        with self.db:
            b, a, n = (
                self._intern("observations", before),
                self._intern("actions", action),
                self._intern("observations", after),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO transitions(before_id,action_id,after_id) VALUES(?,?,?)",
                (b, a, n),
            )
            tid = int(
                self.db.execute(
                    "SELECT id FROM transitions WHERE before_id=? AND action_id=? AND after_id=?",
                    (b, a, n),
                ).fetchone()[0]
            )
            step = int(
                self.db.execute(
                    "SELECT COALESCE(MAX(step_index)+1,0) FROM step_events WHERE episode_id=?",
                    (episode,),
                ).fetchone()[0]
            )
            prev = self.last_hash(episode)
            history_hash = _link(
                prev, self._hash_of("actions", a), self._hash_of("observations", n)
            )
            # The same episode history and action that once led elsewhere: the game is random,
            # or kept something a reset does not show. (The same screen leading elsewhere is
            # expected when the game has hidden state.)
            witnesses = [
                dict(row)
                for row in self.db.execute(
                    """SELECT s.id event_id,s.episode_id,t.id transition_id,t.after_id
                FROM step_events s JOIN transitions t ON s.transition_id=t.id
                WHERE s.prev_hash=? AND t.action_id=? AND t.after_id<>? ORDER BY s.id LIMIT 1""",
                    (prev, a, n),
                )
            ]
            cur = self.db.execute(
                "INSERT INTO step_events"
                "(episode_id,step_index,transition_id,prev_hash,history_hash) VALUES(?,?,?,?,?)",
                (episode, step, tid, prev, history_hash),
            )
            self._bump()
            return {
                "event_id": cur.lastrowid,
                "episode_id": episode,
                "step_index": step,
                "transition_id": tid,
                "before_obs_id": b,
                "after_obs_id": n,
                "history_hash": history_hash,
                "conflicting_witnesses": witnesses,
            }

    def finish_episode(
        self, episode: int, reason: str, result: dict[str, Any], status: str = "completed"
    ) -> None:
        with self.db:
            self.db.execute(
                "UPDATE episodes SET status=?,stop_reason=?,result=? WHERE id=? AND status='running'",
                (status, reason, canonical(result), episode),
            )

    def summary(self) -> dict[str, Any]:
        summary = {
            "dataset_version": self.version,
            **{
                key: int(self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for key, table in (
                    ("n_unique_observations", "observations"),
                    ("n_unique_transitions", "transitions"),
                    ("n_total_transitions", "step_events"),
                    ("n_started_episodes", "episodes"),
                )
            },
        }
        # One initial observation per episode, then one successor per recorded step.
        summary["n_total_observations"] = (
            summary["n_started_episodes"] + summary["n_total_transitions"]
        )
        return summary

    def observation(self, oid: int) -> dict[str, Any]:
        row = self.db.execute("SELECT payload FROM observations WHERE id=?", (oid,)).fetchone()
        if row is None:
            raise ValueError(f"unknown observation ID {oid}")
        return cast(dict[str, Any], json.loads(row[0]))

    def observation_ids(self) -> list[int]:
        return [int(row[0]) for row in self.db.execute("SELECT id FROM observations ORDER BY id")]

    def observation_hashes(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT hash FROM observations")}

    def transition(self, tid: int) -> dict[str, Any]:
        row = self.db.execute(
            (
                "SELECT t.*,a.payload action FROM transitions t JOIN actions a ON "
                "a.id=t.action_id WHERE t.id=?"
            ),
            (tid,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown transition ID {tid}")
        return {
            "transition_id": tid,
            "before_obs_id": row["before_id"],
            "after_obs_id": row["after_id"],
            "o": self.observation(row["before_id"]),
            "action": json.loads(row["action"]),
            "o_next": self.observation(row["after_id"]),
        }

    def transition_ids(self, after_id: int = 0, limit: int | None = None) -> list[int]:
        sql = "SELECT id FROM transitions WHERE id>? ORDER BY id"
        args: tuple[Any, ...] = (after_id,)
        if limit is not None:
            sql += " LIMIT ?"
            args += (limit,)
        return [int(row[0]) for row in self.db.execute(sql, args)]

    def episode_ids(self, episode_id: int) -> tuple[list[int], list[int]]:
        """Chronological observation/transition occurrences, including the reset."""
        row = self.db.execute(
            "SELECT initial_obs_id FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown episode ID {episode_id}")
        rows = self.db.execute(
            "SELECT s.transition_id,t.after_id FROM step_events s JOIN transitions t "
            "ON t.id=s.transition_id WHERE s.episode_id=? ORDER BY s.step_index",
            (episode_id,),
        ).fetchall()
        return [int(row[0]), *[int(item[1]) for item in rows]], [int(item[0]) for item in rows]

    def episodes(self) -> list[dict[str, Any]]:
        """Every episode in start order, with its number of recorded steps."""
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT e.id episode_id,e.purpose,e.initial_obs_id,"
                "e.start_hash,(SELECT COUNT(*) FROM step_events s WHERE s.episode_id=e.id) n_steps "
                "FROM episodes e ORDER BY e.id"
            )
        ]

    def episode_steps(self, episode_id: int) -> list[dict[str, Any]]:
        """The episode's recorded steps in time order: action, resulting observation, hashes."""
        return [
            {**dict(row), "action": json.loads(row["action"])}
            for row in self.db.execute(
                "SELECT s.step_index,s.transition_id,t.after_id after_obs_id,s.history_hash,"
                "a.payload action FROM step_events s JOIN transitions t ON t.id=s.transition_id "
                "JOIN actions a ON a.id=t.action_id WHERE s.episode_id=? ORDER BY s.step_index",
                (episode_id,),
            )
        ]

    def repeated_histories(self) -> set[str]:
        """History hashes reached more than once: the only points where a replay can reuse work."""
        return {
            str(row[0])
            for row in self.db.execute(
                "SELECT h FROM (SELECT history_hash h FROM step_events UNION ALL "
                "SELECT start_hash FROM episodes) GROUP BY h HAVING COUNT(*)>1"
            )
        }

    def event(self, kind: str, phase: str, payload: dict[str, Any]) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO events(timestamp,kind,phase,payload) VALUES(?,?,?,?)",
                (time.time(), kind, phase, canonical(payload)),
            )
            return int(cur.lastrowid or 0)

    def record(self, kind: str, status: str, payload: dict[str, Any]) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO records(kind,status,payload) VALUES(?,?,?)",
                (kind, status, canonical({"created_at": time.time(), **payload})),
            )
            return int(cur.lastrowid or 0)

    def record_payload(self, rid: int) -> dict[str, Any]:
        row = self.db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
        if row is None:
            raise ValueError(f"unknown record {rid}")
        return cast(dict[str, Any], json.loads(row[0]))

    def update_record(self, rid: int, status: str, payload: dict[str, Any]) -> None:
        payload = {**self.record_payload(rid), **payload}
        if status != "running":
            payload["finished_at"] = time.time()
        with self.db:
            self.db.execute(
                "UPDATE records SET status=?,payload=? WHERE id=?",
                (status, canonical(payload), rid),
            )

    def diagnostic(self, payload: dict[str, Any]) -> int:
        with self.db:
            return int(
                self.db.execute(
                    "INSERT INTO diagnostics(payload) VALUES(?)", (canonical(payload),)
                ).lastrowid
                or 0
            )

    def get_diagnostic(self, did: int) -> dict[str, Any]:
        row = self.db.execute("SELECT payload FROM diagnostics WHERE id=?", (did,)).fetchone()
        if row is None:
            raise ValueError(f"unknown diagnostic ID {did}")
        return cast(dict[str, Any], json.loads(row[0]))

    def request_result(self, rid: str, payload: Any) -> dict[str, Any] | None:
        row = self.db.execute("SELECT payload,result FROM requests WHERE id=?", (rid,)).fetchone()
        if row is None:
            return None
        if row[0] != canonical(payload):
            raise ValueError("request ID reused with a different operation/payload")
        return cast(dict[str, Any], json.loads(row[1]))

    def remember_request(self, rid: str, payload: Any, result: Any) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO requests VALUES(?,?,?)", (rid, canonical(payload), canonical(result))
            )

    def close(self) -> None:
        self.db.close()
