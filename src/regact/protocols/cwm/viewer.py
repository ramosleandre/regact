"""Read-only CWM inspection and bounded, on-demand prediction replay; no stored videos."""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

from regact.envclient.obs import Obs
from regact.problems.base import BaseProblem, build_problem
from regact.protocols.cwm.bundle import verify_bundle
from regact.protocols.cwm.config import ExecutionConfig
from regact.protocols.cwm.store import canonical, digest
from regact.protocols.cwm.validation import check_observation
from regact.protocols.cwm.worker import Worker


def png(problem: BaseProblem, obs: dict[str, Any]) -> bytes:
    from PIL import Image

    frame = problem.render_frame(Obs.from_json(obs))
    if frame is None:
        raise ValueError("this problem has no image renderer")
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


@contextmanager
def database(task: Path) -> Iterator[sqlite3.Connection]:
    path = task / "cwm" / "experience.sqlite3"
    if not path.is_file():
        raise ValueError("no CWM experience database in this task")
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def info_trace(task: Path) -> list[tuple[int, dict[str, Any]]]:
    """``(real actions so far, observation info)`` after every real step of the task, in order.
    Explicit resets count as one action each, as in the task's action budget."""
    with database(task) as db:
        rows = db.execute(
            """SELECT json_extract(o.payload,'$.info') info,
                (SELECT COUNT(*) FROM episodes e WHERE e.id<=s.episode_id
                    AND e.purpose LIKE 'reset\\_%' ESCAPE '\\') resets
            FROM step_events s JOIN transitions t ON t.id=s.transition_id
            JOIN observations o ON o.id=t.after_id ORDER BY s.id"""
        ).fetchall()
    return [
        (index + 1 + row["resets"], json.loads(row["info"] or "{}"))
        for index, row in enumerate(rows)
    ]


def _obs(db: sqlite3.Connection, oid: int) -> dict[str, Any]:
    row = db.execute("SELECT payload FROM observations WHERE id=?", (oid,)).fetchone()
    if row is None:
        raise ValueError("unknown observation")
    return cast(dict[str, Any], json.loads(row[0]))


def inspect(task: Path, *, before: int = 0, limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 100 or before < 0:
        raise ValueError("invalid page bounds")
    status = json.loads((task / "cwm" / "status.json").read_text())
    for new, old in (
        ("n_unique_observations", "n_observations"),
        ("n_unique_transitions", "n_transitions"),
        ("n_total_transitions", "n_step_events"),
    ):
        if new not in status:
            status[new] = status.get(old, 0)

    def names(value: Any) -> Any:
        if isinstance(value, list):
            return [names(item) for item in value]
        if isinstance(value, dict):
            aliases = {
                "dataset_revision": "dataset_version",
                "cwm_revision": "cwm_version",
                "dream_actions": "simulation_actions",
                "dream_action_sequence": "simulation_action_sequence",
                "dream_prediction_hashes": "simulation_prediction_hashes",
            }
            return {aliases.get(k, k): names(v) for k, v in value.items()}
        return value

    status = names(status)
    status["phase"] = {0: "Dataset preparation", 1: "CWM Modeling", 2: "Active Exploration"}.get(
        status.get("phase"), status.get("phase")
    )
    with database(task) as db:
        rows = db.execute(
            "SELECT * FROM records WHERE (?=0 OR id<?) ORDER BY id DESC LIMIT ?",
            (before, before, limit),
        ).fetchall()
        records = [{**dict(row), "payload": names(json.loads(row["payload"]))} for row in rows]
        for record in records:
            record["payload"].pop("manifest", None)
        episodes = [
            dict(row)
            for row in db.execute(
                "SELECT id,purpose,status,stop_reason,initial_obs_id FROM episodes ORDER BY id"
            )
        ]
        events = [
            {**dict(row), "payload": json.loads(row["payload"])}
            for row in db.execute(
                "SELECT * FROM events WHERE kind IN ('phase_changed','cwm_accepted"
                "','observation_determinism_violation','cwm_run_finished') ORDER B"
                "Y id"
            )
        ]
    return {
        "status": status,
        "records": records,
        "next_before": rows[-1]["id"] if len(rows) == limit else None,
        "episodes": episodes,
        "phase_events": events,
    }


def evidence(task: Path, kind: str, identifier: int) -> Any:
    with database(task) as db:
        if kind == "observation":
            return _obs(db, identifier)
        if kind == "diagnostic":
            row = db.execute("SELECT payload FROM diagnostics WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("unknown diagnostic")
            return cast(dict[str, Any], json.loads(row[0]))
        if kind == "transition":
            row = db.execute(
                (
                    "SELECT t.*, a.payload action FROM transitions t JOIN actions a ON"
                    " a.id=t.action_id WHERE t.id=?"
                ),
                (identifier,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown transition")
            return {
                **dict(row),
                "action": json.loads(row["action"]),
                "o": _obs(db, row["before_id"]),
                "o_next": _obs(db, row["after_id"]),
            }
        raise ValueError("unknown evidence kind")


class Playback:
    """One viewer cache, at most 4 plans / 64 MiB each, with serialized replay."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.plans: OrderedDict[tuple[str, str, int], list[dict[str, Any]]] = OrderedDict()

    def load(self, task: Path, kind: str, identifier: int) -> dict[str, Any]:
        if kind == "controller":
            with database(task) as db:
                payload = self._controller_record(db, identifier)
                return {"kind": "real", "frames": len(payload["observation_sequence"]),
                        "source": "recorded observation IDs for this controller call"}
        if kind == "episode":
            with database(task) as db:
                episode = db.execute("SELECT * FROM episodes WHERE id=?", (identifier,)).fetchone()
                if episode is None:
                    raise ValueError("unknown episode")
                count = db.execute(
                    "SELECT COUNT(*) FROM step_events WHERE episode_id=?", (identifier,)
                ).fetchone()[0]
                return {
                    "kind": "real",
                    "frames": count + 1,
                    "episode": dict(episode),
                    "source": "recorded observation IDs",
                }
        if kind not in ("plan", "simulation"):
            raise ValueError("playback kind must be episode, plan or simulation")
        with self.lock:
            key = (str(task.resolve()), kind, identifier)
            if key not in self.plans:
                self.plans[key] = self._replay(task, kind, identifier)
                while len(self.plans) > 4:
                    self.plans.popitem(last=False)
            self.plans.move_to_end(key)
            return {
                "kind": "predicted",
                "frames": len(self.plans[key]),
                "source": "saved actions + frozen CWM, hashes verified",
            }

    def _replay(self, task: Path, kind: str, identifier: int) -> list[dict[str, Any]]:
        with database(task) as db:
            row = db.execute(
                "SELECT payload FROM records WHERE id=? AND kind=?",
                (identifier, "plan" if kind == "plan" else "exploration"),
            ).fetchone()
            if row is None:
                raise ValueError("no completed plan with this ID")
            record = json.loads(row[0])
            if kind == "plan" and not record.get("candidate_found"):
                raise ValueError("this search produced no candidate")
            initial = _obs(db, record["initial_observation_id"])
        actions = record.get("actions" if kind == "plan" else "simulation_action_sequence")
        hashes = record.get(
            "predicted_observation_hashes" if kind == "plan" else "simulation_prediction_hashes"
        )
        if kind == "simulation":
            actions = record.get("simulation_action_sequence", record.get("dream_action_sequence"))
            hashes = record.get(
                "simulation_prediction_hashes", record.get("dream_prediction_hashes")
            )
        if actions is None or hashes is None:
            raise ValueError("no completed simulation trajectory recorded")
        bundle = task / "cwm" / "bundles" / record["model_bundle"]
        if not bundle.resolve().is_relative_to((task / "cwm" / "bundles").resolve()):
            raise ValueError("invalid bundle path")
        verify_bundle(bundle)
        status = json.loads((task / "cwm" / "status.json").read_text())
        # Replay uses only current execution settings; removed experiment options
        # in historical logs must not prevent viewing saved trajectories.
        saved_execution = status["config"].get("execution", {})
        execution = ExecutionConfig(
            **{f.name: saved_execution[f.name] for f in fields(ExecutionConfig) if f.name in saved_execution}
        )
        if len(actions) != len(hashes):
            raise ValueError("incomplete saved prediction hashes")
        frames = [initial]
        size = len(canonical(initial).encode())
        # Viewing cannot consume experiment time/actions or change its experience DB.
        from regact.orchestration.task import _secret_module_paths

        denied = _secret_module_paths(problem_for(task).secret_modules())
        with Worker(
            bundle,
            execution,
            deny_read=denied,
            deadline=time.monotonic() + 120,
            budget_key="viewer reconstruction time limit",
            budget_seconds=120,
        ) as worker:
            state = record.get("initial_state")
            if state is None:
                state = worker.call("get_initial_state", obs=initial)
            for action, expected in zip(actions, hashes, strict=True):
                state = worker.call("step", state=state, action=action)
                obs = check_observation(worker.call("render", state=state))
                if digest(obs) != expected:
                    raise ValueError(
                        "replay mismatch: saved CWM does not reproduce the recorded prediction"
                    )
                size += len(canonical(obs).encode())
                if size > 64 * 1024 * 1024:
                    raise ValueError("playback exceeds the 64 MiB plan cache allowance")
                frames.append(obs)
        return frames

    @staticmethod
    def _controller_record(db, identifier):
        row = db.execute("SELECT payload FROM records WHERE id=? AND kind='exploration'", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Unknown controller call")
        payload = json.loads(row[0])
        if not payload.get("observation_sequence"):
            raise ValueError("This historical call has no per-call replay sequence")
        return payload

    def frame(self, task: Path, kind: str, identifier: int, index: int) -> dict[str, Any]:
        if index < 0:
            raise ValueError("negative frame index")
        if kind == "controller":
            with database(task) as db:
                payload = self._controller_record(db, identifier)
                ids = payload["observation_sequence"]
                if index >= len(ids):
                    raise ValueError("frame index out of range")
                frame = {"obs": _obs(db, ids[index]), "observation_id": ids[index],
                         "step": index, "predicted": False, "episode_id": payload["episode_id"]}
                if index:
                    tid = payload["transition_sequence"][index - 1]
                    row = db.execute("SELECT a.payload FROM transitions t JOIN actions a ON a.id=t.action_id WHERE t.id=?", (tid,)).fetchone()
                    frame.update(transition_id=tid, action=json.loads(row[0]))
                return frame
        if kind in ("plan", "simulation"):
            with self.lock:
                frames = self.plans.get((str(task.resolve()), kind, identifier))
                if frames is None:
                    raise ValueError("plan cache expired; click Load again")
                if index >= len(frames):
                    raise ValueError("frame index out of range")
                return {"obs": frames[index], "step": index, "predicted": True}
        if kind != "episode":
            raise ValueError("unknown playback kind")
        with database(task) as db:
            if index == 0:
                row = db.execute(
                    "SELECT initial_obs_id FROM episodes WHERE id=?", (identifier,)
                ).fetchone()
                if row is None:
                    raise ValueError("unknown episode")
                return {
                    "obs": _obs(db, row[0]),
                    "observation_id": row[0],
                    "step": 0,
                    "predicted": False,
                }
            row = db.execute(
                (
                    "SELECT s.*,t.after_id,a.payload action FROM step_events s JOIN tr"
                    "ansitions t ON t.id=s.transition_id JOIN actions a ON a.id=t.acti"
                    "on_id WHERE s.episode_id=? AND s.step_index=?"
                ),
                (identifier, index - 1),
            ).fetchone()
            if row is None:
                raise ValueError("frame index out of range")
            return {
                "obs": _obs(db, row["after_id"]),
                "observation_id": row["after_id"],
                "action": json.loads(row["action"]),
                "transition_id": row["transition_id"],
                "event_id": row["id"],
                "step": index,
                "predicted": False,
            }


def problem_for(task: Path) -> BaseProblem:
    config = json.loads((task / "config.json").read_text())["problem"]
    return build_problem(config["name"], config.get("kwargs", {}))


def source(task: Path, bundle: str, filename: str = "") -> dict[str, Any]:
    if len(bundle) != 64 or any(c not in "0123456789abcdef" for c in bundle):
        raise ValueError("invalid bundle digest")
    root = task / "cwm" / "bundles" / bundle
    manifest = json.loads((root / "manifest.json").read_text())
    files = manifest["files"]
    if not filename:
        return {"files": list(files), "runtime": manifest["runtime"], "bytes": manifest["bytes"]}
    file = root / filename
    if (
        filename not in files
        or file.is_symlink()
        or not file.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("invalid source path")
    if file.stat().st_size > 200_000:
        return {
            "source": (
                "File exceeds viewer preview limit (200 KB); inspect the archived bundle on disk."
            )
        }
    return {"source": file.read_text()}
