"""The Code World Model feature (representation + transition model).

Records every real env transition into the agent's workdir (via the env-wrapper
seam, server-side — the agent cannot forge or lose data) and scaffolds
``world_model/``: the ``parse``/``render``/``State``/``step`` stubs the agent
fills, plus a self-contained ``verify.py`` it runs to measure - against the
recorded data - representation coherence (``render(parse(o)) == o``), parser
injectivity, and transition coherence (``render(step(parse(o), a)) == o2``). The
feature also contributes data-integrity numbers (conflicting transitions,
coverage) to each submission's metrics. Design + decisions: ``context/cwm_v2.md``
and ``context/cwm_v3.md`` (v3 = the transition model).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from regact.env.wrapper import EnvWrapper
from regact.envclient.obs import Action, Obs
from regact.features.base import (
    Feature,
    FeatureContext,
    Hook,
    RunDeps,
    TemplateFile,
    register_feature,
)
from regact.tools.base import Tool

logger = logging.getLogger(__name__)

_TRANSITIONS_RELPATH = os.path.join("data", "transitions.jsonl")
_PROMPT_MD = Path(__file__).parent / "prompts" / "cwm.md"
_VERIFY_TEMPLATE = Path(__file__).parent / "templates" / "cwm_verify.py"

_STATE_STUB = '''\
"""State: your compressed representation of ONE observation.

render(state) rebuilds the WHOLE observation from this, so the state must carry
everything the observation shows. Keep it small: a state as big as the obs has
understood nothing. Design the fields yourself; instances are plain data.
"""


class State:
    pass
'''

_PARSER_STUB = '''\
"""parse(obs) -> State: read one json observation dict into your state.

Inspect a transition first (``from verify import load_transitions``) to see this
game's observation structure - do not assume it.
"""

from model_state import State


def parse(obs: dict) -> State:
    raise NotImplementedError
'''

_RENDER_STUB = '''\
"""render(state) -> obs dict: rebuild the observation EXACTLY as observed.

Everything the observation contains - not just the main frame - must come back
out, same shape and values.
"""

from model_state import State


def render(state: State) -> dict:
    raise NotImplementedError
'''

_TRANSITION_STUB = '''\
"""step(state, action) -> State: advance the world one action.

Return the NEXT state, so that render(step(parse(o), a)) is the next observation.
Fill this once parse/render are coherent; verify reports transition accuracy as
n/a until then. Because step maps State -> State, you can chain it to "dream" a
plan forward in code without spending real environment steps.
"""

from model_state import State


def step(state: State, action) -> State:
    raise NotImplementedError
'''

_NOTES_STUB = '''\
"""Your notes on the world model: interpretation, open questions, known flaws.

Free text (this docstring). Not counted in complexity, not executed.
"""
'''


class TransitionRecorder:
    """Appends transitions to JSONL, one write-and-close per line (readable live; a
    crash truncates at most one line). One recorder per task, shared by every env
    instance the lifecycle builds. Runs in the trusted orchestrator (never the
    sandboxed eval subprocess), so it sees all env traffic.

    Two destinations, with different trust:

    - ``canonical`` (in the run output dir, outside the agent sandbox) is
      **authoritative** and written strictly — an I/O fault there propagates and
      crashes the run (a failing recorder means a dysfunctional feature; env
      wrappers are trusted and not caught, by design).
    - ``mirror`` (in the agent-writable workdir) is a **best-effort** copy so the
      agent can read/verify its data. It is symlink-guarded (realpath containment
      + ``O_NOFOLLOW``) so the trusted writer cannot be turned into an
      out-of-workdir write, and a fault on it is logged, not raised — the agent
      owns this territory and must not be able to crash the run (nor does a lost
      mirror line matter: the canonical copy is authoritative).

    TODO(cwm): the mirror is the natural place to expose a *curated subset* to the
    agent (exclude eval-episode transitions, or a held-out train split) while the
    canonical keeps everything; today the mirror is a full copy.
    """

    def __init__(
        self, mirror_path: str, *, canonical_path: str | None = None, mirror_root: str | None = None
    ) -> None:
        self._mirror = mirror_path
        self._canonical = canonical_path
        self._root = os.path.realpath(
            mirror_root if mirror_root is not None else os.path.dirname(mirror_path)
        )
        self._lock = threading.Lock()

    def record(self, before: Obs, action: Action, after: Obs) -> None:
        # Build the line outside the lock; an unserializable field is a real fault
        # and is allowed to propagate (obs cross the HTTP wall as JSON, so this
        # cannot happen for genuine env traffic).
        line = json.dumps(
            {
                "o": before.to_json(),
                "a": action,
                "r": after.reward,
                "o2": after.to_json(),
                "done": after.is_done,
            }
        )
        with self._lock:
            if self._canonical is not None:
                self._append_trusted(self._canonical, line)
            self._append_mirror(self._mirror, line)

    @staticmethod
    def _append_trusted(path: str, line: str) -> None:
        """Strict append to a trusted path (crash on failure)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _append_mirror(self, path: str, line: str) -> None:
        """Best-effort append into agent territory: refuse a symlink escape, and
        never raise (a planted symlink or a chmod must not crash the run)."""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            real = os.path.realpath(path)
            if os.path.commonpath([real, self._root]) != self._root:
                logger.warning(
                    "cwm: mirror line skipped: %s resolves outside %s (symlink?)", path, self._root
                )
                return
            fd = os.open(real, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning("cwm: mirror line not written: %s", exc)


class RecordingEnvWrapper(EnvWrapper):
    """Records ``(o, a, r, o', done)`` around each ``step`` — every value a
    local of that one call, so transitions are genuine by construction. Resets
    are not recorded (their obs enters the dataset as the next step's ``o``)."""

    def __init__(self, inner: Any, recorder: TransitionRecorder) -> None:
        super().__init__(inner)
        self._recorder = recorder

    def step(self, action: Action) -> Obs:
        before: Obs | None = self._inner.last_obs
        after = super().step(action)
        if before is not None:
            self._recorder.record(before, action, after)
        return after


def _data_integrity(path: str) -> dict[str, int]:
    """Coverage + determinism check over the recorded transitions (no model needed): total count,
    distinct transitions (by the full tuple), and how many ``(o, a)`` pairs gave more than one
    outcome (a determinism / hidden-state violation). Mirrors verify.py's data-side counting."""
    outcomes: dict[str, set[str]] = {}
    distinct: set[str] = set()
    n = 0
    try:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    transition = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "o" not in transition or "o2" not in transition:
                    continue
                n += 1
                pair = json.dumps([transition["o"], transition.get("a")], sort_keys=True)
                outcome = json.dumps(
                    [transition["o2"], transition.get("r"), transition.get("done")], sort_keys=True
                )
                outcomes.setdefault(pair, set()).add(outcome)
                distinct.add(pair + outcome)
    except OSError:
        return {}
    return {
        "n_transitions": n,
        "n_distinct_transitions": len(distinct),
        "n_conflicting_transitions": sum(1 for outs in outcomes.values() if len(outs) > 1),
    }


class CwmFeature(Feature):
    """The Code World Model capability (recording + world_model/ scaffold)."""

    name = "cwm"
    # CWM only records transitions and the agent verifies its model OFFLINE against
    # that recorded data (verify.py); it never scores on the live env. So it does
    # not trip the single-instance guard and runs under both lifecycles.
    evaluates_on_env = False

    def __init__(
        self,
        *,
        max_tested_transitions_per_verify: int = 1000,
        max_printed_incoherence_transitions_per_verify: int = 10,
    ) -> None:
        self._max_tested = int(max_tested_transitions_per_verify)
        self._max_printed_failures = int(max_printed_incoherence_transitions_per_verify)
        self._canonical: str | None = None  # set in env_wrapper; read by submission_metrics

    def templates(self, ctx: FeatureContext) -> list[TemplateFile]:
        return [
            TemplateFile("world_model/model_state.py", _STATE_STUB),
            TemplateFile("world_model/model_parser.py", _PARSER_STUB),
            TemplateFile("world_model/model_render.py", _RENDER_STUB),
            TemplateFile("world_model/model_transition.py", _TRANSITION_STUB),
            TemplateFile("world_model/model_notes.py", _NOTES_STUB),
            TemplateFile("world_model/verify.py", self._verify_source()),
        ]

    def prompt_fragment(self, ctx: FeatureContext) -> str | None:
        return _PROMPT_MD.read_text(encoding="utf-8")

    def tools(self, deps: RunDeps) -> list[Tool]:
        return []

    def hooks(self, deps: RunDeps) -> list[Hook]:
        return []

    def submission_metrics(self, deps: RunDeps) -> dict[str, Any]:
        """Data-integrity numbers over the trusted recorded transitions (model-independent):
        total, distinct, and conflicting - the last should be 0 under the deterministic +
        fully-observable hypothesis, so a non-zero value flags it in post-hoc analysis. Logged
        under this feature's ``cwm`` key on each submission; the agent sees the same conflict count
        in ``verify.py`` output."""
        if not self._canonical or not os.path.exists(self._canonical):
            return {}
        return _data_integrity(self._canonical)

    def env_wrapper(self, ctx: FeatureContext) -> Callable[[Any], Any] | None:
        self._canonical = (
            os.path.join(ctx.output_dir, "cwm", "transitions.jsonl") if ctx.output_dir else None
        )
        recorder = TransitionRecorder(
            os.path.join(ctx.workdir, _TRANSITIONS_RELPATH),
            canonical_path=self._canonical,
            mirror_root=ctx.workdir,
        )
        return lambda env: RecordingEnvWrapper(env, recorder)

    def _verify_source(self) -> str:
        """The verify.py template with this run's parameters baked in as defaults."""
        source = _VERIFY_TEMPLATE.read_text(encoding="utf-8")
        return source.replace('int("__MAX_UNIQUE_USED__")', str(self._max_tested)).replace(
            'int("__MAX_INCOHERENCES__")', str(self._max_printed_failures)
        )


register_feature(CwmFeature.name, CwmFeature)
