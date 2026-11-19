"""Crafter: the open-world survival game of Hafner (2021), as a regact problem.

The agent sees what the game's picture shows and nothing more: the 9x7 map window around the
player, written as names instead of pixels, the player's facing, the inventory and the daylight.
The world position and the full map stay hidden. ``crafter`` is imported lazily in
:meth:`make_env`.

Two kinds of task, alike in everything but the map of each life:

- seeded (``crafter_s3``): every life starts on the same map, seed 3. It measures how far an
  agent gets in one world it can come back to.
- unseeded (``crafter_u3``): every life is on a new map, drawn from the task's own stream. It
  measures what an agent learns about the game. When such a task ends, its final controller
  plays one life on each of ``eval_seeds``, maps no task ever shows, for at most
  ``eval_max_actions`` actions (10 episodes of 2000 steps is BALROG's setting), and its CWM is
  replayed alone on the reference action lists of ``data/reference_actions.json``.

In both, a life starts with an empty inventory and the achievements unlocked in earlier lives of
the task stay unlocked.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

from regact.config.schema import InfoMode, ObsMode
from regact.env.renderer import ObsRenderer
from regact.envclient.errors import InvalidActionError
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory, RegactError
from regact.problems.base import BaseProblem, observation_prompt, register_problem
from regact.protocols.managed.evaluation import FINAL_CONTROLLER
from regact.workspace.templates import TemplateFile

_PROMPT = Path(__file__).parents[1] / "prompts" / "crafter.md"
_ENVIRONMENT = Path(__file__).parents[1] / "prompts" / "crafter_environment.md"
_TASK = "crafter_s"  # + the map seed
_UNSEEDED = "crafter_u"  # + the index of a stream of maps
TASK = _TASK + "0"
_STREAM = 1000  # life i of crafter_u<k> is map (k + 1) * _STREAM + i, away from the other seeds
_REFERENCE = Path(__file__).parent / "data" / "reference_actions.json"
ACTIONS = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)  # fmt: skip
ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling",
    "collect_stone", "collect_wood", "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone",
    "place_table", "wake_up",
)  # fmt: skip
VIEW = (9, 7)  # columns, rows of the map window; the player is at its centre
OUTSIDE = "void"  # a window cell beyond the edge of the world
_FACING = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}
_UNIT = 48  # px per tile in pictures
_VITALS = ("health", "food", "drink", "energy")
# the sentences of the Environment section that differ: a seeded task, then an unseeded one
_SCORE = (
    "Your score is the number of different achievements unlocked: more is better, and reaching "
    "them with fewer actions is better.",
    "The game counts achievements: unlocking more of them in a life is better, and reaching "
    "them with fewer actions is better.",
)
_RESET = (
    "at the starting point of the same world",
    "in a new world (the same game and rules, on a map you have not seen)",
)
_WON = (
    "The task is won when every achievement is unlocked.",
    "The task also ends once every achievement has been unlocked over your lives.",
)
_EVALUATION = """
**Final evaluation.** The task can end at any moment, when your budget runs out. One controller is then run without you in {n} more new worlds: one life in each, from the start of a life with nothing unlocked, for at most {cap} actions. The controller runs there as a single call, with no limit of actions per call. A life there ends when the player dies, when the controller stops (`is_done`) or raises an error, or at the action limit. The achievements it unlocks per life are the main result of the task.

The controller evaluated is the file `{file}`, read as it stands when the task ends. It has the same interface as `controller.py` and the framework never runs it during the task: to try it, reset and run its code as `controller.py`. Create it as soon as you have a controller that plays a whole life from its start, keep it valid, and improve it as you learn. Without this file, the controller of your last `RunController` call is evaluated.
"""
_NIGHT = (0, 16, 64)  # what the picture is tinted toward as daylight falls

_HELPER = (
    '"""Crafter action ids and achievement names."""\n\n'
    + "\n".join(f"{name.upper()} = {i}" for i, name in enumerate(ACTIONS))
    + f"\n\nACTION_NAMES = {list(ACTIONS)!r}\nACHIEVEMENTS = {list(ACHIEVEMENTS)!r}\n"
)


class CrafterEnv:
    """One Crafter task behind the gym-like interface the env server drives."""

    def __init__(self, seed: int, *, new_map_each_life: bool = False) -> None:
        self._seed = seed
        self._new_map_each_life = new_map_each_life
        self._lives = 0  # started so far
        self._unlocked: set[str] = set()  # over every life of the task
        self._env = self._world()

    def _world(self) -> Any:
        import crafter

        class Env(crafter.Env):  # type: ignore[misc]
            def _balance_chunk(self, chunk: Any, objs: Any) -> None:
                # upstream iterates a set of objects, whose order changes from run to run
                ordered = sorted(objs, key=lambda o: (type(o).__name__, *map(int, o.pos)))
                super()._balance_chunk(chunk, ordered)

        return Env(seed=self._seed + self._lives if self._new_map_each_life else self._seed)

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self._env = self._world()
        self._lives += 1
        self._env.reset()
        return self._frame(), self._info()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise InvalidActionError(
                f"expected an action in 0..{len(ACTIONS) - 1}; got {action!r}"
            )
        _, reward, done, _ = self._env.step(action)
        dead = self._env._player.health <= 0
        info = self._info()
        won = len(info["achievements"]) == len(ACHIEVEMENTS)
        return self._frame(), round(float(reward), 6), dead or won, bool(done and not dead), info

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        pass

    def _frame(self) -> dict[str, Any]:
        world, player = self._env._world, self._env._player
        px, py = (int(v) for v in player.pos)
        width, height = world.area
        rows = []
        for y in range(py - VIEW[1] // 2, py + VIEW[1] // 2 + 1):
            row = []
            for x in range(px - VIEW[0] // 2, px + VIEW[0] // 2 + 1):
                if not (0 <= x < width and 0 <= y < height):
                    row.append(OUTSIDE)
                    continue
                material, obj = world[x, y]
                if obj is None:
                    row.append(material or OUTSIDE)
                else:
                    name = type(obj).__name__.lower()
                    if name == "plant" and obj.ripe:
                        name = "plant-ripe"
                    elif name == "arrow":
                        name = f"arrow-{_FACING[tuple(int(v) for v in obj.facing)]}"
                    row.append(name)
            rows.append(row)
        return {
            "view": rows,
            "facing": _FACING[tuple(int(v) for v in player.facing)],
            "inventory": {k: int(v) for k, v in player.inventory.items()},
            "sleeping": bool(player.sleeping),
            "daylight": round(float(world.daylight), 2),
        }

    def _info(self) -> dict[str, Any]:
        life = sorted(n for n, count in self._env._player.achievements.items() if count > 0)
        self._unlocked.update(life)
        return {
            "achievements": sorted(self._unlocked),
            "life_achievements": life,
            "available_actions": list(range(len(ACTIONS))),
        }


class CrafterRenderer(ObsRenderer):
    def render(self, native_obs: object, info: dict[str, Any] | None) -> Obs:
        info = dict(info or {})
        return Obs(
            frame=native_obs,
            available_actions=list(info.get("available_actions", [])),
            info=info,
        )


def _unlocked(info: dict[str, Any] | None) -> list[str]:
    return list((info or {}).get("achievements") or [])


def _milestones(env: Any) -> list[str]:
    last, prev = getattr(env, "last_obs", None), getattr(env, "prev_obs", None)
    if last is None:
        return []
    before = set(_unlocked(prev.info if prev is not None else None))
    out = [f"achievement unlocked: {name}" for name in _unlocked(last.info) if name not in before]
    health = ((last.frame or {}).get("inventory") or {}).get("health", 1)
    if getattr(env, "last_terminated", False) and health <= 0:  # a win also ends the life
        out.append("the player died")
    return out


class CrafterProblem(BaseProblem):
    """A task is a Crafter world. The score is the number of achievements unlocked over the
    task's lives; the task is won when all are."""

    name = "crafter"
    main_metrics = (
        "achievements",
        "first_life_achievements",
        "best_life_achievements",
        "eval_progress",
        "eval_crafter_score",
        "eval_model_explained_rate",
    )
    progress_label = "Achievements unlocked"

    def __init__(
        self,
        *,
        seeds: list[int] | None = None,
        eval_seeds: list[int] | None = None,
        eval_max_actions: int | None = 2000,
        unseeded: list[int] | None = None,
    ) -> None:
        self._seeds = [0] if seeds is None else [int(s) for s in seeds]
        self._unseeded = [int(k) for k in unseeded or []]
        self._eval_seeds = [int(s) for s in eval_seeds or []]
        self._eval_max_actions = eval_max_actions

    def make_env(self, task_name: str) -> Any:
        prefix = _UNSEEDED if task_name.startswith(_UNSEEDED) else _TASK
        number = task_name.removeprefix(prefix)
        if not task_name.startswith(prefix) or not number.isdigit():
            raise RegactError(ErrorCategory.EVAL_HARNESS, f"unknown crafter task {task_name!r}")
        if prefix == _UNSEEDED:
            return CrafterEnv((int(number) + 1) * _STREAM, new_map_each_life=True)
        return CrafterEnv(int(number))

    def get_task_names(self) -> list[str]:
        return [f"{_TASK}{seed}" for seed in self._seeds] + [
            f"{_UNSEEDED}{k}" for k in self._unseeded
        ]

    def evaluation_tasks(self, task_name: str) -> list[str]:
        if not task_name.startswith(_UNSEEDED):  # a seeded task learns one map, not the game
            return []
        return [f"{_TASK}{seed}" for seed in self._eval_seeds]

    def evaluation_trajectories(self, task_name: str) -> list[tuple[str, list[Any]]]:
        held_out = set(self.evaluation_tasks(task_name))
        if not held_out or not _REFERENCE.is_file():
            return []
        recorded = json.loads(_REFERENCE.read_text(encoding="utf-8"))
        return [
            (name, actions)
            for name, lists in recorded["tasks"].items()
            if name in held_out
            for actions in lists.values()
        ]

    def evaluation_max_actions(self) -> int | None:
        return self._eval_max_actions

    def evaluation_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """BALROG's progress (the share of the achievements unlocked in an episode, averaged) and
        the Crafter score over the episodes' per-achievement success rates, both in percent."""
        if not episodes:
            return {}
        unlocked = [set(_unlocked(e.get("info"))) for e in episodes]
        rates = {
            name: 100 * sum(name in life for life in unlocked) / len(unlocked)
            for name in ACHIEVEMENTS
        }
        return {
            "eval_episodes": len(unlocked),
            "eval_achievements": sum(map(len, unlocked)) / len(unlocked),
            "eval_progress": 100 * sum(map(len, unlocked)) / len(unlocked) / len(ACHIEVEMENTS),
            "eval_crafter_score": crafter_score(rates.values()),
            "eval_success_rates": rates,
        }

    def obs_renderer(self, task_name: str, *, mode: ObsMode) -> ObsRenderer:
        if mode is not ObsMode.RAW:
            raise RegactError(ErrorCategory.ENV_RUNTIME, f"crafter: obs_mode {mode!r} not supported")
        return CrafterRenderer()

    def environment_prompt(self, task_name: str) -> str | None:
        evaluated = len(self.evaluation_tasks(task_name))
        evaluation = _EVALUATION.format(
            n=evaluated, cap=self._eval_max_actions, file=FINAL_CONTROLLER
        )
        kind = int(task_name.startswith(_UNSEEDED))
        return (
            _ENVIRONMENT.read_text(encoding="utf-8")
            .replace("{score}", _SCORE[kind])
            .replace("{reset}", _RESET[kind])
            .replace("{won}", _WON[kind])
            .replace("{evaluation}", evaluation if evaluated else "")
            .strip()
        )

    def reset_commands(self) -> dict[str, str]:
        return {
            "ResetEnvironment": "Start a new life: the inventory is lost, the achievements "
            "already unlocked are kept."
        }

    def objective_prompt(self, task_name: str) -> str:
        if self.evaluation_tasks(task_name):
            return (
                "Learn to play this game in worlds you have not seen. Your lives here are for "
                f"discovering its rules and for testing what you build; `{FINAL_CONTROLLER}`, as "
                "it stands when the task ends, is the main result. Keep working until the "
                "framework ends the task for budget reasons."
            )
        return (
            "Unlock as many achievements as you can. Keep working until all are unlocked or the "
            "framework ends the task for budget reasons."
        )

    def continue_reminder(self) -> str:
        return "Continue your work: the task is not over."

    def progress_value(self, info: dict[str, Any]) -> float | None:
        return len(_unlocked(info))

    def milestone_detector(self, task_name: str) -> Any:
        return _milestones

    def milestone_kind(self, milestone: str) -> str:
        if milestone.startswith("achievement unlocked"):
            return "progress"
        return "failure" if milestone == "the player died" else "event"

    def helper_templates(
        self,
        task_name: str,
        *,
        info_mode: InfoMode = InfoMode.INFORMATIVE,
        direct_interaction: bool = True,
    ) -> list[TemplateFile]:
        if info_mode is InfoMode.MINIMAL:
            return []
        return [TemplateFile("framework/crafter_helper.py", _HELPER)]

    @classmethod
    def secret_modules(cls) -> tuple[str, ...]:
        return ("crafter",)

    def warmup(self) -> None:
        import crafter  # noqa: F401

    def render_frame(self, obs: Obs) -> Any | None:
        """The game's own picture, rebuilt from the observation: the map window in the game's
        tiles, darker as daylight falls, and the inventory under it."""
        frame = obs.frame if isinstance(obs.frame, dict) else None
        if not frame or not frame.get("view"):
            return None
        import numpy as np
        from PIL import Image

        view = frame["view"]
        columns, rows = len(view[0]), len(view)
        world = Image.new("RGB", (columns * _UNIT, rows * _UNIT), (0, 0, 0))
        for y, row in enumerate(view):
            for x, name in enumerate(row):
                if name == OUTSIDE:
                    continue
                if name == "player":
                    name = "player-sleep" if frame.get("sleeping") else f"player-{frame['facing']}"
                tile = _texture(name)
                if tile.mode == "RGBA":  # a creature: the observation does not say what is under it
                    ground = "path" if name.startswith(("skeleton", "arrow")) else "grass"
                    world.paste(_texture(ground), (x * _UNIT, y * _UNIT))
                world.paste(tile, (x * _UNIT, y * _UNIT), tile if tile.mode == "RGBA" else None)
        dark = (1 - float(frame.get("daylight", 1))) * 0.6 + (0.2 if frame.get("sleeping") else 0)
        if dark > 0:
            world = Image.blend(world, Image.new("RGB", world.size, _NIGHT), min(dark, 0.8))
        image = Image.new("RGB", (columns * _UNIT, (rows + 2) * _UNIT), (0, 0, 0))
        image.paste(world, (0, 0))
        inventory = frame.get("inventory") or {}
        held = [(k, inventory[k]) for k in _VITALS if k in inventory]
        held += [(k, v) for k, v in inventory.items() if k not in _VITALS and v]
        for slot, (name, count) in enumerate(held[: 2 * columns]):
            left, top = (slot % columns) * _UNIT, (rows + slot // columns) * _UNIT
            icon = _texture(name, _UNIT * 2 // 3)
            image.paste(icon, (left + 2, top + _UNIT // 6), icon if icon.mode == "RGBA" else None)
            if 1 <= count <= 9:
                digit = _texture(str(count), _UNIT // 2)
                image.paste(digit, (left + _UNIT // 2, top + _UNIT // 2), digit)
        return np.asarray(image)

    def render_obs_text(self, obs: Obs) -> str | None:
        frame = obs.frame if isinstance(obs.frame, dict) else None
        if not frame or not frame.get("view"):
            return None
        rows = [" ".join(f"{name:<8}" for name in row) for row in frame["view"]]
        held = ", ".join(f"{k}={v}" for k, v in (frame.get("inventory") or {}).items() if v)
        unlocked = _unlocked(obs.info)
        return "\n".join(
            [
                *rows,
                f"facing={frame.get('facing')} sleeping={frame.get('sleeping')} "
                f"daylight={frame.get('daylight')}",
                f"inventory: {held}",
                f"achievements ({len(unlocked)}/{len(ACHIEVEMENTS)}): {', '.join(unlocked)}",
            ]
        )

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        n = len(_unlocked(final_obs.info))
        return {"success": n == len(ACHIEVEMENTS), "steps": steps, "achievements": n}

    def failure_metrics(self, *, steps: int) -> dict[str, Any]:
        return {"success": False, "steps": steps, "achievements": 0}

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(episodes)
        if not n:
            return {"n_episodes": 0, "success_rate": 0.0, "mean_achievements": 0.0}
        return {
            "n_episodes": n,
            "success_rate": sum(bool(e.get("success")) for e in episodes) / n,
            "mean_achievements": sum(e.get("achievements", 0) for e in episodes) / n,
            "mean_steps": sum(e.get("steps", 0) for e in episodes) / n,
        }

    def exploration_score(self, aggregate: dict[str, Any]) -> float:
        return float(aggregate.get("mean_achievements", 0) or 0)

    def derived_trace_metrics(
        self, task_name: str, trace: list[tuple[int, dict[str, Any]]]
    ) -> dict[str, Any]:
        """``achievements``: how many are unlocked at the end; ``actions_to_last``: the real
        actions played when the last one was unlocked; ``lives``: per life, the actions played in
        the task when it ended and the achievements unlocked within it."""
        best, at = 0, 0
        lives: list[list[int]] = []
        previous = None
        for actions, info in trace:
            if len(_unlocked(info)) > best:
                best, at = len(_unlocked(info)), actions
            # a reset is counted as one action, so a step right after one is two further on
            if previous is None or actions > previous + 1:
                lives.append([actions, 0])
            lives[-1] = [actions, len(info.get("life_achievements") or [])]
            previous = actions
        out: dict[str, Any] = {"achievements": best, "actions_to_last": at}
        if lives:
            out.update(
                lives=lives,
                first_life_achievements=lives[0][1],
                best_life_achievements=max(n for _, n in lives),
            )
        return out

    def build_prompt(
        self,
        task_name: str,
        *,
        info_mode: InfoMode,
        obs_mode: ObsMode = ObsMode.RAW,
        direct_interaction: bool = True,
    ) -> str:
        if info_mode is InfoMode.MINIMAL:
            source = (
                "from your own scripts with `make_env()`"
                if direct_interaction
                else "in recorded experience from `framework.data_api`"
            )
            return observation_prompt(
                "# Game\n\nDiscover the rules by interaction. Inspect `obs.frame` and "
                f"`obs.available_actions` {source}; the framework tells you nothing more "
                "about this task.",
                direct_interaction=direct_interaction,
            )
        body = (
            _PROMPT.read_text(encoding="utf-8")
            .replace("{actions}", ", ".join(f"`{i}` {name}" for i, name in enumerate(ACTIONS)))
            .replace("{achievements}", ", ".join(f"`{name}`" for name in ACHIEVEMENTS))
            .replace("{n_achievements}", str(len(ACHIEVEMENTS)))
            .rstrip()
        )
        return observation_prompt(body, direct_interaction=direct_interaction)

    def config_kwargs(self) -> dict[str, Any]:
        return {
            "seeds": self._seeds,
            "unseeded": self._unseeded,
            "eval_seeds": self._eval_seeds,
            "eval_max_actions": self._eval_max_actions,
        }


def crafter_score(rates: Any) -> float:
    """Hafner's score from per-achievement success rates in percent: exp(mean(ln(1 + s))) - 1."""
    rates = list(rates)
    return math.exp(sum(math.log(1 + s) for s in rates) / len(rates)) - 1


@functools.cache
def _texture(name: str, size: int = _UNIT) -> Any:
    """One of the game's own tiles, resized without smoothing (they are 16 px pixel art)."""
    from importlib import resources

    from PIL import Image

    path = resources.files("crafter") / "assets" / f"{name}.png"
    if not path.is_file():
        path = resources.files("crafter") / "assets" / "unknown.png"
    with path.open("rb") as handle:
        tile = Image.open(handle)
        tile.load()
    return tile.resize((size, size), Image.NEAREST)


register_problem("crafter", lambda kwargs: CrafterProblem(**kwargs))
