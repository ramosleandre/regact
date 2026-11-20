"""Crafter: the open-world survival game of Hafner (2021), as a regact problem.

The agent sees what the game's picture shows and nothing more: the 9x7 map window around the
player, written as names instead of pixels, the player's facing, the inventory and the daylight.
The world position and the full map stay hidden. ``crafter`` is imported lazily in
:meth:`make_env`.

Three kinds of task. A life always starts with an empty inventory and nothing unlocked, and its
score is the number of achievements unlocked within it.

- seeded (``crafter_s3``): every life starts on the same map, seed 3. It measures how far an
  agent gets in one world it can come back to.
- unseeded (``crafter_u3``): every life is on a new map, drawn from the task's own stream. It
  measures how an agent plays while it learns the game.
- controller synthesis (``crafter_c3``): the maps of ``crafter_u3``, but the result is a program.
  When the task ends, its final controller plays one life on each of ``eval_seeds``, maps no
  task ever shows, for at most ``eval_max_actions`` actions (BALROG's setting is 10 episodes of
  2000 steps), and its CWM is replayed alone on the reference action lists of
  ``data/reference_actions.json``.
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
_SYNTHESIS = "crafter_c"  # + the same index: the same maps, scored by the final controller
TASK = _TASK + "0"
_STREAM = 1000  # life i of crafter_u<k> is map (k + 1) * _STREAM + i, away from the other seeds
_REFERENCE = Path(__file__).parent / "data" / "reference_actions.json"
ACTIONS = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)  # fmt: skip
_ACTION_ID = {name: i for i, name in enumerate(ACTIONS)}
# what each action does, as BALROG's prompt tells its agent
_ACTION_HELP = (
    "do nothing",
    "move left on flat ground",
    "move right on flat ground",
    "move up on flat ground",
    "move down on flat ground",
    "multiuse action to collect material, drink from lake and hit creature in front",
    "sleep when energy level is below maximum",
    "place a stone in front",
    "place a table",
    "place a furnace",
    "place a plant",
    "craft a wood pickaxe with a nearby table and wood in inventory",
    "craft a stone pickaxe with a nearby table, wood, and stone in inventory",
    "craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory",
    "craft a wood sword with a nearby table and wood in inventory",
    "craft a stone sword with a nearby table, wood, and stone in inventory",
    "craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory",
)
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
_DIED = "the player died"
_SCORE = "Your score in a life is the number of different achievements unlocked in it: play each life to unlock as many as you can. Every life you start counts, also one you leave by a reset, and the result of the task is read against the total number of actions you have played."
_SCORE_SYNTHESIS = "A life is scored by the number of different achievements unlocked in it."
_RESET = (
    "at the starting point of the same world",
    "in a new world (the same game and rules, on a map you have not seen)",
)
_EVALUATION = """
**Final evaluation.** The task can end at any moment, when your budget runs out. One controller is then run without you in {n} more new worlds: one life in each, for at most {cap} actions. The controller runs there as a single call, with no limit of actions per call. A life there ends when the player dies, when the controller stops (`is_done`) or raises an error, or at the action limit. The achievements it unlocks per life are the result of the task.

The controller evaluated is the file `{file}`, read as it stands when the task ends. It has the same interface as `controller.py` and the framework never runs it during the task: to try it, reset and run its code as `controller.py`. Create it as soon as you have a controller that plays a whole life from its start, keep it valid, and improve it as you learn. Without this file, the controller of your last `RunController` call is evaluated.
"""
_NIGHT = (0, 16, 64)  # what the picture is tinted toward as daylight falls

_HELPER = (
    '"""Crafter action and achievement names."""\n\n'
    f"ACTIONS = {list(ACTIONS)!r}\nACHIEVEMENTS = {list(ACHIEVEMENTS)!r}\n"
)


class CrafterEnv:
    """One Crafter task behind the gym-like interface the env server drives."""

    def __init__(
        self, seed: int, *, new_map_each_life: bool = False, life_max_actions: int = 10000
    ) -> None:
        self._seed = seed
        self._new_map_each_life = new_map_each_life
        self._life_max_actions = life_max_actions
        self._lives = 0  # started so far
        self._env = self._world()

    def _world(self) -> Any:
        import crafter

        class Env(crafter.Env):  # type: ignore[misc]
            def _balance_chunk(self, chunk: Any, objs: Any) -> None:
                # upstream iterates a set of objects, whose order changes from run to run
                ordered = sorted(objs, key=lambda o: (type(o).__name__, *map(int, o.pos)))
                super()._balance_chunk(chunk, ordered)

        seed = self._seed + self._lives if self._new_map_each_life else self._seed
        return Env(seed=seed, length=self._life_max_actions)

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self._env = self._world()
        self._lives += 1
        self._env.reset()
        return self._frame(), self._info()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if type(action) is not str or action not in _ACTION_ID:
            raise InvalidActionError(f"expected an action name of available_actions; got {action!r}")
        _, reward, done, _ = self._env.step(_ACTION_ID[action])
        dead = self._env._player.health <= 0
        return self._frame(), round(float(reward), 6), dead, bool(done and not dead), self._info()

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
        return {"achievements": life, "available_actions": list(ACTIONS)}


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
    if getattr(env, "last_terminated", False) and health <= 0:
        out.append(_DIED)
    return out


class CrafterProblem(BaseProblem):
    """A task is a Crafter world or a stream of them. A life is scored by the achievements
    unlocked within it; no task ends early, since one good life is not a solved game."""

    name = "crafter"
    main_metrics = (
        "achievements",
        "first_life_achievements",
        "best_life_achievements",
        "mean_life_achievements",
        "eval_progress",
        "eval_crafter_score",
        "eval_model_explained_rate",
    )
    progress_label = "Achievements unlocked in one life"

    def __init__(
        self,
        *,
        seeds: list[int] | None = None,
        eval_seeds: list[int] | None = None,
        eval_max_actions: int | None = 2000,
        life_max_actions: int = 2000,
        unseeded: list[int] | None = None,
        synthesis: list[int] | None = None,
    ) -> None:
        self._seeds = [0] if seeds is None else [int(s) for s in seeds]
        self._unseeded = [int(k) for k in unseeded or []]
        self._synthesis = [int(k) for k in synthesis or []]
        self._eval_seeds = [int(s) for s in eval_seeds or []]
        self._eval_max_actions = eval_max_actions
        self._life_max_actions = int(life_max_actions)

    def make_env(self, task_name: str) -> Any:
        prefix = next((p for p in (_UNSEEDED, _SYNTHESIS) if task_name.startswith(p)), _TASK)
        number = task_name.removeprefix(prefix)
        if not task_name.startswith(prefix) or not number.isdigit():
            raise RegactError(ErrorCategory.EVAL_HARNESS, f"unknown crafter task {task_name!r}")
        if prefix != _TASK:
            return CrafterEnv(
                (int(number) + 1) * _STREAM,
                new_map_each_life=True,
                life_max_actions=self._life_max_actions,
            )
        return CrafterEnv(int(number), life_max_actions=self._life_max_actions)

    def get_task_names(self) -> list[str]:
        return [
            *(f"{_TASK}{seed}" for seed in self._seeds),
            *(f"{_UNSEEDED}{k}" for k in self._unseeded),
            *(f"{_SYNTHESIS}{k}" for k in self._synthesis),
        ]

    def evaluation_tasks(self, task_name: str) -> list[str]:
        if not task_name.startswith(_SYNTHESIS):
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
        actions = "\n".join(f"- `{n}`: {h}" for n, h in zip(ACTIONS, _ACTION_HELP, strict=True))
        return (
            _ENVIRONMENT.read_text(encoding="utf-8")
            .replace("{actions}", actions)
            .replace("{achievements}", ", ".join(f"`{name}`" for name in ACHIEVEMENTS))
            .replace("{score}", _SCORE_SYNTHESIS if evaluated else _SCORE)
            .replace("{life}", str(self._life_max_actions))
            .replace("{reset}", _RESET[not task_name.startswith(_TASK)])
            .replace("{evaluation}", evaluation if evaluated else "")
            .strip()
        )

    def action_format(self) -> str:
        return "An action is one of the names in `available_actions`, as a string."

    def reset_commands(self) -> dict[str, str]:
        return {"ResetEnvironment": "Start a new life, with an empty inventory and nothing unlocked."}

    def objective_prompt(self, task_name: str) -> str:
        if self.evaluation_tasks(task_name):
            return (
                f"Produce a program that plays this game by itself: `{FINAL_CONTROLLER}`, a "
                "controller that plays a whole life well, from its start, in a world it has never "
                "seen. It is the result of this task: when the task ends, it is run without you "
                "in new worlds and scored by the achievements it unlocks per life. Your own lives "
                "here are the means: use them to learn the game and to test what you build. Keep "
                "working until the framework ends the task for budget reasons."
            )
        return (
            "Unlock as many achievements as you can within a life. Keep working until the "
            "framework ends the task for budget reasons."
        )

    def continue_reminder(self, task_name: str | None = None) -> str:
        reminder = "Continue your work: the task is not over."
        if task_name and self.evaluation_tasks(task_name):
            reminder += (
                f" `{FINAL_CONTROLLER}` is what will be evaluated: keep it up to date with the "
                "best whole-life controller you have."
            )
        return reminder

    def is_perfect(self, aggregate: dict[str, Any]) -> bool:
        return False

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
        """``lives``: per life, the task's real actions when it ended, the achievements unlocked
        within it and whether the player died (a life can also end at the action limit or be
        abandoned by a reset). The mean and the worst are over every life started.
        ``achievements`` counts those unlocked in at least one life, ``actions_to_last`` the real
        actions played when the last of them first was."""
        seen: set[str] = set()
        at = 0
        lives: list[list[int]] = []
        previous = None
        for actions, info in trace:
            if not seen.issuperset(_unlocked(info)):
                seen.update(_unlocked(info))
                at = actions
            # a reset is counted as one action, so a step right after one is two further on
            if previous is None or actions > previous + 1:
                lives.append([actions, 0, 0])
            lives[-1] = [actions, len(_unlocked(info)), int(_DIED in (info.get("milestones") or []))]
            previous = actions
        out: dict[str, Any] = {"achievements": len(seen), "actions_to_last": at}
        if lives:
            ended = [n for _, n, _ in lives]
            out.update(
                lives=lives,
                first_life_achievements=lives[0][1],
                best_life_achievements=max(n for _, n, _ in lives),
                mean_life_achievements=round(sum(ended) / len(ended), 2),
                worst_life_achievements=min(ended),
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
            .replace("{actions}", ", ".join(f"`{name}`" for name in ACTIONS))
            .replace("{achievements}", ", ".join(f"`{name}`" for name in ACHIEVEMENTS))
            .replace("{n_achievements}", str(len(ACHIEVEMENTS)))
            .rstrip()
        )
        return observation_prompt(body, direct_interaction=direct_interaction)

    def config_kwargs(self) -> dict[str, Any]:
        return {
            "seeds": self._seeds,
            "unseeded": self._unseeded,
            "synthesis": self._synthesis,
            "eval_seeds": self._eval_seeds,
            "eval_max_actions": self._eval_max_actions,
            "life_max_actions": self._life_max_actions,
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
