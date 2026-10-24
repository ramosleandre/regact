"""Baba Is You: the engine's rules on small boards, and every campaign level replayed."""

import json

import pytest

from regact.problems.base import build_problem
from regact.problems.grid_games import babaisyou as bbu
from regact.problems.grid_games.babaisyou import engine


def _level(width, height, *placed):
    return {
        "width": width,
        "height": height,
        "objects": [{"x": x, "y": y, "name": name, "dir": 3} for name, x, y in placed],
    }


def _sentence(y, *words, x=0):
    return [(f"text_{w}", x + i, y) for i, w in enumerate(words)]


def _rules(state):
    return {bbu.rule_text(r) for r in state.rules}


def test_sentences_read_across_and_down_with_and_and_not():
    level = _level(
        8,
        8,
        *_sentence(0, "baba", "and", "keke", "is", "you"),
        *_sentence(1, "not", "rock", "is", "push"),
        ("text_flag", 6, 2),
        ("text_is", 6, 3),
        ("text_win", 6, 4),
        ("baba", 0, 7),
        ("rock", 1, 7),
    )
    rules = _rules(engine.load(level))
    assert {"BABA IS YOU", "KEKE IS YOU", "NOT ROCK IS PUSH", "FLAG IS WIN"} <= rules
    assert "ROCK IS PUSH" not in rules  # a NOT sentence must not also read without its NOT


def test_push_win_and_a_rule_broken_by_pushing_its_word():
    level = _level(
        8,
        6,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "flag", "is", "win"),
        ("baba", 0, 4),
        ("rock", 1, 4),
        ("flag", 3, 4),
        *_sentence(5, "rock", "is", "push"),
    )
    state = engine.load(level)
    state = engine.step(state, "r")  # pushes the rock along
    assert [(o.x, o.y) for o in state.objects if o.name == "rock"] == [(2, 4)]
    state = engine.step(state, "r")
    assert state.won is False  # the rock was pushed onto the flag, baba is next to it
    state = engine.step(state, "r")
    assert state.won  # baba reached the flag (the rock moved on)


def test_an_object_that_is_both_you_and_defeat_destroys_itself():
    level = _level(
        8,
        4,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "baba", "is", "defeat"),
        ("baba", 4, 3),
    )
    state = engine.step(engine.load(level), engine.WAIT)
    assert not [o for o in state.objects if o.name == "baba"]


def test_a_hot_key_melts_the_door_before_the_door_can_defeat_it():
    level = _level(
        9,
        5,
        *_sentence(0, "key", "is", "you"),
        *_sentence(1, "key", "is", "hot"),
        *_sentence(2, "door", "is", "defeat"),
        *_sentence(3, "door", "is", "melt"),
        ("key", 4, 4),
        ("door", 5, 4),
    )
    state = engine.step(engine.load(level), "r")
    assert [o.name for o in state.objects if not o.is_text] == ["key"]


def test_observation_is_readable_and_lists_the_active_rules():
    level = _level(6, 3, *_sentence(0, "baba", "is", "you"), ("tile", 2, 2), ("baba", 2, 2))
    state = engine.load(level)
    # A tile lists its entries in a fixed order, whatever the engine's internal order.
    assert bbu.board(state)[0][0] == ["BABA"] and bbu.board(state)[2][2] == [
        "baba:down",
        "tile:down",
    ]
    assert [bbu.rule_text(r) for r in state.rules] == ["BABA IS YOU"]


def _installed_levels():
    data = json.loads(bbu._DATA.read_text(encoding="utf-8"))
    levels = [level for pack in data["packs"] for level in pack["levels"]]
    if not all(bbu.level_path(level["sha256"]).is_file() for level in levels):
        pytest.skip("needs make install-babaisyou")
    return data, levels


def test_every_campaign_solution_checked_on_the_real_game_wins_here():
    """A solution checked on the real game wins in this engine, at the recorded move."""
    _, levels = _installed_levels()
    for level in levels:
        script = bbu.solution_path(level["solution_sha256"]).read_text()
        moves = script.split('echo -n "')[1].split('"')[0]
        _, won = engine.play(json.loads(bbu.level_path(level["sha256"]).read_text()), moves)
        assert won == level["reference"], level["id"]


def test_a_stage_is_played_through_the_environment():
    data, _ = _installed_levels()
    pack = data["packs"][0]
    problem = build_problem("babaisyou", {"max_levels": 2})
    env = problem.make_env(pack["key"])
    env.reset()
    code = {"u": 1, "d": 2, "l": 3, "r": 4, "p": 5}
    for level in pack["levels"][:2]:
        moves = bbu.solution_path(level["solution_sha256"]).read_text()
        for move in moves.split('echo -n "')[1].split('"')[0][: level["reference"]]:
            _, _, done, _, info = env.step(code[move])
    assert done and info["levels_completed"] == 2
    assert problem.secret_modules() == ("regact.problems.grid_games.babaisyou",)
