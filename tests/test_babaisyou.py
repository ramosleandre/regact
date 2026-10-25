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


def test_a_rule_spelled_by_a_transformation_acts_only_next_turn():
    # TEXT ON ROCK IS KEKE turns the ON word (lying on the rock) into a keke; what is left of the
    # line then reads ROCK IS KEKE, and the rock only turns a turn later.
    level = _level(
        8,
        8,
        *_sentence(0, "baba", "is", "you"),
        ("text_text", 5, 2),
        ("text_on", 5, 3),
        ("rock", 5, 3),
        ("text_rock", 5, 4),
        ("text_is", 5, 5),
        ("text_keke", 6, 6),
        ("baba", 7, 6),
    )
    state = _play(level, "l")  # pushes KEKE under IS: TEXT ON ROCK IS KEKE
    assert _at(state, "rock") == [(5, 3)]
    assert _at(engine.step(state, engine.WAIT), "rock") == []


def test_observation_is_readable_and_lists_the_active_rules():
    level = _level(6, 3, *_sentence(0, "baba", "is", "you"), ("tile", 2, 2), ("baba", 2, 2))
    state = engine.load(level)
    # A tile lists its entries in a fixed order, whatever the engine's internal order.
    assert bbu.board(state)[0][0] == ["BABA"] and bbu.board(state)[2][2] == [
        "baba:down",
        "tile:down",
    ]
    assert [bbu.rule_text(r) for r in state.rules] == ["BABA IS YOU"]


def _at(state, name):
    return [(o.x, o.y) for o in state.objects if o.name == name]


def _play(level, moves):
    state = engine.load(level)
    for m in moves:
        state = engine.step(state, m)
    return state


def test_a_row_of_you_objects_moves_one_tile_as_a_group():
    level = _level(8, 4, *_sentence(1, "text", "is", "you"))
    for move, xs in (("r", [1, 2, 3]), ("l", [0, 1, 2])):
        state = _play(level, move)
        assert sorted(o.x for o in state.objects if o.is_text) == xs, move


def test_a_column_of_stop_you_objects_moves_together():
    level = _level(
        8,
        8,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "baba", "is", "stop"),
        ("baba", 4, 4),
        ("baba", 4, 5),
        ("baba", 4, 6),
    )
    assert sorted(_at(_play(level, "u"), "baba")) == [(4, 3), (4, 4), (4, 5)]


def test_a_shut_object_pushed_into_an_open_empty_tile_is_destroyed():
    level = _level(
        10,
        6,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "box", "is", "shut"),
        *_sentence(2, "empty", "is", "open"),
        *_sentence(3, "box", "is", "push"),
        ("baba", 2, 5),
        ("box", 3, 5),
    )
    state = _play(level, "r")
    assert _at(state, "box") == [] and _at(state, "baba") == [(3, 5)]


def test_a_stack_of_pushed_objects_moves_together():
    level = _level(
        8,
        8,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "rock", "is", "push"),
        ("baba", 3, 3),
        ("rock", 3, 4),
        ("text_keke", 3, 4),
    )
    state = _play(level, "d")
    assert _at(state, "rock") == [(3, 5)] and _at(state, "text_keke") == [(3, 5)]


def test_pull_follows_the_mover_and_blocks_like_stop():
    level = _level(
        8,
        5,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "rock", "is", "pull"),
        ("rock", 2, 3),
        ("baba", 3, 3),
    )
    state = _play(level, "r")
    assert _at(state, "baba") == [(4, 3)] and _at(state, "rock") == [(3, 3)]
    state = _play(level, "l")  # walking into a PULL object is blocked
    assert _at(state, "baba") == [(3, 3)]


def test_a_line_that_is_push_and_pull_moves_one_tile_when_pushed():
    level = _level(
        10,
        4,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "text", "is", "pull"),
        ("text_rock", 3, 3),
        ("text_rock", 4, 3),
        ("text_rock", 5, 3),
        ("baba", 6, 3),
    )
    xs = sorted(o.x for o in _play(level, "l").objects if o.name == "text_rock")
    assert xs == [2, 3, 4]


def test_swap_trades_places_with_what_moves_into_it():
    level = _level(
        8,
        5,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "rock", "is", "swap"),
        ("baba", 2, 3),
        ("rock", 3, 3),
    )
    state = _play(level, "r")
    assert _at(state, "baba") == [(3, 3)] and _at(state, "rock") == [(2, 3)]


def test_shift_carries_and_turns_what_stands_on_it():
    level = {
        "width": 8,
        "height": 6,
        "objects": [
            *(
                {"x": x, "y": y, "name": n, "dir": 0}
                for n, x, y in _sentence(0, "baba", "is", "you")
            ),
            *(
                {"x": x, "y": y, "name": n, "dir": 0}
                for n, x, y in _sentence(1, "belt", "is", "shift")
            ),
            {"x": 2, "y": 4, "name": "baba", "dir": 0},
            {"x": 3, "y": 4, "name": "belt", "dir": 1},  # the belt faces up
        ],
    }
    state = _play(level, "r")  # baba steps onto the belt and is carried up the same turn
    baba = next(o for o in state.objects if o.name == "baba")
    assert (baba.x, baba.y, baba.dir) == (3, 3, engine.UP)


def test_tele_sends_an_object_to_the_other_teleporter():
    level = _level(
        10,
        5,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "love", "is", "tele"),
        ("baba", 1, 3),
        ("love", 2, 3),
        ("love", 7, 3),
    )
    assert _at(_play(level, "r"), "baba") == [(7, 3)]


def test_fall_drops_to_the_floor_and_a_falling_key_opens_a_door():
    level = _level(
        8,
        8,
        *_sentence(0, "key", "is", "fall"),
        *_sentence(1, "key", "is", "open"),
        *_sentence(2, "door", "is", "shut"),
        *_sentence(3, "door", "is", "stop"),
        ("key", 6, 4),
        ("door", 6, 7),
        ("key", 1, 5),
    )
    state = _play(level, engine.WAIT)
    assert _at(state, "key") == [(1, 7)] and _at(state, "door") == []


def test_on_makes_a_rule_hold_only_where_the_condition_does():
    level = _level(
        10,
        5,
        *_sentence(0, "keke", "is", "you"),
        *_sentence(1, "keke", "on", "grass", "is", "defeat"),
        ("keke", 2, 3),
        ("keke", 5, 3),
        ("grass", 6, 3),
    )
    assert "KEKE ON GRASS IS DEFEAT" in _rules(engine.load(level))
    assert _at(_play(level, "r"), "keke") == [(3, 3)]  # the one that stepped onto grass is gone


def test_a_condition_word_with_nothing_before_it_does_not_hide_the_sentence_after_it():
    rules = {bbu.rule_text(r) for r in engine._parse_line(["on", "grass", "is", "keke"])}
    assert rules == {"GRASS IS KEKE"}


def test_facing_checks_the_tile_in_front():
    level = _level(
        12,
        6,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "ghost", "facing", "wall", "is", "push"),
        *_sentence(2, "wall", "is", "stop"),
        ("baba", 2, 4),
        ("ghost", 3, 4),
        ("wall", 3, 5),  # the ghost faces down, at the wall
    )
    assert _at(_play(level, "r"), "ghost") == [(4, 4)]


def test_make_creates_its_object_under_the_maker():
    level = _level(
        8,
        4,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "baba", "make", "grass"),
        ("baba", 3, 3),
    )
    assert _at(_play(level, "r"), "grass") == [(4, 3)]


def test_more_copies_into_free_neighbours():
    level = _level(
        8,
        6,
        *_sentence(0, "lava", "is", "more"),
        *_sentence(1, "wall", "is", "stop"),
        ("lava", 3, 3),
        ("wall", 4, 3),
    )
    assert sorted(_at(_play(level, engine.WAIT), "lava")) == [(2, 3), (3, 2), (3, 3), (3, 4)]


def test_word_lets_an_object_stand_for_its_name_in_a_sentence():
    level = _level(
        8,
        6,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "rock", "is", "word"),
        ("rock", 2, 3),
        ("text_is", 3, 3),
        ("text_win", 4, 3),
        ("baba", 2, 5),
    )
    assert "ROCK IS WIN" in _rules(engine.load(level))


def test_group_members_take_the_group_rules():
    level = _level(
        8,
        6,
        *_sentence(0, "baba", "is", "group"),
        *_sentence(1, "group", "is", "you"),
        ("baba", 3, 4),
    )
    assert _at(_play(level, "r"), "baba") == [(4, 4)]


def test_empty_is_a_noun_for_every_free_tile():
    level = _level(5, 4, *_sentence(0, "empty", "is", "rock"), ("baba", 0, 3))
    state = engine.load(level)
    assert len(_at(state, "rock")) == 5 * 4 - 3 - 1


def test_empty_tiles_that_are_you_push_what_they_move_into():
    level = _level(8, 5, *_sentence(0, "empty", "is", "you"), ("text_rock", 4, 3))
    assert _at(_play(level, "l"), "text_rock") == [(3, 3)]


def test_x_is_all_keeps_the_object_and_adds_every_other_kind():
    level = _level(8, 5, *_sentence(0, "hand", "is", "all"), ("hand", 3, 3), ("star", 6, 3))
    state = engine.load(level)
    assert _at(state, "hand") == [(3, 3)] and sorted(_at(state, "star")) == [(3, 3), (6, 3)]


def test_a_direction_turns_an_object():
    level = _level(8, 4, *_sentence(0, "keke", "is", "right"), ("keke", 3, 3))
    keke = next(o for o in _play(level, engine.WAIT).objects if o.name == "keke")
    assert keke.dir == engine.RIGHT


def test_sleep_stops_you_from_moving():
    level = _level(
        8,
        4,
        *_sentence(0, "baba", "is", "you"),
        *_sentence(1, "baba", "is", "sleep"),
        ("baba", 3, 3),
    )
    assert _at(_play(level, "r"), "baba") == [(3, 3)]


def _installed_levels():
    data = json.loads(bbu._DATA.read_text(encoding="utf-8"))
    levels = [level for pack in data["packs"] for level in pack["levels"]]
    if not all(bbu.level_path(level["sha256"]).is_file() for level in levels):
        pytest.skip("needs make install-babaisyou")
    return data, levels


def _moves(level):
    if "solution_url" not in level:
        return json.loads(bbu.SOLUTIONS.read_text())[level["id"]]
    script = bbu.solution_path(level["solution_sha256"]).read_text()
    return script.split('echo -n "')[1].split('"')[0]


def test_every_level_is_won_by_its_solution_at_the_recorded_move():
    """Worlds 0-3: solutions checked on the real game; worlds 4-9: the wiki's, played here."""
    _, levels = _installed_levels()
    for level in levels:
        level_data = json.loads(bbu.level_path(level["sha256"]).read_text())
        _, won = engine.play(level_data, _moves(level))
        assert won == level["reference"], level["id"]


def test_a_stage_is_played_through_the_environment():
    data, _ = _installed_levels()
    pack = data["packs"][0]
    problem = build_problem("babaisyou", {"max_levels": 2})
    env = problem.make_env(pack["key"])
    env.reset()
    code = {"u": 1, "d": 2, "l": 3, "r": 4, "p": 5}
    for level in pack["levels"][:2]:
        for move in _moves(level)[: level["reference"]]:
            _, _, done, _, info = env.step(code[move])
    assert done and info["levels_completed"] == 2
    assert problem.secret_modules() == ("regact.problems.grid_games.babaisyou",)
