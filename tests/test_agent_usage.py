"""Token usage harvested from the CLI agents' own session logs before their home is deleted."""

import json
import os
from pathlib import Path

from regact.agent.claude_adapter import ClaudeAgent
from regact.agent.codex_adapter import CodexAgent
from regact.agent.usage import claude_usage, codex_usage


def _jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + "not json\n")


def _claude_line(msg_id: str, model: str, **usage: int) -> dict:
    return {"requestId": "req-" + msg_id, "message": {"id": msg_id, "model": model, "usage": usage}}


def test_claude_usage_counts_each_response_once_per_model(tmp_path: Path) -> None:
    one = {"input_tokens": 2, "output_tokens": 10, "cache_read_input_tokens": 100}
    _jsonl(
        tmp_path / "projects" / "-work" / "s1.jsonl",
        [
            _claude_line("a", "claude-sonnet-5", **one),
            _claude_line("a", "claude-sonnet-5", **one),  # same response, second content block
            _claude_line("b", "claude-opus-5-5", input_tokens=1, cache_creation_input_tokens=7),
            {"message": {"model": "<synthetic>", "usage": {"output_tokens": 0}}},
        ],
    )
    usage = claude_usage(str(tmp_path))
    assert usage["responses"] == 2
    assert usage["models"]["claude-sonnet-5"] == {
        "input": 2,
        "output": 10,
        "cache_write": 0,
        "cache_read": 100,
    }
    assert usage["models"]["claude-opus-5-5"]["cache_write"] == 7
    assert claude_usage(str(tmp_path / "empty")) is None


def test_codex_usage_takes_the_cumulative_total_and_rate_limit_endpoints(tmp_path: Path) -> None:
    def token_count(ts: str, total_in: int, used: float) -> dict:
        return {
            "timestamp": ts,
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": total_in,
                        "cached_input_tokens": 5,
                        "output_tokens": 3,
                    }
                },
                "rate_limits": {"primary": {"used_percent": used, "window_minutes": 10080}},
            },
        }

    _jsonl(
        tmp_path / "sessions" / "2026" / "10" / "01" / "rollout-x.jsonl",
        [
            {"type": "turn_context", "payload": {"model": "gpt-5.6-sol"}},
            token_count("t1", 10, 40.0),
            token_count("t2", 30, 41.5),
        ],
    )
    usage = codex_usage(str(tmp_path))
    assert usage["models"] == {
        "gpt-5.6-sol": {"input": 30, "cache_read": 5, "output": 3, "reasoning_output": 0}
    }
    assert usage["rate_limits_start"]["primary"]["used_percent"] == 40.0
    assert usage["rate_limits_end"]["primary"]["used_percent"] == 41.5
    assert codex_usage(str(tmp_path / "empty")) is None


async def test_claude_close_harvests_usage_before_deleting_the_home(tmp_path: Path) -> None:
    root = tmp_path / "claude-home"
    root.mkdir()
    (root / ".credentials.json").write_text("{}")
    agent = ClaudeAgent({"claude_home": str(root)})
    home = Path(agent._config_dir())
    _jsonl(home / "projects" / "-w" / "s.jsonl", [_claude_line("a", "m", output_tokens=4)])
    await agent.close()
    assert not home.exists()
    assert agent.usage()["models"]["m"]["output"] == 4


async def test_codex_close_harvests_usage_before_deleting_the_home(tmp_path, monkeypatch) -> None:
    agent = CodexAgent({"codex_home": str(tmp_path / "codex-home")})
    monkeypatch.setattr(agent, "_freshest_auth", lambda: None)
    home = agent._config_dir()
    _jsonl(
        Path(home, "sessions", "r", "rollout-1.jsonl"),
        [
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"output_tokens": 9}},
                },
            }
        ],
    )
    await agent.close()
    assert not os.path.exists(home)
    assert agent.usage()["models"]["unknown"]["output"] == 9
