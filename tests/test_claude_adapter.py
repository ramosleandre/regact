"""Claude Code adapter: recognising a usage-limit error and its reset time."""

from datetime import datetime
from zoneinfo import ZoneInfo

from regact.agent.claude_adapter import limit_reset_unix


def test_claude_limit_message_gives_the_reset_time() -> None:
    paris = ZoneInfo("Europe/Paris")
    now = datetime(2026, 10, 6, 16, 14, tzinfo=paris).timestamp()

    def reset(message: str) -> str | None:
        at = limit_reset_unix(message, now)
        return datetime.fromtimestamp(at, paris).strftime("%m-%d %H:%M") if at else None

    assert reset("You've hit your session limit \u00b7 resets 7:20pm (Europe/Paris)") == "10-06 19:20"
    assert reset("You've hit your session limit \u00b7 resets 1am (Europe/Paris)") == "10-07 01:00"
    weekly = "You've hit your weekly limit \u00b7 resets Oct 12, 8am (Europe/Paris)"
    assert reset(weekly) == "10-12 08:00"
    # An error that arrives just after its own reset time is retried now, not tomorrow.
    assert reset("You've hit your session limit \u00b7 resets 4:10pm (Europe/Paris)") == "10-06 16:10"
    assert reset("You've hit your session limit \u00b7 resets 3pm (Europe/Paris)") == "10-07 15:00"
    assert reset("limit resets 9pm (Nowhere/Zone)") is None
    assert reset("HTTP 500") is None


async def test_the_conversation_home_is_kept_for_a_resumable_task_and_reused(tmp_path) -> None:
    from regact.agent.claude_adapter import ClaudeAgent

    root = tmp_path / "claude-home"
    first = ClaudeAgent({"claude_home": str(root)})
    assert first.resume_token() is None  # no conversation yet
    home = first._config_dir()
    first._session_id = "conversation-1"
    token = first.resume_token()
    assert token == {"session_id": "conversation-1", "home": home}
    first.keep_session = True
    await first.close()
    assert (root / "session").is_dir() and len(list((root / "session").iterdir())) == 1

    second = ClaudeAgent({"claude_home": str(root)})
    second.resume_from(token)
    assert second._config_dir() == home  # the same home, so --resume finds the conversation
    assert second._command("continue")[0][-2:] == ["--resume", "conversation-1"]
    await second.close()  # a final exit: the home is dropped
    assert list((root / "session").iterdir()) == []

    third = ClaudeAgent({"claude_home": str(root)})
    try:
        third.resume_from(token)
    except RuntimeError as error:
        assert "gone" in str(error)
    else:
        raise AssertionError("a missing home must refuse the resume")
