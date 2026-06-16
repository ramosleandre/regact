"""Tests for the error/log taxonomy."""

from regact.obs.errors import ErrorCategory, LogComponent, LogRecord, RegactError


def test_enum_values() -> None:
    assert ErrorCategory.AGENT_API.value == "agent_api"
    assert LogComponent.ENV_SERVER.value == "env_server"


def test_log_record_to_json() -> None:
    rec = LogRecord(
        ts="2026-01-01T00:00:00",
        component=LogComponent.EVAL,
        level="ERROR",
        event="boom",
        error_category=ErrorCategory.ENV_RUNTIME,
        detail={"k": 1},
    )
    out = rec.to_json()
    assert out["component"] == "eval"
    assert out["error_category"] == "env_runtime"
    assert out["task"] is None
    assert out["detail"] == {"k": 1}


def test_log_record_no_error_category() -> None:
    rec = LogRecord(ts="t", component=LogComponent.AGENT, level="INFO", event="hi")
    assert rec.to_json()["error_category"] is None


def test_regact_error_category() -> None:
    err = RegactError(ErrorCategory.LOOP_CRASH, "kaboom")
    assert err.category is ErrorCategory.LOOP_CRASH
    assert str(err) == "kaboom"


def test_regact_error_cause() -> None:
    cause = ValueError("root")
    err = RegactError(ErrorCategory.ENV_RUNTIME, "wrapped", cause=cause)
    assert err.__cause__ is cause
