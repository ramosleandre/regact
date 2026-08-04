"""Unit tests: the readiness report.

Cover what regact owns — the rows produced and how they render — without asserting
anything about which CLIs happen to be installed on the machine running the tests.
"""

from __future__ import annotations

from regact.doctor import FAIL, OK, SKIP, WARN, Row, collect, format_report


def test_collect_covers_every_section() -> None:
    sections = {row.section for row in collect()}
    assert {"core", "sandbox", "agent backends", "game extras"} <= sections


def test_core_reports_the_interpreter_and_the_package() -> None:
    core = [row for row in collect() if row.section == "core"]
    # The suite runs under a supported interpreter with regact importable, so both pass.
    assert all(row.status == OK for row in core), core
    assert any("regact" in row.detail for row in core)


def test_every_agent_backend_is_reported() -> None:
    """A backend must never be silently missing from the report — an in-process one is
    listed as skipped rather than omitted."""
    from regact.config.schema import AgentName

    reported = {row.name for row in collect() if row.section == "agent backends"}
    assert reported == {name.value for name in AgentName}


def test_in_process_backends_are_skipped_not_failed() -> None:
    rows = {r.name: r for r in collect() if r.section == "agent backends"}
    assert rows["scripted"].status == SKIP
    assert rows["alan"].status != SKIP  # subprocess backend: actually probed, never skipped


def test_endpoint_is_only_checked_when_asked() -> None:
    assert not [row for row in collect() if row.section == "endpoint"]
    # A closed port must warn, never crash or fail the run.
    row = next(r for r in collect(endpoint="http://127.0.0.1:1/v1") if r.section == "endpoint")
    assert row.status == WARN and "unreachable" in row.detail


def test_report_summarizes_failures() -> None:
    rows = [Row("core", "python", FAIL, "3.13"), Row("core", "regact importable", OK, "/x")]
    out = format_report(rows)
    assert "1 core check(s) failed" in out
    assert "python" in out and "3.13" in out


def test_report_is_green_without_failures() -> None:
    assert "core is ready" in format_report([Row("core", "python", OK, "3.12")])
