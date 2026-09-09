"""The console silencer must survive a third-party lib that reconfigures its own logger.

Regression: arc_agi's `base`/`scorecard` loggers RE-RAISE their level to INFO and RE-ADD a stdout
handler on every game load, which overwrote a one-shot `setLevel(WARNING)` - so the "Found latest
version of <game>" spam kept reaching the operator's terminal. The fix attaches a level FILTER
(not just a level), which the lib's re-config cannot undo. This test simulates that exact pattern.
"""

import io
import logging

from regact.obs.console import configure_console_logging


def test_silencing_survives_a_lib_reconfiguring_its_logger() -> None:
    configure_console_logging()  # attaches the durable filter to arc_agi.base (among others)

    lg = logging.getLogger("arc_agi.base")
    # Replay arc_agi/base.py on a game load: raise to INFO, drop handlers, add a stdout one.
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    lg.addHandler(handler)

    lg.info("Found latest version of ar25: ar25-0c556536")  # the spam
    lg.error("a genuine env-side error")  # a real error must still come through

    out = buf.getvalue()
    assert "Found latest version" not in out  # INFO dropped despite the lib re-raising the level
    assert "a genuine env-side error" in out  # WARNING+ still reaches the terminal


def test_configure_is_idempotent_no_duplicate_filters() -> None:
    """Called once per experiment, but tests reuse the process; reconfiguring must not stack a new
    filter each time (which would still work, but leak)."""
    from regact.obs.console import _MinLevelFilter

    configure_console_logging()
    configure_console_logging()
    got = [f for f in logging.getLogger("arc_agi.base").filters if isinstance(f, _MinLevelFilter)]
    assert len(got) == 1


def test_agent_decision_log_reaches_run_log_without_readmitting_http_noise(tmp_path) -> None:
    """A whole benchmark ran with the agent loop's INFO decisions discarded: regact configures no
    root handler, so they hit Python's lastResort floor of WARNING. The absence of a line then
    proved nothing about whether the event happened."""
    import logging

    from regact.obs.console import configure_console_logging

    run_log = tmp_path / "run.log"
    configure_console_logging(str(run_log))
    try:
        logging.getLogger("alancode.query.loop").info("Escalating max_tokens to %d", 12000)
        logging.getLogger("httpx").info("POST /v1/chat/completions 200 OK")
    finally:
        configure_console_logging(None)  # detaches before the file closes

    written = run_log.read_text()
    assert "Escalating max_tokens to 12000" in written
    assert "httpx" not in written  # _INFO_NOISY must stay suppressed


def test_reconfiguring_does_not_leave_a_handler_on_a_closed_file(tmp_path) -> None:
    """Tests and repeated experiments reuse the process; a stale handler would write to a closed
    file on the next run."""
    import logging

    from regact.obs.console import configure_console_logging

    configure_console_logging(str(tmp_path / "first.log"))
    configure_console_logging(str(tmp_path / "second.log"))
    try:
        logging.getLogger("alancode").info("after reconfigure")
    finally:
        configure_console_logging(None)

    assert "after reconfigure" in (tmp_path / "second.log").read_text()
    assert "after reconfigure" not in (tmp_path / "first.log").read_text()
    assert len(logging.getLogger("alancode").handlers) == 0
