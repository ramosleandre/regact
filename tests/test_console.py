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
