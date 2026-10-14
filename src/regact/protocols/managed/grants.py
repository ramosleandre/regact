"""One framework command of each name per agent tool call.

The agent's tool calls are the budget a run is compared on, so a shell loop or a
background process must not issue many commands under one call. Each tool call the
agent starts grants one command of each name; the control channel claims a grant
before executing. A refusal is never the default: with no evidence of which call a
command belongs to, the command runs.
"""

from __future__ import annotations

import asyncio
import threading
import time

# A command can reach the control channel before the loop has read the start of the
# tool call that issued it (two independent streams), so a claim waits this long for
# a new call before it is refused.
GRANT_WAIT_SECONDS = 5.0
GRANT_WAIT_LOGGED_SECONDS = 0.5  # a grant that took this long is recorded, to audit the wait
_POLL_SECONDS = 0.02


class CommandGrants:
    def __init__(self, wait_seconds: float = GRANT_WAIT_SECONDS) -> None:
        self.wait_seconds = wait_seconds
        self._lock = threading.Lock()
        self._open: dict[str, set[str]] = {}  # tool call id -> command names it issued
        # The latest finished call keeps its unused grants until the next call starts:
        # a command sent to the background arrives after its own call has returned.
        self._last: set[str] | None = None
        self._seen = False  # False until the agent's stream showed one tool call
        self._requests: set[str] = set()  # request ids already granted (client retries)

    def call_started(self, call_id: str) -> None:
        with self._lock:
            self._seen = True
            self._open[call_id] = set()
            self._last = None

    def call_finished(self, call_id: str) -> None:
        with self._lock:
            used = self._open.pop(call_id, None)
            if used is not None:
                self._last = used

    def _take(self, name: str, request_id: str | None) -> str | None:
        """"granted", "unattributed" (no tool call ever seen: let through), or None."""
        with self._lock:
            if request_id is not None and request_id in self._requests:
                return "granted"
            if not self._seen:
                return "unattributed"
            holders = [*self._open.values()]
            if self._last is not None:
                holders.append(self._last)
            for used in holders:
                if name not in used:
                    used.add(name)
                    if request_id is not None:
                        self._requests.add(request_id)
                    return "granted"
            return None

    async def claim(self, name: str, request_id: str | None = None) -> tuple[str, float]:
        """(outcome, seconds waited); outcome is "granted", "unattributed" or "refused"."""
        start = time.monotonic()
        while True:
            outcome = self._take(name, request_id)
            waited = time.monotonic() - start
            if outcome is not None:
                return outcome, waited
            if waited >= self.wait_seconds:
                return "refused", waited
            await asyncio.sleep(_POLL_SECONDS)
