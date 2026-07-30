"""Shared test fixtures."""

import contextlib

import pytest

from regact.envclient.client import EnvClient


@pytest.fixture(autouse=True)
def _closed_env_clients(monkeypatch: pytest.MonkeyPatch):
    """Close every EnvClient a test builds.

    Test helpers hand out bare ``EnvClient(TestClient(app), ...)`` objects; unclosed,
    each request spins its own AnyIO portal and the clients leak across the run.
    Tracking construction here closes them all without each test managing lifetime.
    """
    created: list[EnvClient] = []
    orig_init = EnvClient.__init__

    def _tracking_init(self: EnvClient, *args: object, **kwargs: object) -> None:
        orig_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(EnvClient, "__init__", _tracking_init)
    yield
    for client in created:
        with contextlib.suppress(Exception):
            client.close()
