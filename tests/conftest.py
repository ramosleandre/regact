"""Shared test fixtures."""

import contextlib

import pytest

from regact.envclient.client import EnvClient


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-costly",
        action="store_true",
        default=False,
        help="run costly tests that call REAL agents (spends credits / needs a served model).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect ``costly`` tests unless ``--run-costly`` is passed - a HARD gate, so a real-agent
    smoke can never run (and never spend credits) in CI or a plain ``pytest``."""
    if config.getoption("--run-costly"):
        return
    skip = pytest.mark.skip(reason="costly: pass --run-costly to run (real agent, spends credits).")
    for item in items:
        if "costly" in item.keywords:
            item.add_marker(skip)


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
