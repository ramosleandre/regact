"""Smoke test: the package imports and exposes a version string."""

import regact


def test_version() -> None:
    assert isinstance(regact.__version__, str)
    assert regact.__version__
