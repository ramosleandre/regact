"""``python -m regact.doctor`` — can this machine run regact?

A quick, dependency-free readiness check that prints one ``[ ok ]`` / ``[warn]`` /
``[fail]`` line per capability, so a new user learns what works and what to install
BEFORE launching a run. It never raises: a missing optional piece is a ``warn``, only
a broken core (wrong Python, package not importable) is a ``fail``.

Distinct from ``regact.security.probe``: the probe deeply *attacks* the OS sandbox to
verify the R1-R6 isolation contract; doctor just reports what is present. Run the probe
(``make probe``) for the hardened check; run doctor first to see the lay of the land.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys

_OK = "\033[32m[ ok ]\033[0m"
_WARN = "\033[33m[warn]\033[0m"
_FAIL = "\033[31m[fail]\033[0m"


def _line(status: str, name: str, detail: str = "") -> None:
    print(f"  {status} {name:<22} {detail}")


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check() -> int:
    """Print the readiness report; return the number of hard failures."""
    fails = 0
    print("regact doctor — machine readiness\n")

    # ── core ──
    print("core")
    py_ok = sys.version_info[:2] in ((3, 11), (3, 12))
    _line(_OK if py_ok else _FAIL, "python 3.11 / 3.12", f"found {sys.version.split()[0]}")
    fails += 0 if py_ok else 1

    pkg_ok = _module_present("regact")
    _line(
        _OK if pkg_ok else _FAIL,
        "regact importable",
        "" if pkg_ok else "run `pip install -e .` (or prefix PYTHONPATH=src)",
    )
    fails += 0 if pkg_ok else 1

    # ── OS sandbox backend ──
    print("\nsandbox (optional — runs unconfined without one)")
    backends = {
        "sandbox-exec": "macOS Seatbelt",
        "bwrap": "Linux bubblewrap",
        "apptainer": "HPC Apptainer",
        "singularity": "HPC Singularity",
    }
    found_backend = False
    for binary, label in backends.items():
        if shutil.which(binary):
            _line(_OK, binary, label)
            found_backend = True
    if not found_backend:
        _line(_WARN, "no sandbox backend", "runs with security.sandbox=none only")

    # ── agent CLIs (optional — pick the one you use) ──
    print("\nagent backends (optional — install the one you run)")
    for binary, label in (("claude", "Claude Code CLI"), ("codex", "codex CLI")):
        path = shutil.which(binary)
        _line(_OK if path else _WARN, binary, label if path else f"not on PATH ({label})")
    _line(_OK if _module_present("alancode") else _WARN, "alancode", "in-process Alan backend")

    # ── game extras (optional — install the problem you run) ──
    print("\ngame extras (optional — install the problem you run)")
    _line(
        _OK if _module_present("arc_agi") else _WARN,
        "arc_agi",
        'ARC-AGI-3  (pip install ".[arc]")',
    )
    _line(
        _OK if _module_present("minigrid") else _WARN,
        "minigrid",
        'MiniGrid  (pip install ".[minigrid]")',
    )

    print()
    if fails:
        print(f"{_FAIL} {fails} core check(s) failed — fix these before running.")
    else:
        print(f"{_OK} core is ready. Warnings above are optional pieces you can install as needed.")
    return fails


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="regact.doctor", description=__doc__).parse_args(argv)
    return 1 if check() else 0


if __name__ == "__main__":
    raise SystemExit(main())
