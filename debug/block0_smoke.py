"""Block 0 manual smoke: the package imports and the tooling is wired.

Run:  python debug/block0_smoke.py
Then: make check   (ruff + mypy + pytest, all green)
"""

import regact


def main() -> None:
    print(f"regact {regact.__version__} imported OK — scaffold is alive.")
    print("Run `make check` for lint + typecheck + tests.")


if __name__ == "__main__":
    main()
