"""Stop a running experiment, or one of its tasks, from another terminal.

    python -m regact.stop <run directory>                 # every task of the launch
    python -m regact.stop <run directory> --task ls20     # one task (its folder under the run)
    python -m regact.stop <run directory> --force         # do not wait for in-flight tools

The running launch sees the request within a few seconds: each task finishes the tool it is
executing, ends with ``interrupted`` and keeps its agent's conversation, and tasks not yet
started are left untouched. Continue later with the same command plus ``resume=<run directory>``.
"""

from __future__ import annotations

import argparse
import os

from regact.orchestration.signals import request_stop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regact.stop", description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="the launch's timestamped run directory")
    parser.add_argument("--task", help="stop only this task (its folder name under the run)")
    parser.add_argument("--force", action="store_true", help="do not wait for in-flight tools")
    args = parser.parse_args(argv)
    target = os.path.join(args.run_dir, args.task) if args.task else args.run_dir
    if not os.path.isdir(target):
        parser.error(f"no such directory: {target}")
    print(f"stop requested: {request_stop(target, force=args.force)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
