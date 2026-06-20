"""Run + interpret the sandbox conformance probe on this machine.

Run:  make debug D=sec_probe ARGS="--sandbox"          (confined; ALL DEFENDED on linux/hpc)
      make debug D=sec_probe ARGS="--sandbox --json"   (machine-readable)
      make debug D=sec_probe                            (baseline, no sandbox = vulnerable)
Thin wrapper over `python -m regact.security.probe`; prints the picked backend first.
"""
import subprocess
import sys

from regact.security.runtime import detect


def main() -> None:
    print(f"detect() -> {detect().value}\n")
    cmd = [sys.executable, "-m", "regact.security.probe", *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
