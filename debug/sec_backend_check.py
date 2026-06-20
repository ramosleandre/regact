"""Which OS sandbox will regact pick on THIS machine / compute node?

Run:  make debug D=sec_backend_check
Prints the facts detect() keys on, then its decision. Run it on a fresh Linux box
or an HPC compute node BEFORE a real run, to know if you'll get bwrap/apptainer/none.
"""
import os
import shutil
import subprocess
import sys

from regact.security.runtime import detect


def _userns_ok() -> bool:
    try:
        r = subprocess.run(["unshare", "-Urm", "true"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> None:
    print(f"platform               : {sys.platform}")
    for tool in ("sandbox-exec", "bwrap", "apptainer", "singularity"):
        print(f"which {tool:<16}: {shutil.which(tool) or '-- (absent)'}")
    print(f"unshare -Urm true      : {'ok' if _userns_ok() else 'BLOCKED (no bwrap)'}")
    print(f"SINGULARITY_ALLOWED_DIR: {os.environ.get('SINGULARITY_ALLOWED_DIR', 'unset')}")
    print(f"=> detect()            : {detect().value}")


if __name__ == "__main__":
    main()
