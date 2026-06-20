"""Preview the exact sandbox command regact will build — without executing it.

Run:  make debug D=sec_wrap_preview ARGS="--workdir /tmp/wd --allow src --forbid environnement"
      make debug D=sec_wrap_preview ARGS="--deny-egress --image /path/agent.sif"
Lets you eyeball that the discovered paths are wired into the launcher correctly.
"""
import argparse

from regact.security.runtime import detect, wrap_argv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/tmp/agent_wd")
    p.add_argument("--allow", action="append", default=[])
    p.add_argument("--forbid", action="append", default=[])
    p.add_argument("--deny-egress", action="store_true")
    p.add_argument("--image", default=None)
    a = p.parse_args()
    argv = wrap_argv(detect(), ["echo", "hi"], workdir=a.workdir, allow_read=a.allow,
                     forbid_read=a.forbid, deny_egress=a.deny_egress, image=a.image)
    print(f"detect() -> {detect().value}\n")
    print(" ".join(argv))


if __name__ == "__main__":
    main()
