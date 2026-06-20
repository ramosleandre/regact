#!/usr/bin/env python3
"""probe.py -- sonde de conformance du sandbox (squelette).

Tente le catalogue d'attaque DEPUIS le process ou elle tourne et verifie les
invariants R1-R6. On la lance NUE (vulnerable) puis DANS une piece (defendue):

    python3 probe.py                              # nu    -> VULNERABLE
    sandbox-exec -p "$PROFILE" python3 probe.py   # piece -> DEFENDU

  want=deny  => l'attaque DOIT echouer (sinon vulnerable)
  want=allow => l'usage legitime DOIT marcher (sinon on a casse l'agent)
"""
import json
import os
import socket
import sys
import tempfile

CHECKS = []  # (rid, name, want, ok, detail)


def record(rid, name, want, attack_succeeded, detail):
    ok = (not attack_succeeded) if want == "deny" else attack_succeeded
    CHECKS.append((rid, name, want, ok, detail))


def can_read(path):
    try:
        n = len(open(path, "rb").read(32))
        return True, "lu %do" % n
    except Exception as e:
        return False, type(e).__name__


def can_connect(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close()
        return True, "connecte"
    except Exception as e:
        return False, type(e).__name__


def main():
    workdir = tempfile.mkdtemp(prefix="agent_workdir_")
    open(os.path.join(workdir, "solution.py"), "w").write("# le travail de l'agent\n")
    secret = os.environ.get("PROBE_SECRET")
    if not secret:
        fd, secret = tempfile.mkstemp(suffix="_ar25.py")
        os.write(fd, b"WINNING = [3, 1, 2, 0]\n")
        os.close(fd)

    # R1 -- l'agent DOIT pouvoir bosser (sinon on l'a casse)
    ok, d = can_read(os.path.join(workdir, "solution.py"))
    record("R1", "lire son propre workdir", "allow", ok, d)

    # R2 -- lire le jeu / remonter / enumerer DOIT echouer
    ok, d = can_read(secret)
    record("R2", "A1  open(secret du jeu)", "deny", ok, d)
    try:
        sib = os.listdir(os.path.dirname(workdir))
        ok, d = True, "%d voisins vus" % len(sib)
    except Exception as e:
        ok, d = False, type(e).__name__
    record("R2", "A8  lister le dossier parent (autres exp.)", "deny", ok, d)
    ok, d = can_read("/etc/passwd")
    record("R2", "A1  lire un chemin absolu hors allowlist", "deny", ok, d)

    # R5 -- sortir sur internet externe DOIT echouer (runs notes)
    ok, d = can_connect("example.com", 443)
    record("R5", "E1  egress internet externe", "deny", ok, d)

    print("%-4s %-44s %-6s %s" % ("INV", "CHECK", "WANT", "VERDICT"))
    print("-" * 80)
    vuln = 0
    for rid, name, want, ok, detail in CHECKS:
        verdict = "DEFENDU" if ok else "*** VULNERABLE ***"
        vuln += 0 if ok else 1
        print("%-4s %-44s %-6s %-18s (%s)" % (rid, name, want, verdict, detail))
    print("-" * 80)
    print("GLOBAL:", "TOUT DEFENDU" if vuln == 0 else "%d VULNERABILITE(S)" % vuln)
    if "--json" in sys.argv:
        print(json.dumps([dict(zip(("rid", "name", "want", "ok", "detail"), c)) for c in CHECKS]))


if __name__ == "__main__":
    main()
