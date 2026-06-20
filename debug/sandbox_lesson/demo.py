"""Lecon: in-process vs subprocess vs la "piece" (sandbox OS).

Lance:  python3 debug/sandbox_lesson/demo.py
Rien n'est cache ici: on OBSERVE le comportement reel, ligne a ligne.
"""

import os
import subprocess
import sys
import tempfile


def run(argv):
    """Lance un AUTRE process et attend qu'il finisse (flush pour garder l'ordre)."""
    sys.stdout.flush()
    return subprocess.run(argv, capture_output=False)


# --- un "secret" qui vit UNIQUEMENT dans la RAM de CE process (le parent) ---
SECRET_IN_MEMORY = "answer=42 (vit dans la RAM du parent)"

# --- un "secret" pose sur le DISQUE (comme la source d'un jeu) ---
f = tempfile.NamedTemporaryFile("w", suffix="_game.py", delete=False)
f.write("WINNING_MOVES = [3, 1, 2, 0]  # la reponse du jeu, sur le disque\n")
f.close()
SECRET_FILE = os.path.realpath(f.name)  # chemin canonique (/var -> /private/var)
print("(secret sur disque: %s)\n" % SECRET_FILE)

print("== 1) IN-PROCESS : on execute le code de l'agent DANS notre process ==")
# le code partage NOTRE espace memoire (nos variables globales)
agent = compile("print('   [agent in-process] je lis la RAM du parent :', SECRET_IN_MEMORY)", "<agent>", "exec")
eval(agent, globals())
print()

print("== 2) SUBPROCESS : on execute le code de l'agent dans un AUTRE process ==")
child = (
    "try:\n"
    "    print('   [agent subprocess] secret parent =', SECRET_IN_MEMORY)\n"
    "except NameError as e:\n"
    "    print('   [agent subprocess] je NE vois PAS la RAM du parent :', repr(e))\n"
)
run([sys.executable, "-c", child])
print()

print("== 3) CRASH ==")
print("   in-process : si l'agent fait os._exit(), NOTRE programme meurt aussi")
print("                (on ne le lance pas, ca tuerait la demo)")
r = run([sys.executable, "-c", "import os; os._exit(7)"])
print("   subprocess : l'agent a crashe, code=%d, et NOUS sommes TOUJOURS vivants" % r.returncode)
print()

print("== 4) LE POINT CLE : un subprocess NE cache PAS les fichiers ==")
child = "print('   [agent subprocess] je lis le fichier secret :', open(%r).read().strip())" % SECRET_FILE
run([sys.executable, "-c", child])
print("   -> AUTRE process, MEME utilisateur, il lit QUAND MEME le secret. subprocess != octets caches.")
print()

print("== 5) LA PIECE (macOS sandbox-exec) : l'OS REFUSE la lecture par REGLE ==")
# Profil: tout permis SAUF lire ce fichier precis. Le fichier EXISTE, le chemin est CONNU,
# mais le noyau refuse la lecture -> c'est ca "donner des droits OS a un process".
profile = '(version 1)(allow default)(deny file-read* (literal "%s"))' % SECRET_FILE
code = "open(%r).read(); print('LU')" % SECRET_FILE
sys.stdout.flush()
res = subprocess.run(["sandbox-exec", "-p", profile, sys.executable, "-c", code], capture_output=True, text=True)
if res.returncode == 0:
    print("   (sandbox-exec a laisse passer -- inattendu)")
else:
    lines = res.stderr.strip().splitlines()
    print("   [agent dans la piece] lecture REFUSEE par l'OS :", lines[-1] if lines else "(deny)")
print("   -> meme chemin connu, meme utilisateur : l'OS bloque. La 'piece' = des droits, pas une cachette.")

os.unlink(SECRET_FILE)
