## Working in the terminal

You act only through the Bash tool: every file operation is a shell command. Each Bash
call runs in a fresh shell at your working directory (no `cd`, environment, or venv state
carries across calls), so reference files by their relative path and use non-interactive
flags (avoid vi, nano, or anything that waits for input).

Useful commands:

Create or overwrite a file:
```bash
cat > code_library/explore.py <<'EOF'
from framework.make_env import make_env
# ... your code ...
EOF
```

Edit a file in place with sed:
```bash
sed -i 's/old/new/g' code_library/explore.py                      # replace every occurrence
sed -i '5s/.*/        return obs.available_actions[0]/' solution.py   # rewrite line 5
```

Read a file, or a slice with line numbers:
```bash
cat solution.py
nl -ba solution.py | sed -n '1,40p'
```

Run a script, or list a directory:
```bash
python code_library/explore.py
ls code_library
```

Submit your solution (this prints your score) or end the task:
```bash
{control_commands}
```
