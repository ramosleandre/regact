# Working in the terminal

## Interaction loop

Every iteration you will:
- Think (if you are a thinking / reasoning model).
- Verbalize your reasoning about the task, the environment, and your next step.
- Issue exactly ONE bash command, written inside a single ```bash code block. It is
  extracted from your answer and executed for you.
- Receive that command's output, then continue on the next iteration.

Your command runs in a FRESH shell at your working directory: no `cd`, environment
variable, or virtualenv state carries across commands or iterations. Always reference
files by their path relative to the working directory, and use non-interactive flags
only (never `vi`, `nano`, or anything that waits for input). Everything you want to run
this turn goes inside the one block - chain steps with `&&` or a heredoc if you need
several.

## Advice for thinking models

If you are a thinking / reasoning model, verbalize in your answer the conclusions,
insights, and decisions you reached while thinking. Your private thinking is NOT carried
over to the next iteration - only your visible answer is. If you keep your findings in
your thinking alone, every iteration starts blind and you will re-derive (and often
repeat) the same reasoning instead of building on it. Treat the visible part of your
answer as your working memory: each turn, briefly restate what you now know about the
environment, what your current controller does, what worked or failed on the last
command, and what you are about to try - so the next iteration continues your progress
instead of restarting it.

## Correct answer format

A well-formed answer is your reasoning as plain prose, then exactly ONE fenced bash block.
Do not wrap it in any tag or envelope - just the prose, then the block. For example your
whole answer would be:

From the last run I confirmed `obs.available_actions` is `[0..6]`, that action 2 moves
forward and action 5 toggles a door, and that my controller reaches the key but never
picks it up - the pickup action is 3, not 5. Next I will make the controller pick up
the key when it is adjacent, then re-test on the env.

```bash
sed -i 's/return A_TOGGLE/return A_PICKUP/' code_library/doorkey_controller.py && python code_library/test_doorkey.py
```

The reasoning before it is your memory; the block is your action. Write only the prose and
the one fenced block - no surrounding tag or envelope around them - and always end the block
with its closing fence line, or it will not run.

## Typical commands

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

## Special actions through `framework/control.py`

Some framework actions are run through the `framework/control.py` script - each prints its
result (e.g. your score) to stdout. Submit your solution once `solution.py` is correctly
edited (you receive a score so you can check it works as expected), and exit the task once
you are satisfied with your solution and do not want to improve it further:
```bash
{control_commands}
```
