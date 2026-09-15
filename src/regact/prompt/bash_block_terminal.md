# Working in the terminal

## Interaction loop

Every iteration you will:
- Think (if you are a thinking / reasoning model).
- Verbalize your reasoning about the task, the environment, and your next step.
- Issue exactly ONE bash command, inside a single ```bash code block. It is extracted from your answer and executed, and its output comes back on the next iteration.

Your command runs in a FRESH shell at your working directory: no `cd`, environment variable, or virtualenv state carries across commands or iterations. Always reference files by their path relative to the working directory, and use non-interactive flags only (never `vi`, `nano`, or anything that waits for input). Everything you want to run this turn goes inside the one block - chain steps with `&&` or a heredoc if you need several.

## Advice for thinking models

If you are a thinking / reasoning model, verbalize in your answer the conclusions, insights, and decisions you reached while thinking. Your private thinking is NOT carried over to the next iteration - only your visible answer is. If you keep your findings in your thinking alone, every iteration starts blind and you will re-derive (and often repeat) the same reasoning instead of building on it. Treat the visible part of your answer as your working memory: each turn, briefly restate what you now know about the environment, what your current controller does, what worked or failed on the last command, and what you are about to try - so the next iteration continues your progress instead of restarting it.

## Correct answer format

Your answer is your reasoning as plain prose, then exactly ONE fenced bash block, closed with its fence line. The prose is your memory for the next iteration; the block is your action. Only a block tagged `bash` is executed. For example, your whole answer would be:

From the last run I confirmed the action ids for moving and for interacting with the cell ahead, and that my controller reaches the target but never interacts with it. Next I will fix that step in the controller and re-run the test script.

```bash
sed -i 's/return A_MOVE/return A_INTERACT/' code_library/my_controller.py && python code_library/test_controller.py
```

## Typical commands

Create or overwrite a file and run it in the same command (a bare `cat > ... <<'EOF'` prints nothing, so on its own you cannot tell whether the file was written):
```bash
cat > code_library/explore.py <<'EOF'
from framework.make_env import make_env
# ... your code ...
EOF
python code_library/explore.py
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
