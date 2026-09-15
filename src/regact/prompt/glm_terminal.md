# Working in the terminal

## Interaction loop

Every iteration you will:
- Think (if you are a thinking / reasoning model).
- Verbalize your reasoning about the task, the environment, and your next step.
- Issue exactly ONE `Bash` tool call whose `command` argument is the shell command to run this turn. It is executed for you and its output comes back on the next iteration.

Your command runs in a FRESH shell at your working directory: no `cd`, environment variable, or virtualenv state carries across commands or iterations. Always reference files by their path relative to the working directory, and use non-interactive flags only (never `vi`, `nano`, or anything that waits for input). Everything you want to run this turn goes inside the one command - chain steps with `&&` or a heredoc if you need several.

## Advice for thinking models

If you are a thinking / reasoning model, verbalize in your answer the conclusions, insights, and decisions you reached while thinking. Your private thinking is NOT carried over to the next iteration - only your visible answer is. If you keep your findings in your thinking alone, every iteration starts blind and you will re-derive (and often repeat) the same reasoning instead of building on it. Treat the visible part of your answer as your working memory: each turn, briefly restate what you now know about the environment, what your current controller does, what worked or failed on the last command, and what you are about to try - so the next iteration continues your progress instead of restarting it.

## Correct answer format

Your answer is your reasoning as plain prose, then exactly ONE `Bash` tool call in exactly this shape, ending at its closing `</tool_call>`:

<tool_call>Bash<arg_key>command</arg_key><arg_value>YOUR SHELL COMMAND</arg_value></tool_call>

The prose is your memory for the next iteration; the call is your action. Any other shape is not executed. For example, your whole answer would be:

From the last run I confirmed the action ids for moving and for interacting with the cell ahead, and that my controller reaches the target but never interacts with it. Next I will fix that step in the controller and re-run the test script.

<tool_call>Bash<arg_key>command</arg_key><arg_value>sed -i 's/return A_MOVE/return A_INTERACT/' code_library/my_controller.py && python code_library/test_controller.py</arg_value></tool_call>

## Typical commands (the `command` value)

Create or overwrite a file:
cat > code_library/explore.py <<'EOF'
from framework.make_env import make_env
# ... your code ...
EOF

Edit a file in place with sed:
sed -i 's/old/new/g' code_library/explore.py                      # replace every occurrence
sed -i '5s/.*/        return obs.available_actions[0]/' solution.py   # rewrite line 5

Read a file, or a slice with line numbers:
cat solution.py
nl -ba solution.py | sed -n '1,40p'

Run a script, or list a directory:
python code_library/explore.py
ls code_library
