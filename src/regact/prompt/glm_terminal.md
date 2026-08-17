# Working in the terminal

## Interaction loop

Every iteration you will:
- Think (if you are a thinking / reasoning model).
- Verbalize your reasoning about the task, the environment, and your next step.
- Issue exactly ONE `Bash` tool call whose `command` argument is the shell command to run
  this turn. It is executed for you and its output comes back on the next iteration.

Your command runs in a FRESH shell at your working directory: no `cd`, environment
variable, or virtualenv state carries across commands or iterations. Always reference
files by their path relative to the working directory, and use non-interactive flags
only (never `vi`, `nano`, or anything that waits for input). Everything you want to run
this turn goes inside the one command - chain steps with `&&` or a heredoc if you need
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

A well-formed answer is your reasoning as plain prose, then exactly ONE `Bash` tool call
in this EXACT shape, ending at its closing `</tool_call>`:

<tool_call>Bash<arg_key>command</arg_key><arg_value>YOUR SHELL COMMAND</arg_value></tool_call>

The rules that make it run:
- `Bash` comes immediately after `<tool_call>`, with no space.
- The command goes between `<arg_value>` and `</arg_value>`, and it MAY span multiple
  lines (a heredoc is fine).
- Close with `</tool_call>`.

Use no other shape. In particular do NOT write `<tool_call>Bash && <command>`, a
`command:` label, backticks around the command, or a bare `<tool_call>` opener - the only
accepted argument form is `<arg_key>command</arg_key><arg_value>...</arg_value>`. For
example your whole answer would be:

From the last run I confirmed action 2 moves forward and action 3 picks up; my controller
reaches the key but never picks it up. Next I will make it pick up when the key is
adjacent, then re-test on the env.

<tool_call>Bash<arg_key>command</arg_key><arg_value>sed -i 's/return A_TOGGLE/return A_PICKUP/' code_library/doorkey_controller.py && python code_library/test_doorkey.py</arg_value></tool_call>

The reasoning before it is your memory; the tool call is your action. Emit exactly one
`<tool_call>...</tool_call>` per answer and stop at its closing tag.

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

## Special actions through `framework/control.py`

Some framework actions are run through the `framework/control.py` script - each prints its
result (e.g. your score) to stdout. Submit your solution once `solution.py` is correctly
edited (you receive a score so you can check it works as expected), and exit the task once
you are satisfied with your solution and do not want to improve it further. Run them as the
`command` of a `Bash` tool call, exactly like any other shell command:
{control_commands}
