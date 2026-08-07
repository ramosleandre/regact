### Code World Model

Write your understanding of the environment down **as a program**, and check it
against recorded reality — instead of keeping it implicit in your head.

Everything that happens in the environment is recorded for you: every real
transition `(o, a, r, o', done)` lands in `data/transitions.jsonl` (one JSON
line each; obs stored exactly as your code receives them). You cannot alter
this data — it is ground truth. Read it freely.

Your world model lives in `world_model/` — one model, edited in place:

- `model_state.py` — a `State` class: what the world IS (objects, positions,
  counters), designed by you.
- `model_parser.py` — `parse(obs) -> State`: read one obs dict into your state.
- `model_render.py` — `render(state) -> obs dict`: rebuild the obs **exactly**
  as observed, same format, every field.
- `model_transition.py` - `step(state, action) -> (State, reward, done)`: how
  the world MOVES - the rule that turns one state into the next.
- `model_notes.py` — your interpretation, open questions, known flaws (free
  text in the docstring).
- Add any helper `.py` files you want; keep all world-model code in this folder.

**Verify after every meaningful edit**: run `python world_model/verify.py`.
It reports **coherence** — the fraction of recorded observations where
`render(parse(o)) == o` exactly - and **transition accuracy** - the fraction
of recorded `(o, a)` pairs where `render(step(parse(o), a))` reproduces the
observed next obs, reward and done exactly - plus your model's code complexity
and the exact transitions where the model fails. A failure is information, not
noise: load the pointed transition (`from verify import load_transitions`),
study why your model cannot explain it, and revise. When a transition refuses
to fit, suspect the representation too - the two models are one theory.

Work in that order: ground the representation first (`step` may keep raising
until then - verify reports transition accuracy as n/a, not as failures).
Then discover the rule. When several rules fit the recorded history, prefer
the real-world experiment whose outcome discriminates between them.

Prefer **small** models: a `State` that compresses the obs into few meaningful
variables, with small code. A state that merely copies the obs dict reaches
100% coherence having understood nothing - and then `step` is unwritable,
because predicting raw pixels is the whole problem again. A compressed,
meaningful state is exactly what makes the transition rule short.

The payoff: a model that passes both checks is a **certified simulator** of
the game. Your controller can `import world_model` and plan inside it -
search action sequences, evaluate outcomes - at zero real-action cost, and
only then act. That is what this tool is for.
