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
- `model_notes.py` — your interpretation, open questions, known flaws (free
  text in the docstring).
- Add any helper `.py` files you want; keep all world-model code in this folder.

**Verify after every meaningful edit**: run `python world_model/verify.py`.
It reports **coherence** — the fraction of recorded observations where
`render(parse(o)) == o` exactly — plus your model's code complexity and the
exact transitions where the model fails. A failure is information, not noise:
load the pointed transition (`from verify import load_transitions`), study why
your representation cannot reproduce it, and revise.

Prefer **small** models: a `State` that compresses the obs into few meaningful
variables, with small `parse`/`render` code. A model at 100% coherence that
merely copies the obs dict has understood nothing — compression is the point.
A model at 100% coherence with a compact state means you have grounded what
the pixels ARE; that grounding is what your controller should be built on.
Couple this tool with your understanding of what the environment is really about.
