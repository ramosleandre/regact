
# Stochastic world model

This environment is random, or hides part of its state, so `step` cannot always know what comes next. This section changes the rules above where they differ.

## Random choices

`step(state, action, choose)` takes a third argument. Wherever the outcome is not decided by the State and the action, ask for it: `value = choose(name, dist)`.

- `name` is a string, unique within one step. The part before `@` is the choice's group: `"cow@3,4"` and `"cow@5,1"` are two uses of the choice `cow`, and statistics are reported per group.
- `dist` is a dictionary `value -> probability` whose probabilities sum to 1, or a list of equally likely values. Values are strings, numbers, booleans or None.
- `choose` returns one value. `step` must be fully decided by its inputs and the values `choose` returns: use no other source of randomness.

A `step` that never calls `choose` is deterministic. `get_initial_state` and `render` are unchanged: they stay deterministic.

## How a recorded step is checked

For each recorded transition, the framework looks for values of your choices under which `render(step(state, action, choose))` equals the recorded next observation exactly. These values are the step's *explanation*, and the State they give is the one carried on. The framework tries the most probable values first, for at most __EXPLAIN_NODES__ runs of `step` per transition.

A transition with no explanation is a `prediction_mismatch`, as before. Its counterexample has an `explanation` field: `choices_given_probability_0` lists choices whose needed value your `dist` excludes, and `search_budget_reached` says the search stopped before trying everything. The predicted observation shown is the one from the most probable values.

## Optional: `world_model/model_explain.py`

`explain(state, action, obs) -> dict` receives the State before the step, the action and the observed next observation, and returns values for some of the step's choices, by name. The framework takes these values as given and searches only the other choices.

Write it when a choice has many possible values and its answer is visible in the observation, for example what appeared in a newly visible cell: the search cannot guess several such choices at once. `explain` cannot make a wrong model pass: its values are replayed through `step` and `render` like any others, and a wrong value only makes the step unexplained. It runs in a separate process from `step`: nothing it computes reaches `step` except the values it returns, each of which must be a value your `dist` allows.

## The cost in bits

An explained step costs the sum of `-log2(probability)` over the values of its choices. The total is the number of bits your model needs to describe what happened: fewer is better.

- A model that forbids something that happens is contradicted.
- A model that allows everything is accepted but pays many bits on every step.
- The best model gives each outcome its real probability.

Bits never refuse a model. An accepted result has a `bits` field:

- `total`, `per_step`: what your model paid on the recorded experience.
- `references_per_step.adaptive_frequencies`: your own explanations priced with frequencies counted as the data came in. Paying more than this means your probabilities are worse than counting.
- `references_per_step.no_model`: a coder that knows nothing about the game and predicts each part of the observation from its previous value.
- `calibration`: every value your choices could take, grouped by the probability you gave it (`declared_up_to` is the top of each group). `expected_to_happen` is the sum of those probabilities and `happened` is how many did happen. The two should be close in every group: `happened` far above means you call things rarer than they are, far below means you spread probability on things that do not happen.
- `by_choice`: per choice group, its `uses`, the `bits` paid, how often each value was `observed` in your explanations, the average probability you `declared` for that value when it happened, and `excess_bits`, what declaring the observed frequencies would have saved. These compare values across uses, so they mean something when a choice's values mean the same thing every time (a direction, a change), not when they depend on the State (a position, an absolute level).
- `largest_surprises`: the most expensive steps.

A choice that costs bits on every step is often not random: it may follow a rule you have not found yet (a counter, a function of time). Modelling it as a choice first and replacing it by a rule later is a normal way to work.

## Acting with a stochastic model

At each real step the framework explains the new observation with your accepted CWM and gives the resulting State to your controller. If the observation cannot be explained, the call stops and you return to the Modeling Phase, as with any contradiction.

**Stopping on a surprise.** A step can be explained and still be a warning: one of its choices took a value your model called very unlikely. The call then stops with `stop_reason: surprise`; the CWM stays accepted, and the `surprise` field names the choice, its value and its probability. You set the threshold, in bits, with an attribute on your controller object: `stop_on_surprise_bits`. A value of `b` stops the call when a choice takes a value of probability below 1 in 2^b. Without the attribute it is __SURPRISE_BITS__ (1 in 2^__SURPRISE_BITS__).

- Lower it (6 to 8) when the controller follows a long plan that is only safe if things go as expected: a surprise then hands control back to you early.
- Raise it (12 to 16) when rare but harmless events keep interrupting, for example a rare thing coming into view while exploring. Giving such events a more realistic probability in the model is the better fix.
- Set it to `None` to never stop on a surprise.

The call's feedback has a `bits` field: what the call's steps cost.

## What the State may hold

The State is what you know, not the whole world. Something you have not seen yet can be absent from it, and its first appearance is then a choice. Remembering what you saw makes its next appearance free, so a State may keep a memory of the world. In this protocol a State may be large: up to __STATE_BYTES__ serialized bytes.

## Optional: `world_model/model_reset.py`

After a reset command a new episode begins. By default its first State is `get_initial_state(first observation)`, which forgets everything. `reset(state, obs) -> State` receives the State the previous episode ended in and the first observation of the new one, and returns the new State: keep in it whatever is still true after a reset. `render` of the State it returns must equal `obs`. It is used only when the previous episode was explained to its end.
