# CWM v4 - design plan

Branch: `cwm_v4`. Status: PLAN, nothing implemented. Written against `main` at 2ab93fb.

CWM v4 turns the world-model feature from "a scaffold plus a verify script the agent may run"
into a **phase-enforced protocol**: gather, model, explore. The agent cannot progress without a
world model that passes machine-checked coherence, and cannot touch the environment except
through an exploration controller its own model predicts will find something new.

Two measurements below are BLOCKING - they change what is buildable, not just how. They are at
the top for that reason.

---

## 1. Blocking finding: v4 requires a fully observable environment

The v4 loop assumes "deterministic, fully observable" (phase 2 concludes that any env/CWM
disagreement is a CWM defect). I probed that assumption on the actual MiniGrid tasks rather than
assuming it - 2000 random steps per configuration, hashing exactly what the agent sees
(`frame` + `available_actions`):

| task | fully_obs | (obs,action) keys | non-deterministic |
|---|---|---:|---:|
| Empty-16x16 | **True** | 1198 | **0 (0.0%)** |
| Empty-16x16 | **False** | 562 | **53 (9.4%)** |
| DoorKey-8x8 | **True** | 1046 | **0 (0.0%)** |
| DoorKey-8x8 | **False** | 943 | **29 (3.1%)** |

And the second diagnostic, on partially observable runs:

| task (fully_obs=False) | obs-determinism | history-determinism |
|---|---|---|
| Empty-16x16 | 53 / 562 violations | **0 / 2000** |
| DoorKey-8x8 | 29 / 943 violations | **0 / 1997** |
| FourRooms | 27 / 803 violations | **0 / 1975** |

**Consequences.**

1. **The current benchmark setting is incompatible with v4.** Every arm of
   bench-minigrid-02 and -r2 runs `problem.kwargs.fully_obs=false` (the `-po` arms). Under it,
   3-9% of `(obs, action)` pairs lead to two different next observations, so v4's
   transition-coherence check is **unsatisfiable by construction** - the agent could never leave
   phase 1, whatever it writes. v4 runs must set `fully_obs=true`.
2. **Your obs/history distinction is exactly the right diagnostic, and it works.** Where
   obs-determinism fails, history-determinism holds perfectly. So the framework can *tell the two
   apart automatically*: obs-det fails + history-det holds = **deterministic but partially
   observable**; both fail = **stochastic or leaking state**.
3. **Answering "is the framework compromised?"** - yes, and differently per case:
   - *obs-det fails, history-det holds*: v4's contract is unsatisfiable, the agent is stuck in
     phase 1 forever. **Exit the task** with `exit_reason=partially_observable`. This is not a
     bug; it is the wrong env for this protocol. (A history-conditioned CWM is a coherent future
     variant - v5 - but it is a different and much harder benchmark.)
   - *both fail*: the env is stochastic or has hidden mutable state. **Exit** with
     `exit_reason=nondeterministic_env`, and investigate: this would also invalidate the
     controller scoring, since two runs of one controller need not agree.
   - Either way, exiting immediately is right: the alternative is burning a full walltime on a
     task whose success condition cannot be met.

**Recommendation:** run the determinism probe as a **pre-flight**, once per (task, seed), before
the agent starts, and refuse to launch the task rather than discovering it at turn 200.

---

## 2. Blocking finding: `max_turns` is not one concept across agents

You asked whether a turn means the same thing for Alan and the CLI agents, and how many they
use. Measured across 658 runs in `experiments/`:

| agent | runs | median turns | median iterations | median tool calls | iterations/turn |
|---|---:|---:|---:|---:|---:|
| alan | 606 | 4 | 23 | 14 | **5.8** |
| codex | 45 | 2 | 1 | **44** | 0.5 |
| claude | 7 | 3 | 2 | **180** | 0.7 |

A regact **turn** is one `send()`. For Alan that is one keep-alive cycle containing ~6 model
completions. For a CLI agent the harness hands the whole task to the CLI, which runs *its own*
agentic loop to completion inside that single `send()` - Claude does a median of **180 tool calls
in 3 turns**.

**Consequences.**

1. `limits.max_turns=350` is meaningless for CLI agents; they finish in 2-3. The binding limit
   for them is `max_tool_calls` or walltime. Any v4 budget must be expressed in **tool calls**
   (or env steps), never turns.
2. **The phase machine cannot live in the orchestration loop.** The loop only sees turn
   boundaries, and a CLI agent would pass through all three phases inside one turn without the
   loop ever regaining control. The phase gate must be enforced **where the agent acts** - at the
   HTTP boundary, on every env and control call.

That single constraint drives most of the architecture below, and it is why v4 is *more*
server-side than v2 rather than less.

---

## 3. Phase machine

State lives framework-side in a `PhaseState` object owned by the run, mutated only by the server
at tool/env boundaries, and mirrored into `experiment_state.json` for viz.

```
PHASE 0  GATHER          env: ALLOWED        exit: >= X distinct observations
PHASE 1  MODEL           env: REFUSED        exit: UpdateCodeWorldModel passes all checks
PHASE 2  EXPLORE         env: INDIRECT only  exit: never (loop 2 -> 2), or back to 1 on incoherence
```

- **X** is `features.cwm.min_unique_obs` (default 10).
- Phase 0 -> 1 is evaluated **after** each env call: the call that satisfies the threshold
  completes normally, and the phase flips before the next one. ("The first tool call after which
  this condition is satisfied enters phase 1.")
- Phase 1 -> 2 only via `UpdateCodeWorldModel` returning success.
- Phase 2 -> 1 only when a real-env rollout contradicts the CWM.
- Phase 2 -> 2 on a successful exploration.

Every phase transition emits a `PhaseTransition` event into the transcript, so viz can offer the
same **"Jump to"** affordance already used for flagged calls and submissions (remark 98).

**Enforcement.** Each of `EnvServer`'s env routes and each framework tool declares the phases it
is legal in. An illegal call is not an exception: it returns a normal tool result whose body is
the corrective message (section 7), exactly as a no-tool-call turn is handled today. The agent is
never blocked, only told.

**Open design point (Q1).** For a CLI agent the entire run may be one `send()`. The corrective
message therefore has to be *self-sufficient*: the CLI will not consult the system prompt again.
Every refusal must restate the current phase and the exact next action.

---

## 4. Data layer

### 4.1 Structure

Framework-side, outside the workdir. A **transition graph**, not an append-only log:

```python
@dataclass(frozen=True)
class ObsNode:
    obs_hash: str          # sha1 of the canonical obs projection (see 4.2)
    obs: dict              # the full Obs payload, stored once
    first_index: int       # data index where it was first seen
    visits: int            # times observed
    is_done: bool          # terminal observations have no outgoing action edges

@dataclass(frozen=True)
class Edge:
    src: str               # obs_hash, or ROOT for a reset
    action: Action | None  # None == reset
    dst: str               # obs_hash
    indices: list[int]     # every data index that traversed it
    count: int
```

- Nodes are **deduplicated by hash**; an observation seen 40 times is one node with `visits=40`.
  This is the memory win over v2's JSONL, which stored 40 copies.
- `index_data` is a monotonic counter over *recorded transitions* (not over unique ones), so
  `load_transitions(*indexes)` keeps a stable meaning across a run.
- **Episode boundaries.** A virtual `ROOT` node per `(task, seed)`. `reset()` emits
  `ROOT --(None)--> obs0`. A terminal observation (`is_done`) gets no outgoing action edges; the
  only legal continuation is a reset, which is an edge from ROOT again. This makes "done" a
  property of a node rather than a special record type, and removes the v2 ambiguity about what
  follows a terminal step.
- **Determinism is a graph invariant, checked on insert:** an `(src, action)` pair that resolves
  to a second distinct `dst` is an anomaly (section 9), recorded rather than raised.

### 4.2 What is hashed

The hash defines "unique observation" for phase 0 and for parser injectivity, so it must be
stated exactly. Proposal: `sha1(json(frame, available_actions))`, canonical (sorted keys).

- `reward` and `is_done` are *consequences* of arriving, not part of the situation; including
  them would split one state into several.
- `info` is **excluded by default** but configurable, because problems put bookkeeping there
  (MiniGrid's shim injects `available_actions` into `info`; other keys may be step counters,
  which would make every observation unique and silently break phase 0).

**Open question (Q2):** is that the projection you want? It is the one place where a wrong choice
degrades quietly - too much in the hash and every obs is unique (phase 0 ends instantly, parser
injectivity is trivial); too little and genuinely different situations collide.

### 4.3 Access

Data lives framework-side. The workdir gets a thin client, not a copy:

```python
from framework.data import load_transitions, save_to_png, graph_summary

load_transitions(3, 7, 12)     # -> [{index, obs, action, next_obs, ...}, ...]
load_transitions()             # -> summary only, never the whole graph by accident
graph_summary()                # -> {n_obs, n_transitions, n_edges, frontier: [...], ...}
save_to_png(4)                 # -> writes <work_dir>/images/image_4.png, returns the path
```

This removes v2's workdir mirror entirely (`_TRANSITIONS_RELPATH`), which existed only because
the agent had no other way to read its own data. One copy, server-side, is also what makes the
CWM-side simulation in phase 2 possible without trusting the workdir.

---

## 5. HTTP: one service, three route families (answers Q99)

The bridge is already in the shape you want, and the answer is that **nothing new is needed at
the transport layer**:

- `security/netbridge.py` is already fully agnostic - "only ports, paths, and an argv - no agent,
  environment, or feature types". It carries host loopback into the network-namespaced sandbox
  over a unix socket. It should not be touched.
- `env/server.py` (`EnvServer`) is the single service behind it, and it *already* has two route
  families: `/env/{game_id}/*` and a **generic control channel** `/control/{game_id}/tool
  {name, input}` that any framework tool is bound onto. The control channel is what lets a CLI
  agent invoke framework tools it cannot receive natively.

v4 adds a **third family on the same server**: `/data/{game_id}/*` (`transitions`, `summary`,
`image`). That gives exactly the separation you asked for:

| layer | what it is | who calls it |
|---|---|---|
| `/env/...` | act in the world | `framework.make_env` (agent code) |
| `/data/...` | read what has been observed | `framework.data` (agent code) |
| `/control/...` | invoke a framework action | `framework/control.py` (agent shell) |

So "the agent's methods on the workdir" and "the framework helper" become a routing distinction
rather than a code-organisation convention. Other parts of regact that would profit: the
controller feature currently reaches the orchestrator in-process for `SubmitSolution`; moving it
behind `/control` uniformly (it partly is) means one auth/logging/permission point.

---

## 6. Tools

| tool | phase | change |
|---|---|---|
| `UpdateCodeWorldModel` | 1, 2 | **new** - validates and registers the CWM framework-side |
| `ProposeExploreController` | 2 | **new** - dream-then-run an exploration controller |
| `SubmitSolution` | 1, 2 | **changed** - free (no env budget), output reduced to the score |
| `ExitTask` | - | **removed** |

**`SubmitSolution` must be free and quiet.** Two requirements that interact:

- *Free*: the scoring rollout must not consume the env action budget **and must not enter the
  transition graph** - otherwise submitting becomes a cheap way to gather data, which would let
  the agent bypass phase 0 and the exploration protocol entirely. It therefore runs on a
  **separate env instance** with recording disabled.
- *Quiet*: return only the aggregate score. v2 returned the full evaluation payload, which is a
  free stdout channel the agent could print into. (This is not hypothetical: models in the last
  round used tool output as their only feedback signal, and one arm's entire run was writes with
  no reads.)

**`ExitTask` removal** is cheap: it is already gated behind `controller.exit_task_enabled`
(`features/controller.py:301`), so removal is deleting the tool, the flag, and the two prompt
sentences that mention it. Runs then end on `max_tool_calls` / walltime, which for the CLI agents
is what actually bound them anyway (section 2).

---

## 7. What the agent sees

### 7.1 Phase banner (prepended to every framework tool result)

```
[PHASE 1/3 - MODEL] env access is closed. Unique observations gathered: 14.
```

### 7.2 Refusals (illegal call for the phase)

Modelled on the existing no-tool-call reminder, and self-sufficient because a CLI agent will not
re-read the system prompt:

```
Not available in this phase. You are in PHASE 1 (MODEL): the environment is closed and your
task is to make your world model coherent with the 14 observations you have already gathered.
Edit world_model/{state,parser,render,transition}.py, then run
    python framework/control.py UpdateCodeWorldModel
You cannot reach the environment again until that check passes.
```

### 7.3 `UpdateCodeWorldModel` failure

One block per failed check, capped, concrete. This is also the message phase 2 reuses when the
env contradicts the CWM.

```
UpdateCodeWorldModel: REJECTED - 2 of 3 checks failed.

[1] PARSER INTEGRITY - failed (2 collisions of 14 observations)
    parse() maps these DISTINCT observations to the SAME state:
      obs #3 and obs #11  -> State(agent=(2,3), dir=1)
    Two situations your model cannot tell apart. Load them with
    load_transitions(3, 11) and find what your State is dropping.

[2] REPRESENTATION COHERENCE - failed
    render(parse(obs)) != obs for 4 of 14 observations. First divergence, obs #7:
      field 'frame[2][5]': expected [2,5,0], got [1,0,0]
    Your state is losing information that render() then cannot reconstruct.
    (Size check PASSED: state is 18% of obs, budget 50%.)

[3] TRANSITION COHERENCE - passed (37 of 37 transitions)
```

### 7.4 `ProposeExploreController` results

Rejected (predicts nothing new):

```
ProposeExploreController: REJECTED - not run on the environment.
Simulated in your world model for 800 steps: 0 states not already in your data.
An exploration controller must reach at least one state you have never observed.
```

Accepted:

```
ProposeExploreController: ACCEPTED - your model predicted 6 new states.
Running on the real environment...
Result: 23 transitions recorded, 6 observations new.
Your world model remained coherent. You are still in PHASE 2.
```

Note what is deliberately withheld: the *content* of the dreamt states. The agent is told **how
many**, never what - otherwise the CWM becomes an oracle it can query for free observations
rather than a model it must fix.

Contradiction (back to phase 1):

```
ProposeExploreController: the environment CONTRADICTED your world model.
Your model is no longer valid; you are back in PHASE 1 (MODEL).
[... the same anomaly block as 7.3 ...]
```

**Open question (Q3):** should a rejected proposal cost anything? As specified it is free, so an
agent can brute-force proposals against its own CWM. Free is right for a *model-checking* signal,
but it also means an arm can spend its whole budget there. I would cap consecutive rejections and
say so in the message, rather than charge for them.

---

## 8. Images

`save_to_png(index_data)` writes `<work_dir>/images/image_<index_data>.png` using the existing
`BaseProblem.render_frame(obs)`, which already exists and is already used for videos.

**Should we enforce or advise?** Advise, and make it cheap. Enforcing an image read would (a)
break every text-only model in the fleet, and (b) confound the benchmark - we would be measuring
the vision path, not world-modelling. My proposal:

- The prompt mentions `save_to_png` once, in phase 0, where looking at the world is most useful.
- Suggested, not mandated: render the **first observation of each episode** and any observation
  the agent cannot parse.
- Record whether an arm used it at all, as a per-arm behavioural signature (like first-submit
  index or blind-call ratio). Whether vision-capable models use it unprompted is itself a result.

**Open question (Q4):** do we make the tool refuse for text-only models, or let them call it and
get a path they cannot read? I lean to letting it succeed - a model that saves an image it cannot
see is making a *mistake we want to observe*, not one we should prevent.

---

## 9. Anomalies and metrics

Checked continuously as transitions are inserted, surfaced in `experiment_state.json` and
therefore in `make viz`:

| metric | meaning |
|---|---|
| `obs_determinism_violations` | `(obs,action)` seen resolving to 2+ distinct next observations |
| `history_determinism_violations` | same, keyed on the episode prefix |
| `unique_obs` / `n_transitions` / `n_edges` | graph size |
| `frontier_size` | nodes with untried actions - the exploration target |
| `phase` / `phase_transitions` | current phase and its history |
| `cwm_validations` / `cwm_rejections` | how hard phase 1 was |
| `dreamt_vs_real_divergences` | phase 2 -> 1 events |

Escalation, per section 1: obs-det violated triggers the history check; if history-det holds,
exit `partially_observable`; if both fail, exit `nondeterministic_env`.

---

## 10. Toward multi-agent / phased experiments (remark 100)

v4 is the first *protocol* feature - it constrains what the agent may do and when, rather than
adding a capability. Three things should be built generically rather than as CWM internals, so
the next phased or multi-agent experiment reuses them:

1. **`PhaseState` as a first-class run concept**, not a CWM field: a named phase, a legality
   predicate per tool/route, transition events in the transcript, and a viz affordance. A
   two-agent protocol is then "phases whose legality depends on whose turn it is".
2. **Route-level authorisation** on `EnvServer`: today `bind_control` binds tools globally. If it
   bound them *per principal* (`game_id` already scopes sessions), two agents could share one
   server with different tool sets and different env visibility - which is most of what a
   multi-agent environment needs.
3. **Recording as a policy, not a wrapper.** v2's `RecordingEnvWrapper` records because it wraps.
   With several actors, "who observed this" matters; the graph should carry a principal on each
   edge. Cheap now, very expensive to retrofit.

---

## 11. Refactor plan (ordered, each step shippable)

1. **Pre-flight determinism probe** + `fully_obs` requirement. Standalone, no v4 dependency, and
   it protects every future run. (Also answers whether any *other* task family is silently
   partially observable.)
2. **Transition graph** replacing the JSONL recorder, with `/data/` routes and `framework.data`.
   v2 keeps working on top of it.
3. **`PhaseState`** + route/tool legality + transition events + viz. No CWM logic yet.
4. **`UpdateCodeWorldModel`** with the three checks and the anomaly report format.
5. **`ProposeExploreController`** + framework-side CWM simulation.
6. **`SubmitSolution`** made free and quiet; **`ExitTask`** removed.
7. Prompts rewritten around the phases.

Steps 1-3 are useful even if v4 is abandoned, which is the main reason for that order.

---

## 12. Questions for you

- **Q0 - the blocker.** v4 needs `fully_obs=true`. Is MiniGrid-fully-observable the intended v4
  benchmark, or do you want the history-conditioned variant (harder, and a different claim)?
- **Q1.** For a CLI agent the whole run can be one `send()`. Are we content that phases are
  enforced purely at the HTTP boundary, with the agent learning the rules only from refusals?
- **Q2.** Is `sha1(frame, available_actions)` the right definition of "unique observation"?
  Should `info` be included, excluded, or per-problem?
- **Q3.** Should rejected exploration proposals be capped, charged, or free?
- **Q4.** Should `save_to_png` refuse for text-only models, or succeed and let the mistake show?
- **Q5.** Phase 2 says goals are "formalized through some state the agent wants to reach". Is the
  goal an explicit artifact the agent must *declare* (and we check the controller against it), or
  is it purely internal, with `ProposeExploreController` the only observable? The document
  assumes the latter - it is simpler and needs no goal language - but declaring goals is
  measurable in a way that reaching them is not.
- **Q6.** X (min unique observations) default 10: per task, or scaled by action-space size? On
  MiniGrid-Empty a random walk finds ~130 distinct observations in 2000 steps, so 10 is reached
  almost immediately; on a harder task it may not be.
