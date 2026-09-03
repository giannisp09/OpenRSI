# Porting OpenMLE to Another Domain

Notes from getting OpenRSI running end to end on a laptop, written for the
case where you want the *idea* rather than the machine-learning-engineering
instance of it.

The short version: OpenMLE is not really about ML engineering. It is a general
recipe for **turning a domain into a search problem over programs**, and the ML
tasks are one instantiation. If your domain can produce a verifiable score from
an executable artifact, the whole stack transfers.

---

## 1. The idea in one page

The system is a loop:

```
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │   parent selection ──► operator ──► candidate   │
  │         ▲                (LLM)       program    │
  │         │                              │        │
  │         │                              ▼        │
  │     journal  ◄──── score ◄──── sandboxed run    │
  │   (population)    (verifier)                    │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

Four moving parts, and they are genuinely separable:

| Part | Job | In OpenMLE |
|---|---|---|
| **Task** | Hand out a problem statement + data; run a candidate; return a score | `NatureBenchTask`, `MLEBenchTask` |
| **Operators** | LLM calls that *produce* a program | Draft, Improve, Debug, Crossover |
| **Search** | Decide which parent to mutate next | `evo.py` — island model + three-factor selection |
| **Verifier** | Turn an artifact into a number, without the model in the loop | NatureBench `eval_service.py` over HTTP |

The claim worth internalizing: **the verifier is the hard part, and everything
else is reusable.** Draft/Improve/Debug/Crossover are domain-agnostic — they
operate on source code and an error string. The search algorithm does not know
what an F1 score is. What makes a domain *work* is whether you can build a
verifier that is cheap, automatic, and hard to game.

### Why the operator set is what it is

These four cover the moves a human makes on a program:

- **Draft** — from scratch. Explores; ignores the population.
- **Improve** — parent + its execution feedback → better child. This is the
  workhorse and the one that actually needs a good score signal.
- **Debug** — parent + traceback → fixed child. Distinct from Improve because
  a crash carries different information than a bad score.
- **Crossover** — two parents → child. Recombines ideas.

If you port this, keep all four. Debug in particular is what makes the loop
robust to a weak model — I watched a 1.5B model crash, get caught, and enter
the debug cycle without derailing the run.

---

## 2. Does your domain qualify?

Run through this before writing code. In rough order of how often each one
kills a port:

1. **Can a machine score an attempt, with no human and no LLM judge?**
   This is the gate. "An LLM rates the output 1-10" is not a verifier — it is
   gameable and the search *will* find the exploit. You need ground truth, a
   test suite, a simulator, a solver, or a checker.

2. **Is a run cheap?** The loop needs 100+ evaluations per task. If one
   evaluation takes six hours or costs $50, you are not doing search, you are
   doing a handful of samples. Budget seconds-to-minutes.

3. **Is the score graded, not binary?** Pass/fail gives the search nothing to
   climb. You want a continuous metric, or at least many partial-credit
   subtasks. This matters more than people expect — see §4.

4. **Is the artifact a program?** The operators mutate source code. Domains
   whose output is a program (or a config, a query, a proof script, a policy)
   fit naturally. Domains whose output is prose do not.

5. **Can you sandbox it?** You are executing model-written code. See §5.

### Domains that fit well

| Domain | Artifact | Verifier |
|---|---|---|
| Compiler / kernel optimization | A transformation pass, a CUDA kernel | Correctness tests + measured wall-clock |
| Constraint & scheduling problems | A solver configuration or heuristic | Objective value, feasibility check |
| Database query optimization | A query plan or rewrite | Result equivalence + latency |
| Formal proofs | A Lean/Coq script | The proof checker. The ideal verifier |
| Hardware / circuit design | An HDL module | Simulation + timing/area from the toolchain |
| Quantitative finance | A strategy | Backtest on held-out periods |
| Protein / molecule design | A generator or scoring pipeline | Docking score, folding metric, assay data |
| Game AI | A policy or bot | Win rate against a fixed opponent pool |

### Domains that fit badly

- Anything scored by taste (copywriting, design, "is this a good essay").
- Anything where evaluation needs a human in the loop, or a wet lab, per run.
- Anything where the metric is trivially gameable and you cannot hold out data
  — the search is an *adversary* against your verifier and it is good at its job.

---

## 3. What you actually have to write

Almost all the work is one class. In OpenMLE that is
`third_party/aira-evo/examples/nature_bench/base_task.py`; read it as your
template. The interface the search needs is small:

```python
class YourTask(Task):
    def prepare(self, ...):
        """Materialize the problem: data, README, starter template."""

    def step_task(self, state, code) -> tuple[State, dict]:
        """Execute one candidate and return the score payload.

        The whole contract lives here:
          1. write `code` into a fresh, isolated workspace
          2. run it under a timeout, capture stdout/stderr
          3. hand the artifact to the verifier
          4. return {"score": float, "feedback": str, ...}
        """
```

The `feedback` string is more important than it looks. It is the *only* channel
from the verifier back into the next LLM call. §4 is about getting it wrong.

Beyond the task class you need:

- **A task package format.** OpenMLE uses a directory:
  `problem/` (README, data, starter template), `evaluation/` (evaluator +
  ground truth), `metadata.json` (metric name, direction, published baseline).
  Copy this shape; it is well-judged.
- **A verifier service.** OpenMLE runs it as a separate HTTP process so the
  candidate cannot touch ground truth. Worth imitating — see §5.
- **Operator prompts.** YAML under `configs/solver/operators/`. Mostly you are
  swapping domain vocabulary and the output-format contract.

What you do **not** need to write: the search, the island model, parent
selection, the journal, checkpointing, resume, experience memory. That is the
whole point.

---

## 4. The failure mode to design against

The most useful thing I found while running this was not in the docs, and it
generalizes to any port.

The NatureBench task `s42256-023-00611-x` has four sub-datasets. The score
reported to the search is `aggregate_improvement` — their mean. In a 16-node
run, three of the four saturated by node 6:

| Instance | node 6 → 15 |
|---|---|
| german_credit | 1.000 → 1.000 (saturated) |
| ist_aspirin | 1.000 → 1.000 (saturated) |
| ist_heparin | 0.9987 → 0.9987 (frozen) |
| twin_mortality | 0.6195 → 0.6380 (jitter) |

Two thirds of the budget ran with **one** instance carrying all remaining
headroom. Aggregate scores across those ten nodes differed by less than 0.01,
so parent selection had nearly no gradient, and Improve had nothing to aim at.
Meanwhile the per-instance breakdown *was* being computed, returned by the
verifier, and written to the journal — it just never reached the prompt. The
feedback string was two scalars.

**The general lesson: your scalar score drives selection, but your feedback
string drives generation, and they need different information.** A mean over
sub-problems is fine for ranking candidates and actively harmful as guidance,
because it hides which sub-problem is worth working on.

### The trap on the other side of that fix

Fixing this naively makes things *worse*, and the failure is worth knowing
before you hit it. Three arms, same 16-node budget, one run each:

| arm | best | crashes | twin_mortality mean | nodes preserving the 3 solved instances |
|---|---:|---:|---:|---:|
| A aggregate only | +0.0087 | 2 | 0.6249 | 10/13 |
| B per-instance, no constraint | +0.0046 | 0 | 0.6300 | **1/15** |
| C per-instance + constraint | **+0.0147** | 0 | **0.6398** | **13/15** |

Arm B added the per-instance breakdown and named the weakest instance. It
worked on the target — twin_mortality rose — and **preservation of the solved
instances collapsed from 10/13 nodes to 1/15**. Because the aggregate is a
mean, regressions on solved instances cost more than the available gain on the
weak one, so the net result was worse than doing nothing.

The cause: pointing at the weak component reads as permission to trade the
others away. The phrasing that did the damage was *"gains on twin_mortality
move the aggregate most"* — true, and it implies the rest do not matter.

**The scalar aggregate was silently enforcing a "don't break what works"
constraint. Decomposing the feedback removes it, so you have to put it back
explicitly.** Arm C restates it — the other components are at or above target,
must stay there, and a regression on them outweighs any gain on the weak one —
and recovers both the targeting benefit and preservation above baseline.

This generalizes to any per-component feedback scheme. It is not specific to
this task, this metric, or this model.

### Designing the feedback string

- Report **per-component scores**, not just the aggregate. Name the weakest one.
- **State the preservation constraint in the same breath.** Non-negotiable —
  see above.
- Include **what failed and why** — the traceback, the failing test name, the
  constraint that was violated.
- Suppress detail that carries no signal. A crashed candidate scores the same
  floor everywhere; the breakdown is noise and the traceback is everything.
- Keep the aggregate too. Selection still needs one comparable number.

Corollary for metric design: **prefer many partial-credit components over one
opaque scalar.** Not only does it grade better, it gives the model somewhere to
aim — provided you also tell it what not to sacrifice.

Caveat on the numbers above: one run per arm, one task, one model. The
preservation column is a large effect with a mechanism behind it; the `best`
column on its own sits inside run-to-run noise. Treat it as a hazard to design
around, not a measured speedup.

---

## 5. Sandboxing, and the thing people get wrong

You are running code an LLM wrote, and the search is explicitly optimizing
against your scorer. Two distinct concerns, and conflating them is the mistake:

**Isolation.** OpenMLE's local mode uses a Conda env and says so plainly:

> Conda isolates Python dependencies, not files, networking, processes, or host
> permissions. Model-generated code therefore runs with the current user's local
> access.

That is correct and worth repeating in your own docs. A virtualenv is a
dependency boundary, not a security boundary. For anything beyond a trusted dev
machine, use containers or the documented remote-execution path.

**Verifier integrity.** Keep ground truth out of the candidate's reach, or the
search will find it — not from malice, but because reading the answer key is a
very effective way to maximize a score. OpenMLE's structure here is worth
copying wholesale:

- The verifier is a **separate process**, reached over HTTP.
- Ground truth lives with the verifier, never in the candidate's workspace.
- The candidate writes predictions to an output directory; the verifier reads
  that directory and the answers, and returns only a number.
- Control endpoints are **token-gated**, and the token is deliberately excluded
  from the environment variables passed to candidate code.

That last point is easy to lose. If your control token leaks into the
candidate's env, model-written code can register tasks and rewrite scores.
OpenMLE gets this right via an explicit `candidate_env_allowlist` — an allowlist,
not a denylist. Do the same.

Also hold out a test split the search never sees. Validation score drives the
search; the number you report should come from data the search could not
overfit.

---

## 6. A porting order that works

1. **Verifier first, alone.** Before any LLM: a script that takes an artifact
   directory and returns a number. Run it on a known-good and a known-bad
   solution. If you cannot do this, stop — the domain does not qualify.
2. **One task package.** Pick a problem you know is solvable, that runs in
   under a minute, and where you know roughly what a good score is.
3. **Wire the task class, run one candidate.** Use a small/cheap model. You are
   testing plumbing, not capability. Expect the candidate to fail; you want to
   see the *failure* propagate correctly into a score and a feedback string.
   (I did this shakedown against a 1.5B local model for free, and it caught
   three protocol bugs before I spent anything.)
4. **Then a capable model, bounded budget.** ~15 nodes. Confirm the score
   actually climbs. This is your baseline and everything later is measured
   against it.
5. **Read the trajectory before tuning anything.** The per-instance discovery
   above came from reading `submissions.jsonl` row by row, not from theory.
6. **Only then scale up** nodes, tasks, and search-policy changes.

Two practical notes from doing this:

- **Cheap smoke first, always.** A free local model finds protocol bugs just as
  well as an expensive one, and you will have protocol bugs.
- **Bound the budget while iterating.** The defaults here are 160 nodes and a
  four-hour wall clock. A 16-node cap turns an afternoon into ten minutes and
  is more than enough to see whether the loop climbs.

---

## 7. Where the open research problems are

If you want to contribute method rather than a new domain, these are visible in
the config and mostly untouched. Everything in
`configs/search/airaevo_naturebench.yaml` is a hand-tuned constant:

- **Parent selection** is a fixed weighting: `score: 1.0, delta: 0.4,
  novelty: 0.25`, constant for the whole search. Early search wants novelty,
  late search wants score. Nothing anneals.
- **Operator scheduling** is fixed: `crossover_prob: 0.5`,
  `fresh_draft_prob: 0.2`. These never respond to stagnation or to what kind of
  error the last candidate produced.
- **Debug budget** is fixed: `max_debug_depth: 2`, regardless of whether the
  failure is a typo or a modelling error.
- **Experience retrieval** selects related nodes by *tree ancestry*
  (`ancestor_k`, `sibling_k`), not by similarity of failure mode. Two nodes that
  failed the same way are the useful pair, and they may sit in different
  subtrees.
- **Credit assignment**: when a 15-node search improves, which operator earned
  it? The journal records enough to attribute this and nothing does.

All of these are testable with 16-node runs on one task, which is what makes
them tractable without a cluster.

---

## 8. Reference points in this repo

| What you want | Where |
|---|---|
| The task interface to copy | `OpenMLE-Evo/third_party/aira-evo/examples/nature_bench/base_task.py` |
| Verifier service structure | `NatureBench/eval_service.py` |
| Search algorithm | `.../src/dojo/solvers/evo/evo.py` |
| Operator implementations | `.../src/dojo/core/solvers/operators/` |
| Operator prompts | `.../src/dojo/configs/solver/operators/` |
| Search hyperparameters | `OpenMLE-Evo/tts_search/configs/search/airaevo_naturebench.yaml` |
| Task package layout | `OpenMLE-Evo/.naturebench/data/tasks/<task-id>/` |
| A real trajectory to read | `output/<run>/workspaces/_eval_service/<batch>/<task>/submissions.jsonl` |

Read `base_task.py` and `eval_service.py` together — the contract between them
is the actual interface you are reimplementing, and the three bugs I hit porting
this were all disagreements across that boundary.
