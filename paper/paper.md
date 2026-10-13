# Demand-Driven Demonstration and Query Evaluation in Grid Program Synthesis

Touko Ursin. Code (MIT-0): https://github.com/ToukoUrsin/arc-prize-2026-demand-driven-synthesis.
Kaggle notebook https://www.kaggle.com/code/toukoursin/arc-agi-2-demand-driven-synthesis (version 1); ARC-AGI-2 submission 57124582, public leaderboard score 2.50.

## Abstract

We implement exact demand-driven evaluation for bounded unary grid synthesis.
A provenance DAG postpones query and later-demonstration execution until needed. Across 901 public 1D-ARC
files, one-demonstration/three-query search preserves ranked candidates with
84.49% fewer primitive calls. A separate three-demonstration/one-query study
saves 62.69% beyond cached deferred queries, preserving 1,862 programs and
652 correct tasks. Equivalent packed primitives reduce CPU by 58.16%
on 1D and 48.16%/49.94% on ARC1/ARC2. An anytime ablation adds six ARC1
solves at identical primitive work. Controlled execution-policy comparisons preserve predictions; grammar/depth changes explain accuracy gains.

## Problem and prior work

ARC supplies labeled demonstration grids and unlabeled query inputs. Demonstration-equivalent programs can predict different query outputs.
Merging those prefixes loses possibilities; including queries in every signature
preserves them but evaluates paths that never fit demonstrations.

Observational equivalence and lazy evaluation are established.
[Lau et al.](https://research.ibm.com/publications/programming-by-demonstration-using-version-space-algebra)
study version-space algebra;
[PROSE](https://www.microsoft.com/en-us/research/project/prose-framework/usage/)
uses lazy version spaces. [COGS](https://repositum.tuwien.at/handle/20.500.12708/230505)
refines observational-equivalence graphs using further observations.
[SYNTRA](https://arxiv.org/abs/2509.17393) uses completed hypotheses and pseudo-labels.
Executable exactness checks accompany controlled measurements; broader novelty remains unestablished.

## Algorithm and exactness

Fix ordered deterministic unary operations, dimension bounds and a depth limit.
Construct all grammar parameters from all demonstrations before staging. At each
exact depth, group equal intermediate demonstration vectors into DAG nodes.
Retain every incoming operation edge, including paths returning to a state from
an earlier depth. Mark terminals satisfying demonstrations or admitting a
consistent terminal color map within the step budget.

Deferred execution materializes query vectors only along marked terminals and
their ancestors. Every incoming operation is applied to every represented
predecessor query vector. Query-undefined paths are rejected. Equal query vectors
at an exact-depth node retain their lexicographically least primitive-index path.
Final tuples rank by total steps, primitive indices and terminal-map flag.
Two distinct grids are selected independently per query; missing attempts use
its input as fallback.

Staged execution first builds a first-demonstration DAG, prunes to all ancestors
of feasible terminals, and refines these paths with each remaining demonstration.
A fully fitting terminal must fit every subset. Exact equality and color-map
consistency impose monotone restrictions. Subset maps test feasibility only;
the applied terminal map is fitted using every demonstration. Conflicting pixel
assignments or incompatible shapes reject it, unseen query colors stay unchanged,
and a nonidentity map costs one step. Original demonstration order is retained.

Induction retains every path needed by a fully fitting terminal, not every
currently defined prefix: feasibility and ancestor pruning remove unproductive
paths. Undefined is absorbing under pure deterministic unary partial operations.
Equality of full demonstration-and-query vectors is a congruence under unary
continuations; least paths at equal depth preserve lexicographic extension order.
Terminal fitting and final minimization preserve output tuples and minimum ranked
programs, not syntax multiplicity. With partial operations, demonstration-only merging can lose even the
existence of a query-defined solution; an executable counterexample is included.

The argument assumes pure fixed unary semantics, absorbing undefined values,
unit costs, exact-depth nodes and no beam. Binary operators, hidden grammar state,
weighted costs and nonmonotone terminal constructors require further machinery.
Savings are empirical: provenance can itself cost more than eager execution.

## Protocol

Every 1D task receives 36 public concept-family operations plus ten generic
relations: shifts, denoising, span filling, movement, periodic completion and
copying; widths are bounded at 100. Categories/IDs never enter inference; domain priors prevent matched LLM comparisons.

The main experiment converts each source file's first example into the sole
demonstration and its other examples into three queries. The original-protocol
experiment instead retains all three demonstrations and the single original
query. All 901 files are included; one category contains 51 files. The 2D control uses rigid transforms, crops, component selection, filtering,
compression, halves and repetition, bounds 1–30 and depth two. All engines
share grammar, ranking and primitive semantics. Primitive calls count actual
per-grid function executions, excluding color-map applications and bookkeeping.
Solver CPU includes those costs; common grammar construction is excluded from
paired experiments. Timing includes instrumentation and shared-host contention; no dedicated-machine or significance claim.

## Results

| Protocol, depth three | Execution | Primitive calls | Solver CPU seconds | Correct tasks |
|---|---|---:|---:|---:|
| One demo / three queries, uncached | Eager | 57,717,488 | 458.616 | 625/901 |
| Same | Deferred | 8,951,724 | 76.119 | 625/901 |
| Three demos / one query, cached packed evaluator | Eager | 26,828,810 | 218.705 | 652/901 |
| Same | Deferred | 20,663,875 | 158.834 | 652/901 |
| Same | Staged | 7,708,870 | 56.129 | 652/901 |

The main comparison preserves all 14,962 canonical tuples and 1,962/2,703 correct
query outputs: 84.49% fewer calls and 83.40% less CPU. The second uninterrupted
paired experiment uses identical 200,000-entry task-local caches and the same
packed evaluator across arms. All 1,862 programs match and independently replay
using original NumPy primitives. Staging saves 62.69% calls and 64.66% CPU versus
deferred, or 71.27% calls and 74.34% CPU versus eager. Every task uses fewer calls
than deferred; 900 use less CPU. Execution preserves accuracy.

An unchanged-search NumPy/packed pair reduces CPU from 136.203 to
56.990 seconds (58.16%), preserving programs, graph metrics and calls.

A 2D ablation runs four paired arms on every public task.
With unchanged staged search, NumPy/packed solver CPU is 24.446/12.673 seconds
on ARC1 and 12.348/6.181 on ARC2: 48.16%/49.94% reductions. Every program, attempt, graph metric and call count matches.
With the packed evaluator shared across policies, deferred/staged calls are
2,003,341/777,609 on ARC1 and 653,569/301,193 on ARC2. Staging saves
61.18%/53.92% calls and 66.03%/59.99% CPU respectively. Separate paired experiments use cold task-local component caches and unchanged scores.

Staged search solves 5/400 ARC1, 0/120 ARC2. A separate portfolio preserves
native attempt1, ranking attempt2 among native and learned grids using
37 source, complexity, consistency and omitted-demo support features.
Logistic regularization is selected on reused 200/200 training splits;
confirmation-selected C10 raises native 119→124/400 ARC1
(seven wins, two losses; 130→136/419 queries) and 1→2/120 ARC2.
The added ARC2 task also appears in ARC1. Public 125-task ablations remain
unselected. Removing learned proposals restores native behavior.
An exact shortcut skips validation when every query has one residual candidate:
50 folds omitted, all 520 submissions preserved, ARC1 learned calls
7,381,623→6,826,703 (7.52% fewer). Paired learned CPU is 416.748→402.028s
ARC1,160.750→161.436s ARC2; native cached. Fresh ARC2 entry CPU is 654.991s,
including 493.066s native; all 520 predictions match. ARC1 total unavailable
after timing-write failure; no speedup/significance claim.

The entry compiles demonstration-fitted relational trees into unary operations,
ranked by demonstration-only leave-one-out scores, alongside screened object
constructors. Grammar construction receives no queries. This grammar solves 27/400 ARC1
tasks (30/419 outputs), with 22 wins and no losses; ARC2 improves to 1/120.
Grammar coverage explains accuracy gains. Exact feature evaluation requests
columns only at reached tree nodes, sharing components and vectorized rays.
Paired total CPU, including construction, falls from 356.786 to 207.243 seconds
on ARC1 and 125.962 to 75.482 on ARC2: 41.91%/40.08% less, with identical calls
and predictions; 328/400 and 93/120 tasks are faster. Learned ARC2
staging uses 684,494 versus 317,077 deferred/staged calls (53.68% fewer) and
107.059 versus 75.482 total CPU seconds (29.50% less), preserving 202 programs
and 1/120 tasks. 

An optional anytime depth-three notebook fills duplicate second-answer slots
while preserving every baseline guess. It retains only fully demonstration-
and-query-validated candidates at interruption; ranking among unseen programs
is unclaimed. Five-budget sweeps retain the same constructed grammar
and clone baseline caches. At 5,000 extra calls per task, retaining versus
discarding incomplete passes solves 33/400 versus 27/400 ARC1 tasks: six wins,
zero losses at identical 902,320 extra executions. Fresh whole-solver paired CPU, including construction/baseline, is
143.786/146.635 seconds on ARC1 (+1.98%) and 59.834/59.895 on ARC2.
At 50,000 calls both solve 34/400; ARC2 stays 1/120 at every allowance.
The optional 5,000 default follows this public sweep.
Depth two remains primary because deeper ARC2 search has no measured gain.

## Reproducibility and limits

The native/residual notebook embeds 93 assets, including frozen coefficients,
and requires C++17/make, Python, NumPy, SciPy and sklearn. The fourteen-source
compiler-free notebook remains available; neither needs downloads or GPU. The compiler-free default uses learned staged search;
whole-run settings choose deferred execution or the original grammar.
Compiler-free notebook replays cover 400 ARC1 staged tasks, both 120-task ARC2 policies,
and 520 original-grammar tasks, matching programs and work counts. The separate
1D and original-grammar controls retain their source-bound evidence.
Allowlist mappings omit query labels and metadata; Python API overrides/audit
events deny file/network access during inference, without OS isolation.
Independent replay uses full features and a separate tree interpreter:
67,743,312 values match across 912 grids and all 900 dimension pairs. The latest Pro review executed
integrity/guard checks only; new scientific claims remain unverified.
Records, inputs and commands accompany notebooks.

These are public-development results. Prior learner public outcomes were known;
ARC1 evaluation overlaps ARC2 training. Aggregate weakness motivated engineering; restricted concepts limit accuracy. Authored code uses
MIT-0, ARC data Apache-2.0 and 1D-ARC MIT, with notices retained. Codex assisted
implementation.
