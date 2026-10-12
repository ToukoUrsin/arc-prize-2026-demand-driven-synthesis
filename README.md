# ARC Prize 2026 — Demand-Driven Demonstration and Query Evaluation in Grid Program Synthesis

Code entry for the ARC Prize 2026 ARC-AGI-2 competition and the paper-track writeup
"Demand-Driven Demonstration and Query Evaluation in Grid Program Synthesis" (Touko Ursin).

- `entry.ipynb` — the Kaggle notebook (CPU, internet off). It embeds all 93 source assets,
  writes them to `arc_portfolio_bundle/`, builds the native C++ solver, runs it, ranks the
  second attempt with a frozen logistic model, and writes `submission.json`
  (two attempts per test input).
- `src/` — the same 93 assets extracted for reading. The notebook is the source of truth.

Public evaluation (local, not official): 124/400 ARC-AGI-1 tasks (136/419 test outputs) and
2/120 ARC-AGI-2 tasks (2/167), against 119/400 and 1/120 for the native solver alone.

## Method in one paragraph

A query-aware breadth-first synthesizer over a bounded unary grid DSL evaluates demonstrations
and queries on demand through a provenance DAG, so query outputs are only computed for paths
that fit the demonstrations. A native C++ solver supplies attempt 1; attempt 2 is chosen by a
logistic ranker (C=10, 37 features: source/complexity, shape and color consistency,
omitted-demonstration agreement) over the native second grid and all learned program grids.

## Requirements

Python 3 with NumPy, SciPy and scikit-learn, a C++17 compiler and `make` (all present in the
default Kaggle image). No downloads, no GPU. Inference reads only demonstrations and unlabeled
query inputs from `arc-agi_test_challenges.json`.

## License

Authored code: MIT-0 (`LICENSE`). `src/native/` is derived from the ARC solver by top-quarks
(MIT, `src/native/LICENSE`). ARC data is Apache-2.0 and not included.
