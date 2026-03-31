# NeuralBo

This repository contains the implementation for the AISTATS 2026 paper
"Efficient Bilevel Optimization with KFAC-Based Hypergradients".

NeuralBo is a PyTorch library for implementing bilevel optimization algorithms
for modern neural architectures. The current codebase focuses on reusable
implicit-differentiation solvers and example workloads ranging from image data
cleaning to BERT-scale experiments.

Current reusable core in `src/`:
- `AIDCGSolver` (AID with conjugate-gradient inverse approximation)
- `AIDKFACSolver` (AID with KFAC inverse approximation)
- problem interfaces (`NeuralBoProblem`, `AIDProblem`)

## Install
From repo root:

```bash
uv pip install -e . torchvision \
transformers==4.37.2 \
"curvlinops-for-pytorch @ git+https://github.com/liaodisen/curvlinops.git@new-wce-kfac"
```

## Project Structure
```text
neuralbo/
├─ src/
│  ├─ problem.py              # NeuralBoProblem and AIDProblem interfaces
│  └─ solver/
│     ├─ aid.py               # AIDConfig, KFACConfig, AIDCGSolver, AIDKFACSolver
│     └─ utils.py             # CG utilities
├─ examples/
│  ├─ image_data_cleaning/
│  │  ├─ data_cleaning.py     # image data-cleaning example
│  │  └─ model.py             # image model  builders
│  ├─ bert_data_cleaning/
│  │  ├─ bert_data_cleaning.py
│  │  └─ bert_model.py
│  └─ trec/                   # local TREC split + hard labels
└─ pyproject.toml
```

## Core API
`src/problem.py` defines the interfaces:
- `NeuralBoProblem.init() -> (model, lam)`
- optional data access via `dataloader(role, batch_size)` or `batch(role, batch_size)`
- `AIDProblem` adds:
  - `inner_loss(model, lam, batch)`
  - `outer_loss(model, lam, batch)`

`src/solver/aid.py` provides:
- `AIDConfig`
- `KFACConfig`
- `AIDCGSolver.solve(problem, callback=None) -> (model, lam, history)`
- `AIDKFACSolver.solve(problem, callback=None) -> (model, lam, history)`

For AID solvers, expected roles are: `inner`, `outer`, `hxx`, `xw`.

## Usage
### Image Example
```bash
python examples/image_data_cleaning/data_cleaning.py
```

### BERT Example
```bash
python examples/bert_data_cleaning/bert_data_cleaning.py
```

The BERT example defaults to `examples/trec/` for data and `hard_labels.npy` for noisy train labels.

## Notes
- Weight projection is handled in the example callbacks via manual clipping (`w.clamp_(0, 1)`), not in core solver logic.
- Additional legacy/experimental scripts may exist at repo root, but the maintained reusable interface is under `src/` and `examples/` above.

## TODO
- Add more bilevel optimization baselines, including `BOME` and `AmIGO`.
- Extend support to more modern architectures and large-scale bilevel workloads.

## Acknowledgements
- `AID-KFAC` in this repository relies on the
  [`curvlinops`](https://github.com/f-dangel/curvlinops) library for curvature
  operators and related infrastructure.
