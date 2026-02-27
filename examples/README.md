# Examples

This folder contains runnable examples that exercise the reusable solver APIs.

Subfolders:
- `examples/bert_data_cleaning/`
- `examples/image_data_cleaning/`

## Image Data Cleaning

`image_data_cleaning/data_cleaning.py` is a minimal, self-contained sketch of bilevel data cleaning:
- Inner loss: weighted training loss
- Outer loss: validation loss
- `lam`: per-sample weights in `[0, 1]`

### Requirements
Install PyTorch first. Example (CPU-only):

```bash
python -m pip install torch
```

### Run
```bash
python examples/image_data_cleaning/data_cleaning.py
```

### What to Configure
Open `examples/image_data_cleaning/data_cleaning.py` and edit:

1. `AIDConfig(...)`
   - `batch_size`: mini-batch size for inner/outer/HVP batches
   - `inner_steps`: number of inner updates per outer update
   - `inner_lr`: learning rate for inner updates (theta)
   - `outer_lr`: learning rate for outer updates (lam)
   - `cg_maxiter`, `cg_tol`, `cg_damping`: CG solver settings for AID-CG
   - `epochs`: number of outer iterations

2. `make_toy_data(...)`
   - `n_train`, `n_val`: sizes of train/val sets
   - `d`: input dimension
   - `noise_rate`: fraction of label noise in training set
   - `seed`: deterministic data generation

3. `DataCleaningProblem(...)`
   - `model_factory`: model constructor
   - `device`: `"cpu"` or `"cuda"`
   - `dataloader(role, batch_size)`: provide role-specific iterables/loaders

### Notes
- `AIDCGSolver` follows the AID-CG structure in `solver/aid.py`.
- `solver.solve(problem)` runs the full optimization loop; examples no longer call `solver.step(...)` in a manual epoch loop.
- To try AID-KFAC, pass a `KFACConfig` and a `kfac_loss_builder` to `AIDKFACSolver`.


## BERT Data Cleaning

`bert_data_cleaning/bert_data_cleaning.py` shows a minimal BERT cleaning loop on the local TREC split in `examples/trec/`.

### Requirements
```bash
python -m pip install torch transformers
```

### Run
```bash
python examples/bert_data_cleaning/bert_data_cleaning.py
```

### Notes
- Implemented as `AIDProblem` + shared AID solvers (reuses `src/problem.py` and `src/solver/aid.py`).
- Supports `--alg AID-CG` and `--alg AID-KFAC` (`AID-KFAC` default).
- Train labels default to `examples/trec/hard_labels.npy`.
- BERT freeze policy is configurable with `--fine_tune_level`, `--fine_tune_order`, and `--specific_layer`.
- Default is `--fine_tune_level 0` (only classifier head is trainable).
- Validation/test labels come from `valid.json` and `test.json` (clean labels).
- Hypergradient computation is wrapped in `sdpa_kernel(SDPBackend.MATH)` when available, with a safe no-op fallback on incompatible Torch versions.
