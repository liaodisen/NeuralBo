# Examples

This folder contains runnable examples that exercise the reusable solver APIs.

## Data Cleaning (AID-CG)

`data_cleaning.py` is a minimal, self-contained sketch of bilevel data cleaning:
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
python examples/data_cleaning.py
```

### What to Configure
Open `examples/data_cleaning.py` and edit:

1. `AIDConfig(...)`
   - `batch_size`: mini-batch size for inner/outer/HVP batches
   - `inner_steps`: number of inner-loop steps per outer update
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
   - `input_dim`: feature dimension (matches `d`)
   - `device`: `"cpu"` or `"cuda"`

### Notes
- `AIDCGSolver` follows the AID-CG structure in `solver/aid.py`.
- To try AID-KFAC, pass a `KFACConfig` and a `kfac_loss_builder` to `AIDKFACSolver`.
