# NeuralBo

Reusable PyTorch library for bilevel optimization in neural networks
Implements stocBiO, AID-CG, AID-KFAC and common implicit-gradient baselines.

Designed to turn research code into a clean, reusable toolbox for:
- Hyperparameter optimization
- Meta-learning
- Neural architecture search
- Continual learning


## User Interface (API)
NeuralBo is designed around two concepts: a `NeuralBoProblem` base class and solver
implementations that operate on problem instances. Solvers only depend on the methods
they actually need.

### Core Base Class
```python
from src.problem import NeuralBoProblem, AIDProblem

class NeuralBoProblem:
    """
    Base class for bilevel problems.
    """
    def init(self) -> tuple[nn.Module, torch.Tensor]:
        """Return (model, lam)."""
        ...

    def batch(self, role: str, batch_size: int):
        """Return a batch for a given role."""
        ...

    def batch_roles(self) -> Iterable[str]:
        """Optional: advertise the roles you support."""
        return ()

    def project_lam(self, lam: torch.Tensor) -> torch.Tensor:
        """Optional projection/clipping for lam."""
        return lam


class AIDProblem(NeuralBoProblem):
    """
    AID-specific extension: define inner/outer losses.
    """
    def inner_loss(self, model, lam, batch) -> torch.Tensor: ...
    def outer_loss(self, model, lam, batch) -> torch.Tensor: ...
```

### Solver Contract
```python
class Solver:
    def solve(self, problem: NeuralBoProblem):
        """Return (model, lam, history)."""
        ...
```

### Batch Roles (AID)
For AID-CG / AID-KFAC, `batch(role, batch_size)` uses:
- `inner`: inner-loop steps on `g(theta, lam)`
- `outer`: upper loss `f(theta, lam)`
- `hxx`: Hessian-vector products for `H_xx`
- `xw`: cross term `H_xw`

## Proposed Usage (AID-CG / AID-KFAC)
```python
from solver import AID_CG, AID_KFAC

problem = DataCleaningProblem(
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    model=LogReg(input_dim),
)

theta, lam = problem.init()

# AID-CG / AID-KFAC expect raw tensors and an args-like config
history = AID_CG(
    args,
    theta,
    lam,
    trainset=problem.train_data,
    valset=problem.val_data,
    testset=problem.test_data,
    tevalset=problem.val_data,
    clean_indices=problem.clean_indices,
)

history = AID_KFAC(
    args,
    theta,
    lam,
    trainset=problem.train_data,
    valset=problem.val_data,
    testset=problem.test_data,
    tevalset=problem.val_data,
    clean_indices=problem.clean_indices,
)
```

See `examples/data_cleaning.py` for a self-contained sketch of the problem interface
and an AID-CG style loop.

## Extending Algorithms
To add a new algorithm:
1. Define a new `Problem` mixin that extends `NeuralBoProblem` with the methods your
   solver requires (e.g., `inner_loss`, `outer_loss`, or custom operators).
2. Implement a `Solver` class with `step()` and `solve()` that consumes that mixin.
3. Keep the interface minimal: only require methods the solver actually needs.
4. Add an example problem in `examples/` to document the expected behavior.


## Repo Structure
neuralbo/
├─ src/neurobo/
│  ├─ problem.py          # BilevelProblem interface
│  ├─ solve.py            # Main entrypoint
│  ├─ solvers/
│  │   ├─ stocbio.py
│  │   ├─ aid.py
│  │   └─ baselines.py
│  ├─ ops/
│  │   ├─ hvp.py          # Hessian-vector products
│  │   └─ linear_solve.py # CG / Neumann solvers
│  └─ utils/
│      ├─ seed.py
│      └─ logging.py
│
├─ examples/
│  ├─ toy_quadratic.py
│  └─ mnist_hpo.py
│
├─ experiments/           # Paper reproduction scripts
├─ tests/                 # Smoke + correctness tests
└─ pyproject.toml
