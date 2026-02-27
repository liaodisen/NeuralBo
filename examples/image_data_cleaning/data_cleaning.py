from typing import Tuple, Callable
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.problem import AIDProblem
from src.solver.aid import AIDConfig, AIDCGSolver, AIDKFACSolver, KFACConfig
from model import (
    get_model,
    get_mlp_model,
    get_conv_model,
    get_conv_model2,
    get_resnet_model,
)


TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DataCleaningProblem(AIDProblem):
    """
    Simple data cleaning bilevel problem:
      - inner loss: weighted training loss
      - outer loss: validation loss
      - lam: per-sample weights in [0, 1]
    """

    def __init__(
        self,
        train,
        val,
        model_factory: Callable[[], nn.Module],
        num_classes: int,
        device: str = "cpu",
        batch_size: int = 128,
        flatten_inputs: bool = False,
    ):
        self.device = torch.device(device)
        self.train_x, self.train_y = train
        self.val_x, self.val_y = val
        self.train_x = self.train_x.to(self.device)
        self.train_y = self.train_y.to(self.device)
        self.val_x = self.val_x.to(self.device)
        self.val_y = self.val_y.to(self.device)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_factory = model_factory
        self.flatten_inputs = flatten_inputs

        self._train_ds = self._get_dataset(self.train_x, self.train_y)
        self._val_ds = self._get_dataset(self.val_x, self.val_y)

    def _get_dataset(self, x: torch.Tensor, y: torch.Tensor):
        return torch.utils.data.TensorDataset(
            x,
            y,
            torch.arange(x.shape[0], device=self.device),
        )

    def _get_loader(self, dataset, batch_size: int):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def init(self): 
        model = self.model_factory().to(self.device)
        w = torch.zeros(self.train_x.shape[0], requires_grad=True, device=self.device)
        w.data.add_(0.5)
        return model, w

    def inner_loss(self, model: nn.Module, w: torch.Tensor, batch: TensorBatch):
        x, y, idx = batch
        if self.flatten_inputs:
            x = x.view(x.size(0), -1)
        logits = model(x)
        per_example = F.cross_entropy(logits, y, reduction="none")
        weights = w[idx].clamp(0.0, 1.0)
        return (weights * per_example).mean()

    def outer_loss(self, model: nn.Module, w: torch.Tensor, batch: TensorBatch):
        x, y, _ = batch
        if self.flatten_inputs:
            x = x.view(x.size(0), -1)
        logits = model(x)
        return F.cross_entropy(logits, y, reduction="mean")

    def dataloader(self, role: str, batch_size: int):
        self.validate_role(role)
        if role == "outer":
            return self._get_loader(self._val_ds, batch_size)
        return self._get_loader(self._train_ds, batch_size)

    def project_lam(self, w: torch.Tensor):
        return w.clamp(0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion", "cifar"])
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--noise_rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--poison_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "random"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--outer_lr", type=float, default=1.0)
    parser.add_argument("--alg", type=str, default="AID-CG", choices=["AID-CG", "AID-KFAC"])
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--kfac_mc_samples", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "mlp", "conv", "conv2", "resnet18", "resnet50"],
    )
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--conv2_hidden", type=int, default=100)
    parser.add_argument("--conv2_layers", type=int, default=2)
    return parser.parse_args()


def get_data(
    dataset: str,
    data_root: str,
    n_train=50000,
    n_val=5000,
    noise_rate=0.3,
    seed=0,
    poison_mode="deterministic",
):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required to load MNIST. Install it with: "
            "python -m pip install torchvision"
        ) from exc

    if dataset == "cifar":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_train)
        x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
        y = torch.LongTensor(train_ds.targets)
        x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
        y_test = torch.LongTensor(test_ds.targets)
    else:
        ds_cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        train_ds = ds_cls(root=data_root, train=True, download=True)
        test_ds = ds_cls(root=data_root, train=False, download=True)
        x = train_ds.data.float() / 255.0
        y = train_ds.targets.clone()
        x_test = test_ds.data.float() / 255.0
        y_test = test_ds.targets.clone()

    # Match cleaning.py behavior: global RNG with manual_seed called in main
    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]

    # Match cleaning.py noise injection
    old_y_train = y_train.clone()
    num_noisy = int(noise_rate * n_train)
    num_classes = 10
    if poison_mode == "deterministic":
        noisy_indices = torch.arange(num_noisy)
    elif poison_mode == "random":
        noisy_indices = torch.randperm(n_train)[:num_noisy]
    else:
        raise ValueError(f"Unknown poison mode: {poison_mode}")
    noisy_y = torch.randint(num_classes, size=(num_noisy,))
    y_train[noisy_indices] = noisy_y.data

    if dataset != "cifar":
        mean = x_train.unsqueeze(1).mean([0, 2, 3])
        std = x_train.unsqueeze(1).std([0, 2, 3])
        x_train = torch.flatten((x_train - mean) / (std + 1e-4), start_dim=1)
        x_val = torch.flatten((x_val - mean) / (std + 1e-4), start_dim=1)
        x_test = torch.flatten((x_test - mean) / (std + 1e-4), start_dim=1)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes


def make_model_factory(args, input_dim: int, num_classes: int, device: str):
    if args.model == "linear":
        return lambda: get_model(input_dim, num_classes, device)
    if args.model == "mlp":
        return lambda: get_mlp_model(input_dim, num_classes, args.mlp_hidden, device)
    if args.model == "conv":
        return lambda: get_conv_model(device)
    if args.model == "conv2":
        return lambda: get_conv_model2(device, num_hidden_layers=args.conv2_layers, hidden_size=args.conv2_hidden)
    if args.model in {"resnet18", "resnet50"}:
        return lambda: get_resnet_model(args.model, num_classes=num_classes, pretrained=False, device=device)
    raise ValueError(f"Unknown model: {args.model}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    repo_root = Path(__file__).resolve().parents[2]
    local_data_root = str(repo_root / "data")
    data_path = {
        "mnist": local_data_root,
        "fashion": local_data_root,
        "cifar": local_data_root,
    }
    train, val, test, num_classes = get_data(
        dataset=args.dataset,
        data_root=data_path[args.dataset],
        n_train=args.train_size,
        n_val=args.val_size,
        noise_rate=args.noise_rate,
        seed=args.seed,
        poison_mode=args.poison_mode,
    )
    if args.model in {"conv", "conv2", "resnet18", "resnet50"} and args.dataset != "cifar":
        raise ValueError("conv/resnet models require CIFAR input. Use --dataset cifar.")

    input_dim = int(torch.numel(train[0][0]))
    flatten_inputs = args.model in {"linear", "mlp"}
    model_factory = make_model_factory(args, input_dim, num_classes, args.device)

    problem: AIDProblem = DataCleaningProblem(
        train,
        val,
        model_factory=model_factory,
        num_classes=num_classes,
        device=args.device,
        batch_size=args.batch_size,
        flatten_inputs=flatten_inputs,
    )
    test_x, test_y = test
    test_x = test_x.to(args.device)
    test_y = test_y.to(args.device)
    cfg = AIDConfig(
        epochs=args.epochs,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        outer_lr=args.outer_lr,
    )
    if args.alg == "AID-KFAC":
        try:
            from curvlinops import FisherType
            from curvlinops.weighted_ce_loss import CrossEntropyLossWeighted
        except Exception as exc:
            raise RuntimeError(
                "curvlinops is required for AID-KFAC. Install it before running."
            ) from exc

        wce = CrossEntropyLossWeighted(num_data=train[0].shape[0], reduction="mean").to(args.device)

        def kfac_loss_builder(lam, batch_hxx):
            _, y_h, idx_h = batch_hxx
            wce.data_weights.data = lam.data.clone()
            target_and_data_idx = torch.cat(
                (y_h.reshape(-1, 1), idx_h.reshape(-1, 1).to(lam.device)),
                dim=1,
            )
            return wce, target_and_data_idx

        kfac_cfg = KFACConfig(
            damping=args.damping,
            fisher_type=FisherType.MC,
            mc_samples=args.kfac_mc_samples,
        )
        solver = AIDKFACSolver(
            cfg,
            kfac_cfg=kfac_cfg,
            kfac_loss_builder=kfac_loss_builder,
            inner_optimizer=lambda params: torch.optim.SGD(params, lr=cfg.inner_lr, momentum=0.9),
            outer_optimizer=lambda params: torch.optim.SGD(params, lr=cfg.outer_lr, momentum=0.9),
        )
    else:
        solver = AIDCGSolver(
            cfg,
            inner_optimizer=lambda params: torch.optim.SGD(params, lr=cfg.inner_lr, momentum=0.9),
            outer_optimizer=lambda params: torch.optim.SGD(params, lr=cfg.outer_lr, momentum=0.9),
        )
    state = {"total_time": 0.0, "last_time": time.time()}
    def on_epoch_end(epoch: int, model: nn.Module, w: torch.Tensor, p: AIDProblem):
        now = time.time()
        state["total_time"] += now - state["last_time"]
        state["last_time"] = now
        with torch.no_grad():
            w.clamp_(0.0, 1.0)
            model.eval()
            val_batch = next(iter(p.dataloader("outer", cfg.batch_size)))
            val_loss = p.outer_loss(model, w, val_batch).item()
            logits = model(test_x)
            test_loss = F.cross_entropy(logits, test_y, reduction="mean").item()
            w_min = w.min().item()
            w_max = w.max().item()
            w_mean = w.mean().item()
            print(
                f"[info] epoch {epoch:5d} | val loss {val_loss:6.4f} | "
                f"test loss {test_loss:6.4f} | "
                f"time {state['total_time']:6.2f} | w-min {w_min:4.2f} "
                f"w-max {w_max:4.2f} | w-mean {w_mean:4.2f}"
            )
        return {
            "val_loss": val_loss,
            "test_loss": test_loss,
            "time": state["total_time"],
            "w_min": w_min,
            "w_max": w_max,
            "w_mean": w_mean,
        }

    model, w, history = solver.solve(problem, callback=on_epoch_end)


if __name__ == "__main__":
    main()
