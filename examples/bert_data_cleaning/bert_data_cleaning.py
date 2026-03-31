"""
Minimal BERT data-cleaning example on TREC using the shared AID solver.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.problem import AIDProblem
from src.solver.aid import AIDCGSolver, AIDKFACSolver, AIDConfig, KFACConfig
from bert_model import BertClassifier, BertWrapper

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None
    sdpa_kernel = None


@contextmanager
def sdpa_math_context(enabled: bool = True):
    """
    Force SDPBackend.MATH when available, else no-op.
    This keeps behavior stable across Torch versions.
    """
    if not enabled or sdpa_kernel is None or SDPBackend is None:
        with nullcontext():
            yield
        return

    try:
        ctx = sdpa_kernel(SDPBackend.MATH)
    except TypeError:
        ctx = sdpa_kernel(backends=[SDPBackend.MATH])
    with ctx:
        yield


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sorted_records(raw: dict) -> Iterable[dict]:
    for key in sorted(raw.keys(), key=int):
        yield raw[key]


def load_trec_split(path: Path) -> tuple[list[str], torch.Tensor]:
    raw = json.loads(path.read_text())
    texts: list[str] = []
    labels: list[int] = []
    for rec in _sorted_records(raw):
        texts.append(rec["data"]["text"])
        labels.append(int(rec["label"]))
    return texts, torch.tensor(labels, dtype=torch.long)


def encode_texts(tokenizer: AutoTokenizer, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


class BertDataCleaningProblem(AIDProblem):
    def __init__(
        self,
        train_texts: list[str],
        noisy_train_labels: torch.Tensor,
        val_texts: list[str],
        val_labels: torch.Tensor,
        model_name: str,
        max_len: int,
        device: str = "cpu",
        batch_size: int = 16,
        use_math_sdpa: bool = True,
        wrap_for_kfac: bool = False,
        fine_tune_level: int | str = 0,
        fine_tune_order: str = "reverse",
        specific_layer: int | None = None,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_name = model_name
        self.use_math_sdpa = use_math_sdpa
        self.wrap_for_kfac = wrap_for_kfac
        self.fine_tune_level = fine_tune_level
        self.fine_tune_order = fine_tune_order
        self.specific_layer = specific_layer
        self._model_ref: nn.Module | None = None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_tok = encode_texts(tokenizer, train_texts, max_len=max_len)
        val_tok = encode_texts(tokenizer, val_texts, max_len=max_len)

        train_n = noisy_train_labels.shape[0]
        val_n = val_labels.shape[0]

        self._train_ds = torch.utils.data.TensorDataset(
            train_tok["input_ids"].to(self.device),
            train_tok["attention_mask"].to(self.device),
            train_tok["token_type_ids"].to(self.device),
            noisy_train_labels.long().to(self.device),
            torch.arange(train_n, dtype=torch.long, device=self.device),
        )
        self._val_ds = torch.utils.data.TensorDataset(
            val_tok["input_ids"].to(self.device),
            val_tok["attention_mask"].to(self.device),
            val_tok["token_type_ids"].to(self.device),
            val_labels.long().to(self.device),
            torch.arange(val_n, dtype=torch.long, device=self.device),
        )

    def _get_loader(self, dataset, batch_size: int):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _forward_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        with sdpa_math_context(enabled=self.use_math_sdpa):
            if isinstance(model, BertWrapper):
                model.attention_mask = attention_mask
                model.token_type_ids = token_type_ids
                return model(input_ids)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.logits if hasattr(out, "logits") else out

    def init(self):
        base_model = BertClassifier(
            num_labels=6,
            model_name=self.model_name,
            fine_tune_level=self.fine_tune_level,
            order=self.fine_tune_order,
            specific_layer=self.specific_layer,
        ).to(self.device)
        if self.wrap_for_kfac:
            model = BertWrapper(base_model, attention_mask=None, token_type_ids=None).to(self.device)
        else:
            model = base_model
        self._model_ref = model
        w = torch.full((len(self._train_ds),), 0.5, dtype=torch.float32, device=self.device, requires_grad=True)
        return model, w

    @property
    def model_ref(self) -> nn.Module:
        if self._model_ref is None:
            raise RuntimeError("Model is not initialized yet. Call solver.solve(problem) first.")
        return self._model_ref

    def dataloader(self, role: str, batch_size: int):
        self.validate_role(role)
        if role == "outer":
            return self._get_loader(self._val_ds, batch_size)
        return self._get_loader(self._train_ds, batch_size)

    def inner_loss(self, model: nn.Module, w: torch.Tensor, batch: Any) -> torch.Tensor:
        input_ids, attention_mask, token_type_ids, labels, index = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        labels = labels.to(self.device)
        index = index.to(self.device)

        logits = self._forward_logits(model, input_ids, attention_mask, token_type_ids)
        per_example = F.cross_entropy(logits, labels, reduction="none")
        return (w[index].clamp(0.0, 1.0) * per_example).mean()

    def outer_loss(self, model: nn.Module, w: torch.Tensor, batch: Any) -> torch.Tensor:
        input_ids, attention_mask, token_type_ids, labels, _ = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        labels = labels.to(self.device)
        logits = self._forward_logits(model, input_ids, attention_mask, token_type_ids)
        return F.cross_entropy(logits, labels, reduction="mean")

    def project_lam(self, w: torch.Tensor):
        return w.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(model: nn.Module, dataset, batch_size: int, device: torch.device, use_math_sdpa: bool):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        input_ids, attention_mask, token_type_ids, labels, _ = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        with sdpa_math_context(enabled=use_math_sdpa):
            if isinstance(model, BertWrapper):
                model.attention_mask = attention_mask
                model.token_type_ids = token_type_ids
                logits = model(input_ids)
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = out.logits if hasattr(out, "logits") else out
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.shape[0]
    return total_loss / total, 100.0 * total_correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="BERT data cleaning on TREC with shared AID solver.")
    default_data_dir = Path(__file__).resolve().parents[1] / "trec"
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir))
    parser.add_argument("--label_file", type=str, default="hard_labels.npy")
    parser.add_argument("--alg", type=str, default="AID-KFAC", choices=["AID-CG", "AID-KFAC"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--fine_tune_level",
        type=int,
        default=0,
        help="0=head only (default), 1..12=number of BERT encoder layers to unfreeze, -1=all.",
    )
    parser.add_argument(
        "--fine_tune_order",
        type=str,
        default="reverse",
        choices=["reverse", "forward"],
        help="Layer unfreeze order when fine_tune_level > 0.",
    )
    parser.add_argument(
        "--specific_layer",
        type=int,
        default=None,
        help="If set, only unfreezes this BERT encoder layer index [0..11].",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--x_lr", type=float, default=1e-5)
    parser.add_argument("--w_lr", type=float, default=1.0)
    parser.add_argument("--w_momentum", type=float, default=0.9)
    parser.add_argument("--cg_maxiter", type=int, default=3)
    parser.add_argument("--cg_tol", type=float, default=1e-10)
    parser.add_argument("--cg_damping", type=float, default=1e-3)
    parser.add_argument("--kfac_damping", type=float, default=1e-3)
    parser.add_argument("--kfac_mc_samples", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument(
        "--disable_math_sdpa",
        action="store_true",
        help="Disable forcing SDPBackend.MATH for forward/hypergradient paths.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    train_texts, _ = load_trec_split(data_dir / "train.json")
    val_texts, val_labels = load_trec_split(data_dir / "valid.json")
    test_texts, test_labels = load_trec_split(data_dir / "test.json")
    noisy_train_labels = torch.from_numpy(np.load(data_dir / args.label_file)).long()

    if len(train_texts) != noisy_train_labels.shape[0]:
        raise ValueError(f"train.json size and {args.label_file} size mismatch.")

    problem = BertDataCleaningProblem(
        train_texts=train_texts,
        noisy_train_labels=noisy_train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        model_name=args.model_name,
        max_len=args.max_len,
        device=str(device),
        batch_size=args.batch_size,
        use_math_sdpa=not args.disable_math_sdpa,
        wrap_for_kfac=args.alg == "AID-KFAC",
        fine_tune_level=args.fine_tune_level,
        fine_tune_order=args.fine_tune_order,
        specific_layer=args.specific_layer,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_tok = encode_texts(tokenizer, test_texts, max_len=args.max_len)
    test_ds = torch.utils.data.TensorDataset(
        test_tok["input_ids"],
        test_tok["attention_mask"],
        test_tok["token_type_ids"],
        test_labels.long(),
        torch.arange(test_labels.shape[0], dtype=torch.long),
    )

    cfg = AIDConfig(
        batch_size=args.batch_size,
        inner_steps=args.inner_steps,
        inner_lr=args.x_lr,
        outer_lr=args.w_lr,
        cg_maxiter=args.cg_maxiter,
        cg_tol=args.cg_tol,
        cg_damping=args.cg_damping,
        epochs=args.epochs,
    )

    if args.alg == "AID-KFAC":
        try:
            from curvlinops import FisherType
            from curvlinops.weighted_ce_loss import CrossEntropyLossWeighted
        except Exception as exc:
            raise RuntimeError("AID-KFAC requires curvlinops. Install it before running.") from exc

        wce = CrossEntropyLossWeighted(num_data=len(train_texts), reduction="mean").to(device)

        def kfac_loss_builder(lam, batch_hxx):
            _, attention_mask, token_type_ids, labels, data_index = batch_hxx
            model_ref = problem.model_ref
            if not isinstance(model_ref, BertWrapper):
                raise RuntimeError("AID-KFAC expects BertWrapper model for forward-only input.")

            model_ref.attention_mask = attention_mask.to(device)
            model_ref.token_type_ids = token_type_ids.to(device)
            wce.data_weights.data = lam.data.clone()

            targets = torch.cat(
                (
                    labels.reshape(-1, 1).to(lam.device),
                    data_index.reshape(-1, 1).to(lam.device),
                ),
                dim=1,
            )
            return wce, targets

        kfac_cfg = KFACConfig(
            damping=args.kfac_damping,
            fisher_type=FisherType.MC,
            mc_samples=args.kfac_mc_samples,
        )
        solver = AIDKFACSolver(
            cfg,
            kfac_cfg=kfac_cfg,
            kfac_loss_builder=kfac_loss_builder,
            inner_optimizer=lambda params: AdamW(params, lr=cfg.inner_lr),
            outer_optimizer=lambda params: torch.optim.SGD(params, lr=cfg.outer_lr, momentum=args.w_momentum),
        )
    else:
        solver = AIDCGSolver(
            cfg,
            inner_optimizer=lambda params: AdamW(params, lr=cfg.inner_lr),
            outer_optimizer=lambda params: torch.optim.SGD(params, lr=cfg.outer_lr, momentum=args.w_momentum),
        )

    state = {"total_time": 0.0, "last_time": time.time()}

    def on_epoch_end(epoch: int, model: nn.Module, w: torch.Tensor, p: AIDProblem):
        now = time.time()
        state["total_time"] += now - state["last_time"]
        state["last_time"] = now
        with torch.no_grad():
            w.clamp_(0.0, 1.0)

        if (epoch + 1) % args.eval_step != 0:
            return None

        val_batch = next(iter(p.dataloader("outer", cfg.batch_size)))
        with torch.no_grad():
            val_loss = p.outer_loss(model, w, val_batch).item()
            test_loss, test_acc = evaluate(
                model,
                dataset=test_ds,
                batch_size=args.eval_batch_size,
                device=device,
                use_math_sdpa=not args.disable_math_sdpa,
            )
        model.train()
        print(
            f"[info] epoch {epoch:3d} | val loss {val_loss:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_acc:5.2f} | "
            f"time {state['total_time']:7.2f}s | w-min {w.min().item():.3f} | w-max {w.max().item():.3f}"
        )
        return {
            "val_loss": val_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time": state["total_time"],
        }

    solver.solve(problem, callback=on_epoch_end)


if __name__ == "__main__":
    main()
