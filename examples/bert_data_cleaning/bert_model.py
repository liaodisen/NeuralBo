from __future__ import annotations

from typing import Union

import torch.nn as nn
from transformers import BertModel
import re


def _is_layer_to_unfreeze(param_name: str, first_unfrozen_idx: int, order: str = "reverse") -> bool:
    m = re.match(r"encoder\.layer\.(\d+)\.", param_name)
    if order == "reverse":
        return m is not None and int(m.group(1)) >= first_unfrozen_idx
    return m is not None and int(m.group(1)) < first_unfrozen_idx


def _is_specific_layer(param_name: str, layer_idx: int) -> bool:
    m = re.match(r"encoder\.layer\.(\d+)\.", param_name)
    return m is not None and int(m.group(1)) == layer_idx


class BertClassifier(nn.Module):
    """
    BERT classifier with configurable layer freezing.

    `fine_tune_level`:
    - 0: classifier head only (default)
    - 1..12: unfreeze that many BERT encoder layers
    - -1 or "all": unfreeze all BERT layers
    """

    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "bert-base-uncased",
        fine_tune_level: Union[int, str] = 0,
        order: str = "reverse",
        specific_layer: int | None = None,
    ):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.set_fine_tune_level(
            level=fine_tune_level,
            order=order,
            verbose=False,
            specific_layer=specific_layer,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        return self.fc(self.dropout(pooled))

    def set_fine_tune_level(
        self,
        level: Union[int, str] = 0,
        order: str = "reverse",
        *,
        verbose: bool = False,
        specific_layer: int | None = None,
    ):
        if isinstance(level, str):
            if level.lower() in {"all", "full"}:
                level = -1
            else:
                raise ValueError(f"Unknown level string: {level}")

        if not isinstance(level, int) or not (-1 <= level <= 12):
            raise ValueError("fine_tune_level must be -1, 0..12, or 'all'.")

        # Freeze all BERT params first.
        for p in self.bert.parameters():
            p.requires_grad = False

        # Unfreeze selected BERT params.
        if level == -1:
            for p in self.bert.parameters():
                p.requires_grad = True
        elif level > 0:
            if specific_layer is not None:
                for name, p in self.bert.named_parameters():
                    if _is_specific_layer(name, specific_layer):
                        p.requires_grad = True
            else:
                if order == "reverse":
                    first_unfrozen = 12 - level
                else:
                    first_unfrozen = level
                for name, p in self.bert.named_parameters():
                    if _is_layer_to_unfreeze(name, first_unfrozen_idx=first_unfrozen, order=order):
                        p.requires_grad = True

        # Classifier head always trainable.
        for p in self.fc.parameters():
            p.requires_grad = True

        if verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"[bert] trainable params: {trainable}/{total}")


class BertWrapper(nn.Module):
    def __init__(self, bertclassifier, attention_mask, token_type_ids):
        super(BertWrapper, self).__init__()
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.model = bertclassifier
        
    def forward(self, x):
        return self.model(input_ids = x, attention_mask = self.attention_mask, token_type_ids = self.token_type_ids)
