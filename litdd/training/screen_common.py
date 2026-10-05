"""Helpers shared by the screen-training and screen-evaluation scripts.

Pure-Python helpers (gene folds, worksheet parsing, grid parsing, CSV bookkeeping,
confusion counts) import nothing beyond numpy and pandas. The helpers that train,
score or build HuggingFace datasets import torch, transformers, datasets and
evaluate inside the function body, so this module imports without a GPU stack.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# Tokens accepted in the ``confirm_positive`` column of an annotation worksheet,
# and the subset that means "positive". Any other value marks the row as unannotated.
CONFIRM_TOKENS = frozenset({"0", "1", "yes", "no", "true", "false", "y", "n"})
POSITIVE_TOKENS = frozenset({"1", "yes", "true", "y"})


# --------------------------------------------------------------------------- #
# Pure-Python helpers
# --------------------------------------------------------------------------- #
def gene_fold(gene: str, modulus: int) -> int:
    """Deterministic bucket in ``range(modulus)`` for a gene symbol (md5 of the symbol).

    Callers hold out bucket 0: with ``modulus=5`` one fifth of genes, with ``modulus=10``
    one tenth.
    """
    return int(hashlib.md5(str(gene).encode()).hexdigest(), 16) % modulus


def fold_name(gene: str, modulus: int) -> str:
    """``"heldout"`` when ``gene_fold(gene, modulus) == 0``, otherwise ``"train"``."""
    return "heldout" if gene_fold(gene, modulus) == 0 else "train"


def parse_confirm_positive(value: object) -> int | None:
    """Map a ``confirm_positive`` cell to 1, 0 or None (blank or unrecognised token)."""
    token = str(value).strip().lower()
    if token not in CONFIRM_TOKENS:
        return None
    return 1 if token in POSITIVE_TOKENS else 0


def confirmed_worksheet(worksheet: pd.DataFrame) -> pd.DataFrame:
    """Rows of an annotation worksheet whose ``confirm_positive`` holds an accepted token.

    Returns a copy with an integer ``label`` column and a ``tiab`` column built from
    ``title`` and ``abstract`` (single space between them, outer whitespace stripped).
    """
    labels = worksheet["confirm_positive"].map(parse_confirm_positive)
    out = worksheet[labels.notna()].copy()
    out["label"] = labels.loc[out.index].astype(int)
    out["tiab"] = (out["title"].fillna("") + " " + out["abstract"].fillna("")).str.strip()
    return out


def gene_fullnames(gene_info_gz: str | None) -> dict[str, str]:
    """Symbol to approved full name from an NCBI ``gene_info.gz``; empty when no path is given."""
    import gzip

    if not gene_info_gz:
        return {}
    out: dict[str, str] = {}
    with gzip.open(gene_info_gz, "rt") as fh:
        header = {c: i for i, c in enumerate(fh.readline().lstrip("#").rstrip().split("\t"))}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[header["description"]] not in ("", "-"):
                out[parts[header["Symbol"]]] = parts[header["description"]]
    return out


def parse_floats(values: Iterable[str]) -> list[float]:
    """Grid values given on the command line as strings, as floats."""
    return [float(x) for x in values]


def parse_ints(values: Iterable[str]) -> list[int]:
    """Grid values given on the command line as strings, as ints."""
    return [int(x) for x in values]


def maybe_load_hp_json(path: str | None) -> dict:
    """Selected hyperparameters from a CV-search JSON: its ``best`` entry, or the flat dict."""
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("best", data)


def load_existing(out_csv: str) -> set[tuple[str, str]]:
    """``(model, seed)`` pairs already present in a results CSV; empty when the file is absent."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, newline="") as f:
        return {(row["model"], str(row.get("seed", ""))) for row in csv.DictReader(f) if row.get("model")}


def append_csv_row(out_csv: str, row: dict) -> None:
    """Append one row to a CSV, writing the header first when the file does not exist."""
    is_new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


def cm(preds: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    """Confusion counts (tp, fp, fn, tn) of binary predictions against binary labels."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return {
        "tp": int(((preds == 1) & (labels == 1)).sum()),
        "fp": int(((preds == 1) & (labels == 0)).sum()),
        "fn": int(((preds == 0) & (labels == 1)).sum()),
        "tn": int(((preds == 0) & (labels == 0)).sum()),
    }


# --------------------------------------------------------------------------- #
# Metrics, scoring and training (torch / transformers / datasets / evaluate)
# --------------------------------------------------------------------------- #
def make_compute_metrics(accuracy: bool = False, counts: bool = False) -> Callable:
    """Build the ``compute_metrics`` callable for a ``transformers.Trainer``.

    The returned function takes ``(logits, labels)``, argmaxes the logits and returns
    ``precision``, ``recall`` and ``f1`` (from ``evaluate``), plus ``accuracy`` and the
    confusion counts of :func:`cm` when requested. The Trainer prefixes each key with
    ``eval_``.
    """
    import evaluate

    pr, rc, f1 = (evaluate.load(x) for x in ["precision", "recall", "f1"])
    acc = evaluate.load("accuracy") if accuracy else None

    def fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        out = {}
        if acc is not None:
            out["accuracy"] = acc.compute(predictions=preds, references=labels)["accuracy"]
        out["precision"] = pr.compute(predictions=preds, references=labels, zero_division=0)["precision"]
        out["recall"] = rc.compute(predictions=preds, references=labels, zero_division=0)["recall"]
        out["f1"] = f1.compute(predictions=preds, references=labels)["f1"]
        if counts:
            out.update(cm(preds, labels))
        return out

    return fn


def _batches(texts: Sequence[str], pair_texts: Sequence[str] | None, batch_size: int):
    for i in range(0, len(texts), batch_size):
        first = list(texts[i:i + batch_size])
        if pair_texts is None:
            yield (first,)
        else:
            yield (first, list(pair_texts[i:i + batch_size]))


def score_proba(model, tokenizer, texts: Sequence[str], max_length: int, *,
                pair_texts: Sequence[str] | None = None, batch_size: int = 64) -> np.ndarray:
    """Positive-class probability for each text (softmax over the two logits).

    ``pair_texts`` supplies the second sequence of a text pair, aligned with ``texts``.
    """
    import torch

    model.eval()
    out: list[float] = []
    for batch in _batches(texts, pair_texts, batch_size):
        enc = tokenizer(*batch, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out += torch.softmax(model(**enc).logits, dim=-1)[:, 1].float().cpu().tolist()
    return np.array(out)


def score_argmax(model, tokenizer, texts: Sequence[str], max_length: int, *,
                 pair_texts: Sequence[str] | None = None, batch_size: int = 64) -> np.ndarray:
    """Predicted class (0 or 1) for each text, the argmax of the two logits."""
    import torch

    model.eval()
    out: list[int] = []
    for batch in _batches(texts, pair_texts, batch_size):
        enc = tokenizer(*batch, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out += model(**enc).logits.argmax(-1).cpu().tolist()
    return np.array(out)


def train_and_eval(ds_train, ds_test, tokenizer, *, model_name: str, hp: dict, out_dir: str,
                   logging_steps: int, max_length: int, bf16: bool, seed: int | None = None,
                   pair_col: str | None = None, accuracy: bool = False,
                   attn_implementation: str | None = None):
    """Fine-tune a sequence classifier on ``ds_train`` and evaluate it once on ``ds_test``.

    ``hp`` holds ``epochs``, ``bs`` (train batch size; ``eval_bs`` optional, defaults to
    ``bs``; ``grad_accum`` optional, gradient-accumulation steps, default 1), ``lr`` and ``wd``. When ``seed`` is given it is set before the model is loaded,
    so the classification-head initialisation follows it, and passed to the Trainer as both
    ``seed`` and ``data_seed``. ``pair_col`` names a second text column tokenised as a pair
    with ``tiab``. ``attn_implementation`` is passed to ``from_pretrained`` when given (for
    example ``"sdpa"``, whose memory-efficient kernel avoids materialising the full attention
    matrix at 8,192 tokens when flash-attention is not installed). Returns ``(model, metrics)``
    where ``metrics`` holds precision, recall and f1 on ``ds_test`` rounded to four decimals.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if seed is not None:
        set_seed(seed)
    load_kwargs = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, **load_kwargs)

    def tok(batch):
        if pair_col is None:
            return tokenizer(batch["tiab"], truncation=True, max_length=max_length)
        return tokenizer(batch["tiab"], batch[pair_col], truncation=True, max_length=max_length)

    tokenized_train = ds_train.map(tok, batched=True)
    tokenized_test = ds_test.map(tok, batched=True)

    seed_kwargs = {} if seed is None else {"seed": seed, "data_seed": seed}
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["bs"], per_device_eval_batch_size=hp.get("eval_bs", hp["bs"]),
        gradient_accumulation_steps=hp.get("grad_accum", 1),
        learning_rate=hp["lr"], weight_decay=hp["wd"], logging_steps=logging_steps,
        report_to="none", save_strategy="no", bf16=bf16, **seed_kwargs,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized_train, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=make_compute_metrics(accuracy=accuracy),
    )
    trainer.train()
    metrics = trainer.evaluate(tokenized_test)
    return model, {k: round(float(metrics[f"eval_{k}"]), 4) for k in ("precision", "recall", "f1")}


def as_positive_ds(df: pd.DataFrame, features):
    """Positive-labelled dataset from rows with ``tiab`` and ``g2p_id``, cast to ``features``."""
    from datasets import Dataset

    d = pd.DataFrame({"label": 1, "tiab": df["tiab"].values, "g2p_lgmde": df["g2p_id"].values})
    return Dataset.from_pandas(d, preserve_index=False).cast(features)


def aug_ds(csv_path: str, features, *, n_pos: int | None = None, lgmde_col: str | None = "g2p_id"):
    """Confirmed rows of an augmentation worksheet as a dataset, plus the PMIDs it contains.

    ``n_pos`` keeps the first ``n_pos`` positives (worksheet order) together with every
    confirmed negative. ``lgmde_col`` names the worksheet column copied into ``g2p_lgmde``;
    ``None`` leaves ``g2p_lgmde`` empty.
    """
    from datasets import Dataset

    a = confirmed_worksheet(pd.read_csv(csv_path, dtype=str).fillna(""))
    if n_pos is not None:
        a = pd.concat([a[a["label"] == 1].head(n_pos), a[a["label"] == 0]])
    lgmde = a[lgmde_col].values if lgmde_col else ""
    d = pd.DataFrame({"label": a["label"].astype(int).values, "tiab": a["tiab"].values, "g2p_lgmde": lgmde})
    return Dataset.from_pandas(d, preserve_index=False).cast(features), set(a["pmid"].astype(str))
