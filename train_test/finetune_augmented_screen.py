#!/usr/bin/env python3
"""Finetune BioClinical-ModernBERT-large on the BERT screen, with vs without the molecular-framed
augmentation positives, and measure recovery on the gene-present BERT-negative recall misses
(Reviewer 3 R3.4 / Reviewer 1 R1.3).

The augmentation targets the under-represented molecular-framed-human register. Success = the
augmented model classifies more of the gene-present BERT-negative misses as positive (recovery),
WITHOUT losing test-set F1 — and the gain holds on gene-HELD-OUT misses (generalisation, not
memorisation). Augmentation rows are train-fold only by construction.
"""
from __future__ import annotations

import argparse
import gc
import hashlib

import numpy as np
import pandas as pd

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, train_bs=32, eval_bs=32, lr=1.736e-5, weight_decay=0.3)


def fold(gene: str) -> str:
    return "heldout" if int(hashlib.md5(gene.encode()).hexdigest(), 16) % 5 == 0 else "train"


def augmentation_dataset(csv: str, features, n_pos=None):
    """n_pos: keep only the first n_pos positives (CSV is miss-gene-sorted, highest-value first)
    plus all confirmed negatives — for the learning curve."""
    from datasets import Dataset
    c = pd.read_csv(csv, dtype=str).fillna("")
    cp = c["confirm_positive"].str.strip().str.lower()
    c = c[cp.isin(["0", "1", "yes", "no", "true", "false", "y", "n"])].copy()
    c["label"] = cp.loc[c.index].map(lambda v: 1 if v in ("1", "yes", "true", "y") else 0)
    c["tiab"] = (c["title"] + " " + c["abstract"]).str.strip()
    c["g2p_lgmde"] = ""
    if n_pos is not None:
        c = pd.concat([c[c["label"] == 1].head(n_pos), c[c["label"] == 0]])
    ds = Dataset.from_pandas(c[["tiab", "g2p_lgmde", "label"]].reset_index(drop=True))
    return ds.cast(features)


def load_misses(csv: str, ddg2p: str) -> pd.DataFrame:
    m = pd.read_csv(csv, dtype=str).fillna("").drop_duplicates("pmid")
    dd = pd.read_csv(ddg2p, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    g2gene = dict(zip(dd["g2p id"], dd["gene symbol"].str.strip()))
    m["tiab"] = (m["title"] + " " + m["abstract"]).str.strip()
    m["fold"] = m["g2p"].map(lambda g: fold(g2gene.get(g, "")))
    return m


def metrics_fn():
    import evaluate
    acc, pr, rc, f1 = (evaluate.load(x) for x in ["accuracy", "precision", "recall", "f1"])

    def cm(ep):
        logits, labels = ep
        p = np.argmax(logits, axis=-1)
        return {"accuracy": acc.compute(predictions=p, references=labels)["accuracy"],
                "precision": pr.compute(predictions=p, references=labels, zero_division=0)["precision"],
                "recall": rc.compute(predictions=p, references=labels, zero_division=0)["recall"],
                "f1": f1.compute(predictions=p, references=labels)["f1"]}
    return cm


def train_and_eval(train_ds, test_ds, tokenizer, out_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    tok = lambda b: tokenizer(b["tiab"], truncation=True, max_length=512)  # noqa: E731
    tt = train_ds.map(tok, batched=True)
    te = test_ds.map(tok, batched=True)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=HP["epochs"],
        per_device_train_batch_size=HP["train_bs"], per_device_eval_batch_size=HP["eval_bs"],
        learning_rate=HP["lr"], weight_decay=HP["weight_decay"], logging_steps=100,
        report_to="none", save_strategy="no", bf16=torch.cuda.is_available())
    tr = Trainer(model=model, args=args, train_dataset=tt, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=metrics_fn())
    tr.train()
    m = tr.evaluate(te)
    return model, {k: round(float(m[f"eval_{k}"]), 4) for k in ("precision", "recall", "f1")}


def score_positive(model, tokenizer, texts):
    import torch
    model.eval()
    preds = []
    for i in range(0, len(texts), 64):
        enc = tokenizer(list(texts[i:i + 64]), truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            preds += model(**enc).logits.argmax(-1).cpu().tolist()
    return np.array(preds)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="dir with ds_bert_train, ds_test")
    ap.add_argument("--aug_csv", required=True)
    ap.add_argument("--misses_csv", required=True, help="bert_negative_gene_present.csv")
    ap.add_argument("--ddg2p", required=True)
    ap.add_argument("--out_csv", default="revision/external_recall/augmented_screen_results.csv")
    ap.add_argument("--aug_sizes", default=None,
                    help="comma-separated positive counts for a learning curve, e.g. 0,50,100,150,200,232")
    ap.add_argument("--dry_run", action="store_true", help="data prep only, no training (CPU check)")
    return ap.parse_args()


def main():
    args = parse_args()
    from datasets import concatenate_datasets, load_from_disk
    ds_train = load_from_disk(f"{args.data_dir}/ds_bert_train")
    ds_test = load_from_disk(f"{args.data_dir}/ds_test")
    aug = augmentation_dataset(args.aug_csv, ds_train.features)
    misses = load_misses(args.misses_csv, args.ddg2p)
    print(f"ds_bert_train {ds_train.num_rows} | aug {aug.num_rows} (pos {sum(aug['label'])}) | "
          f"ds_test {ds_test.num_rows} | misses {len(misses)} "
          f"(heldout-gene {int((misses.fold=='heldout').sum())}, train-gene {int((misses.fold=='train').sum())})")
    if args.dry_run:
        print("dry_run OK: datasets aligned, features cast, misses fold-tagged.")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    if args.aug_sizes:  # learning curve over number of augmentation positives
        runs = [(n, ds_train if n == 0 else
                 concatenate_datasets([ds_train, augmentation_dataset(args.aug_csv, ds_train.features, n_pos=n)]))
                for n in (int(x) for x in args.aug_sizes.split(","))]
        key = "n_aug_pos"
    else:
        runs = [("baseline", ds_train), ("augmented", concatenate_datasets([ds_train, aug]))]
        key = "variant"
    for label, tr_ds in runs:
        print(f"\n=== {key}={label}: train {tr_ds.num_rows} ===", flush=True)
        model, f1 = train_and_eval(tr_ds, ds_test, tokenizer, f"./_aug_{label}")
        misses["pred"] = score_positive(model, tokenizer, misses["tiab"])
        rows.append({key: label, **f1,
                     "miss_recovery_pct": round(100 * misses["pred"].mean(), 1),
                     "miss_recovery_heldout_pct": round(100 * misses.loc[misses.fold == "heldout", "pred"].mean(), 1),
                     "miss_recovery_trainfold_pct": round(100 * misses.loc[misses.fold == "train", "pred"].mean(), 1)})
        print(rows[-1])
        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== RESULTS ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
