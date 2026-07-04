#!/usr/bin/env python3
"""Seed-averaged evaluation of the high-recall screen (R1.1/R3.3 reporting rigour).

Trains the CV-selected config (BioClinical-ModernBERT-large, lr 3e-5 / wd 0.1 / 5 epochs) on
the fixed keeper training set (ds_hirecall_train = annotated + augmentation + external train-fold)
under several random seeds, and reports mean +/- std of test F1 and held-out-fold external recall
(per source). Varying only the seed isolates training-randomness noise (head init, shuffling) so
the reported number is an expectation, not one lucky/unlucky run. The shipped model is unchanged;
this only produces the defensible reported figure.
"""
from __future__ import annotations

import argparse
import gc
import hashlib

import numpy as np
import pandas as pd

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, lr=3e-5, wd=0.1)  # CV-selected on ds_hirecall_train


def metrics_fn():
    import evaluate
    pr, rc, f1 = (evaluate.load(x) for x in ["precision", "recall", "f1"])

    def cm(ep):
        logits, labels = ep
        p = np.argmax(logits, axis=-1)
        return {"precision": pr.compute(predictions=p, references=labels, zero_division=0)["precision"],
                "recall": rc.compute(predictions=p, references=labels, zero_division=0)["recall"],
                "f1": f1.compute(predictions=p, references=labels)["f1"]}
    return cm


def train_once(ds_train, ds_test, tokenizer, seed, out_dir):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    set_seed(seed)  # BEFORE model init so the classification-head init varies with the seed
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    tt = ds_train.map(lambda b: tokenizer(b["tiab"], truncation=True, max_length=512), batched=True)
    te = ds_test.map(lambda b: tokenizer(b["tiab"], truncation=True, max_length=512), batched=True)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=HP["epochs"], seed=seed, data_seed=seed,
        per_device_train_batch_size=HP["bs"], per_device_eval_batch_size=HP["bs"], learning_rate=HP["lr"],
        weight_decay=HP["wd"], logging_steps=300, report_to="none", save_strategy="no", bf16=torch.cuda.is_available())
    tr = Trainer(model=model, args=args, train_dataset=tt, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8), compute_metrics=metrics_fn())
    tr.train()
    m = tr.evaluate(te)
    return model, {k: round(float(m[f"eval_{k}"]), 4) for k in ("precision", "recall", "f1")}


def score_proba(model, tokenizer, texts):
    import torch
    model.eval()
    out = []
    for i in range(0, len(texts), 64):
        enc = tokenizer(list(texts[i:i + 64]), truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out += torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
    return np.array(out)


def b10(gene):
    return int(hashlib.md5(gene.encode()).hexdigest(), 16) % 10


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_ds_dir", required=True, help="ds_hirecall_train (keeper training set)")
    ap.add_argument("--test_ds_dir", required=True, help="dir containing ds_test")
    ap.add_argument("--external_csv", required=True)
    ap.add_argument("--random_csv", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--save_dir", default=None,
                    help="if set, save the model+tokenizer for the LAST seed (the shipped LitDD checkpoint)")
    ap.add_argument("--out_csv", default="revision/external_recall/seed_results.csv")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    from datasets import load_from_disk
    ds_train = load_from_disk(args.train_ds_dir)
    ds_test = load_from_disk(f"{args.test_ds_dir}/ds_test")
    ext = pd.read_csv(args.external_csv, dtype=str).drop_duplicates("pmid").copy()
    gb = ext.groupby("pmid")["gene"].apply(lambda gs: {b10(g) for g in gs})
    ho = ext[ext["pmid"].map(gb).map(lambda s: s == {0})].reset_index(drop=True)  # 10% gene-fold held-out
    rnd = pd.read_csv(args.random_csv, dtype=str).fillna("")
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"train {ds_train.num_rows} | test {ds_test.num_rows} | held-out {len(ho)} "
          f"({ho['source'].value_counts().to_dict()}) | seeds {seeds}")
    if args.dry_run:
        print("dry_run OK")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        model, f1 = train_once(ds_train, ds_test, tokenizer, seed, f"./_seed_{seed}")
        ho_p = score_proba(model, tokenizer, ho["tiab"])
        rnd_p = score_proba(model, tokenizer, rnd["tiab"])
        row = {"seed": seed, **f1, "random_fpr_pct": round(100 * (rnd_p >= 0.5).mean(), 2)}
        for s in ["premined", "hpoa", "clingen"]:
            m = (ho["source"] == s).values
            row[f"heldout_recall_{s}_pct"] = round(100 * (ho_p[m] >= 0.5).mean(), 1)
        row["heldout_recall_all_pct"] = round(100 * (ho_p >= 0.5).mean(), 1)
        rows.append(row)
        print(row)
        if args.save_dir and seed == seeds[-1]:
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f"  saved LitDD screen checkpoint (seed {seed}) -> {args.save_dir}")
        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    num = out.select_dtypes("number").drop(columns=["seed"])
    print("\n=== PER SEED ===")
    print(out.to_string(index=False))
    print("\n=== MEAN +/- STD over seeds ===")
    for c in num.columns:
        print(f"  {c}: {num[c].mean():.2f} +/- {num[c].std():.2f}")


if __name__ == "__main__":
    main()
