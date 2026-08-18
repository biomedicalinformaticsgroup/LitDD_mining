#!/usr/bin/env python3
"""External-recall LEARNING CURVE (R3.3): is the +external screen saturated, or does adding
more genes keep lifting held-out recall?

Fixes a 10% gene-fold held-out eval (genes with md5%10==0 — a subset of the old 20% held-out,
so still never trained), then feeds progressively more of the remaining 90% of genes into
training (buckets 1..k) and measures held-out-fold external recall per source at each size. A
rising curve = room to learn (push toward max training / more annotation); a flat curve =
saturated. Base = BioClinical + augmentation, as in finetune_external_recall.py.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os

import numpy as np
import pandas as pd

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, lr=1.736e-5, wd=0.3)


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


def train_and_eval(ds_train, ds_test, tokenizer, out_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    tt = ds_train.map(lambda b: tokenizer(b["tiab"], truncation=True, max_length=512), batched=True)
    te = ds_test.map(lambda b: tokenizer(b["tiab"], truncation=True, max_length=512), batched=True)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=HP["epochs"],
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


def as_positive_ds(df, features):
    from datasets import Dataset
    d = pd.DataFrame({"label": 1, "tiab": df["tiab"].values, "g2p_lgmde": df["g2p_id"].values})
    return Dataset.from_pandas(d, preserve_index=False).cast(features)


def aug_ds(csv, features):
    from datasets import Dataset
    a = pd.read_csv(csv, dtype=str).fillna("")
    cp = a["confirm_positive"].str.strip().str.lower()
    a = a[cp.isin(["0", "1", "yes", "no", "true", "false", "y", "n"])].copy()
    a["label"] = cp.loc[a.index].map(lambda v: 1 if v in ("1", "yes", "true", "y") else 0)
    a["tiab"] = (a["title"] + " " + a["abstract"]).str.strip()
    d = pd.DataFrame({"label": a["label"].astype(int).values, "tiab": a["tiab"].values,
                      "g2p_lgmde": a["g2p_id"].values})
    return Dataset.from_pandas(d, preserve_index=False).cast(features)


def bucket(gene, mod=10):
    return int(hashlib.md5(gene.encode()).hexdigest(), 16) % mod


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--external_csv", required=True)
    ap.add_argument("--aug_csv", required=True)
    ap.add_argument("--random_csv", required=True)
    ap.add_argument("--levels", default="0,1,3,6,9", help="cumulative gene-bucket levels (of 9) to add")
    ap.add_argument("--out_csv", default="revision/external_recall/external_curve_results.csv")
    ap.add_argument("--dump_scores", default="revision/external_recall/external_curve_scores")
    ap.add_argument("--save_dir", default=None,
                    help="if set, save model+tokenizer here (deployment checkpoint) after the last level")
    ap.add_argument("--lr", type=float, default=None, help="override CV-selected learning rate")
    ap.add_argument("--wd", type=float, default=None, help="override CV-selected weight decay")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.lr is not None:
        HP["lr"] = args.lr
    if args.wd is not None:
        HP["wd"] = args.wd
    if args.epochs is not None:
        HP["epochs"] = args.epochs
    print(f"HP: {HP}")
    from datasets import concatenate_datasets, load_from_disk
    ds_train = load_from_disk(f"{args.data_dir}/ds_bert_train")
    ds_test = load_from_disk(f"{args.data_dir}/ds_test")

    ext = pd.read_csv(args.external_csv, dtype=str).drop_duplicates("pmid").copy()
    # per-paper gene buckets under md5%10; held-out = all genes bucket 0 (subset of old 20% held-out)
    buckets = ext.groupby("pmid")["gene"].apply(lambda gs: {bucket(g) for g in gs})
    ext["gbuckets"] = ext["pmid"].map(buckets)
    ext["is_heldout"] = ext["gbuckets"].map(lambda b: b == {0})
    ext["is_mixed"] = ext["gbuckets"].map(lambda b: (0 in b) and b != {0})
    ext["level"] = ext["gbuckets"].map(lambda b: max(b) if 0 not in b else None)  # trainable at max-bucket
    heldout = ext[ext["is_heldout"]].reset_index(drop=True)
    trainable = ext[(~ext["is_heldout"]) & (~ext["is_mixed"])]
    levels = [int(x) for x in args.levels.split(",")]
    print(f"held-out (10% genes) {len(heldout)} ({heldout['source'].value_counts().to_dict()}) | "
          f"trainable pool {len(trainable)} | levels {levels}")
    for k in levels:
        n = (trainable["level"] <= k).sum() if k > 0 else 0
        print(f"  level {k}/9 -> +{n} external train papers")
    if args.dry_run:
        print("dry_run OK")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    base = concatenate_datasets([ds_train, aug_ds(args.aug_csv, ds_train.features)])
    rnd = pd.read_csv(args.random_csv, dtype=str).fillna("")
    os.makedirs(args.dump_scores, exist_ok=True)
    rows = []
    for k in levels:
        add = trainable[trainable["level"] <= k] if k > 0 else trainable.iloc[:0]
        tr_ds = base if k == 0 else concatenate_datasets([base, as_positive_ds(add, ds_train.features)])
        print(f"\n=== level {k}/9: +{len(add)} external, train {tr_ds.num_rows} ===", flush=True)
        model, f1 = train_and_eval(tr_ds, ds_test, tokenizer, f"./_curve_{k}")
        ho_p = score_proba(model, tokenizer, heldout["tiab"])  # also warms up the GPU
        import time
        t0 = time.time()
        rnd_p = score_proba(model, tokenizer, rnd["tiab"])
        aps = round(len(rnd) / (time.time() - t0), 1)  # end-to-end inference throughput (tokenise+forward)
        print(f"  inference throughput: {aps} abstracts/sec ({len(rnd)} abstracts)")
        row = {"level": k, "n_external": len(add), **f1, "random_fpr_pct": round(100 * (rnd_p >= 0.5).mean(), 2),
               "infer_abstracts_per_sec": aps}
        if args.save_dir and k == levels[-1]:
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f"  saved deployment checkpoint -> {args.save_dir}")
        for s in ["premined", "hpoa", "clingen"]:
            m = (heldout["source"] == s).values
            row[f"heldout_recall_{s}_pct"] = round(100 * (ho_p[m] >= 0.5).mean(), 1) if m.any() else None
        row["heldout_recall_all_pct"] = round(100 * (ho_p >= 0.5).mean(), 1)
        rows.append(row)
        print(row)
        pd.DataFrame({"source": heldout["source"].values, "proba": ho_p}).to_csv(
            f"{args.dump_scores}/level{k}_heldout.csv", index=False)
        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== EXTERNAL-RECALL LEARNING CURVE (fixed 10% held-out) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
