#!/usr/bin/env python3
"""Push the BERT screen toward EvAgg's paper-level recall on the external sets (R3.3).

Goal: the screen should answer the SAME question as EvAgg — "is this a genetic-disease
paper? 1/0" — at EvAgg-level recall (~99.5% of premined/HPOA/ClinGen papers), leaving
disease assignment to the downstream cross-encoder/LLM. EvAgg cannot run PubMed-wide; the
BERT screen can, so a high-recall screen is the scalable route to EvAgg's recall.

Lever tested here: add the external truth papers (old-DDG2P premined / ClinGen / HPOA,
gene-present) to TRAINING, with strict gene-fold discipline — train on train-fold genes only,
evaluate recall on held-out-fold genes (never trained on). A held-out-fold recall lift =
generalisation of the register to unseen genes, not memorisation.

Variants: baseline (current training) vs +external-train-fold positives. Eval per variant:
paper-level recall on held-out-fold external papers (per source, = EvAgg's axis), test F1
(precision proxy), deployment FPR on random PubMed. Probabilities are dumped for offline
recall-vs-FPR threshold sweeps.
"""
from __future__ import annotations

import argparse
import gc
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

    def tok(b):
        return tokenizer(b["tiab"], truncation=True, max_length=512)
    tt = ds_train.map(tok, batched=True)
    te = ds_test.map(tok, batched=True)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=HP["epochs"],
        per_device_train_batch_size=HP["bs"], per_device_eval_batch_size=HP["bs"], learning_rate=HP["lr"],
        weight_decay=HP["wd"], logging_steps=200, report_to="none", save_strategy="no", bf16=torch.cuda.is_available())
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
    """The 250-annotation molecular augmentation set (232 pos + 18 neg) -> dataset + its PMIDs,
    so the base model matches the confirmed-best BioClinical + augmentation screen."""
    from datasets import Dataset
    a = pd.read_csv(csv, dtype=str).fillna("")
    cp = a["confirm_positive"].str.strip().str.lower()
    a = a[cp.isin(["0", "1", "yes", "no", "true", "false", "y", "n"])].copy()
    a["label"] = cp.loc[a.index].map(lambda v: 1 if v in ("1", "yes", "true", "y") else 0)
    a["tiab"] = (a["title"] + " " + a["abstract"]).str.strip()
    d = pd.DataFrame({"label": a["label"].astype(int).values, "tiab": a["tiab"].values,
                      "g2p_lgmde": a["g2p_id"].values})
    return Dataset.from_pandas(d, preserve_index=False).cast(features), set(a["pmid"].astype(str))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="dir with ds_bert_train, ds_test")
    ap.add_argument("--external_csv", required=True, help="external_positives.csv (pmid,tiab,g2p_id,source,fold,split)")
    ap.add_argument("--aug_csv", required=True, help="250-annotation augmentation set (base = augmented screen)")
    ap.add_argument("--random_csv", required=True, help="random PubMed sample (deployment FPR)")
    ap.add_argument("--out_csv", default="revision/external_recall/external_recall_results.csv")
    ap.add_argument("--dump_scores", default="revision/external_recall/external_scores")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    from datasets import concatenate_datasets, load_from_disk
    ds_train = load_from_disk(f"{args.data_dir}/ds_bert_train")
    ds_test = load_from_disk(f"{args.data_dir}/ds_test")

    aug, aug_pmids = aug_ds(args.aug_csv, ds_train.features)
    base = concatenate_datasets([ds_train, aug])  # confirmed-best BioClinical + augmentation screen
    ext = pd.read_csv(args.external_csv, dtype=str).drop_duplicates("pmid")
    ext = ext[~ext["pmid"].astype(str).isin(aug_pmids)]  # don't double-count the 250 augmentation papers
    train_add = ext[ext["split"] == "train"]
    heldout = ext[ext["split"] == "heldout"].reset_index(drop=True)
    rnd = pd.read_csv(args.random_csv, dtype=str).fillna("")
    print(f"base = ds_bert_train {ds_train.num_rows} + augmentation {aug.num_rows} = {base.num_rows} "
          f"(pos {sum(base['label'])}) | external train-add {len(train_add)} | "
          f"held-out eval {len(heldout)} ({heldout['source'].value_counts().to_dict()}) | random {len(rnd)}")
    if args.dry_run:
        print("dry_run OK")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ext_pos = as_positive_ds(train_add, ds_train.features)
    runs = [("augmented", base), ("augmented_plus_external", concatenate_datasets([base, ext_pos]))]
    os.makedirs(args.dump_scores, exist_ok=True)
    rows = []
    for label, tr_ds in runs:
        print(f"\n=== variant={label}: train {tr_ds.num_rows} ===", flush=True)
        model, f1 = train_and_eval(tr_ds, ds_test, tokenizer, f"./_ext_{label}")
        ho_p = score_proba(model, tokenizer, heldout["tiab"])
        rnd_p = score_proba(model, tokenizer, rnd["tiab"])
        row = {"variant": label, **f1, "random_fpr_pct": round(100 * (rnd_p >= 0.5).mean(), 2)}
        for s in ["premined", "hpoa", "clingen"]:
            mask = (heldout["source"] == s).values
            row[f"heldout_recall_{s}_pct"] = round(100 * (ho_p[mask] >= 0.5).mean(), 1) if mask.any() else None
        row["heldout_recall_all_pct"] = round(100 * (ho_p >= 0.5).mean(), 1)
        rows.append(row)
        print(row)
        pd.DataFrame({"source": heldout["source"].values, "proba": ho_p}).to_csv(
            f"{args.dump_scores}/{label}_heldout.csv", index=False)
        pd.DataFrame({"proba": rnd_p}).to_csv(f"{args.dump_scores}/{label}_random.csv", index=False)
        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== EXTERNAL-RECALL RESULTS (held-out-fold, EvAgg axis; EvAgg=99.5%) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
