#!/usr/bin/env python3
"""2x2 factorial finetune of the BioClinical-ModernBERT screen (Reviewer 3 R3.4 / R1.3):
{augmentation off/on} x {gene-conditioning off/on}.

- augmentation: append the reviewer-confirmed molecular-framed positives to the training set.
- gene-conditioning: input becomes the pair [TIAB] [SEP] [gene symbol ; prev symbols ; full name],
  so the screen answers "evidence for THIS gene?" rather than the global "is this a GDD paper?".

Each cell is evaluated on (a) test F1 and (b) recovery on the gene-present BERT-negative misses,
split into held-out-gene (generalisation) vs train-fold. For the gene-conditioned cells the test
and miss inputs are paired with the candidate gene; for the non-conditioned cells the gene is
ignored. Augmentation rows are train-fold only.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib

import numpy as np
import pandas as pd

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, lr=1.736e-5, wd=0.3)


def fold(gene: str) -> str:
    return "heldout" if int(hashlib.md5(gene.encode()).hexdigest(), 16) % 5 == 0 else "train"


def gene_fullnames(path: str) -> dict[str, str]:
    out = {}
    with gzip.open(path, "rt") as fh:
        h = {c: i for i, c in enumerate(fh.readline().lstrip("#").rstrip().split("\t"))}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if p[h["description"]] not in ("", "-"):
                out[p[h["Symbol"]]] = p[h["description"]]
    return out


def cond_of(gene: str, prev: str, names: dict[str, str]) -> str:
    forms = [gene] + [x.strip() for x in prev.replace(";", ",").split(",") if x.strip()]
    if names.get(gene):
        forms.append(names[gene])
    return " ; ".join(dict.fromkeys(f for f in forms if f))


def metrics_fn():
    import evaluate
    acc, pr, rc, f1 = (evaluate.load(x) for x in ["accuracy", "precision", "recall", "f1"])

    def cm(ep):
        logits, labels = ep
        p = np.argmax(logits, axis=-1)
        return {"precision": pr.compute(predictions=p, references=labels, zero_division=0)["precision"],
                "recall": rc.compute(predictions=p, references=labels, zero_division=0)["recall"],
                "f1": f1.compute(predictions=p, references=labels)["f1"]}
    return cm


def train_eval(train_df, test_df, tokenizer, pair, out_dir):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    def tok(b):
        return tokenizer(b["tiab"], b["cond"], truncation=True, max_length=512) if pair \
            else tokenizer(b["tiab"], truncation=True, max_length=512)
    tt = Dataset.from_pandas(train_df[["tiab", "cond", "label"]].reset_index(drop=True)).map(tok, batched=True)
    te = Dataset.from_pandas(test_df[["tiab", "cond", "label"]].reset_index(drop=True)).map(tok, batched=True)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=HP["epochs"],
        per_device_train_batch_size=HP["bs"], per_device_eval_batch_size=HP["bs"], learning_rate=HP["lr"],
        weight_decay=HP["wd"], logging_steps=200, report_to="none", save_strategy="no", bf16=torch.cuda.is_available())
    tr = Trainer(model=model, args=args, train_dataset=tt, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8), compute_metrics=metrics_fn())
    tr.train()
    m = tr.evaluate(te)
    return model, {k: round(float(m[f"eval_{k}"]), 4) for k in ("precision", "recall", "f1")}


def predict_pos(model, tokenizer, df, pair):
    import torch
    model.eval()
    preds = []
    for i in range(0, len(df), 64):
        b = df.iloc[i:i + 64]
        args = (list(b["tiab"]), list(b["cond"])) if pair else (list(b["tiab"]),)
        enc = tokenizer(*args, truncation=True, max_length=512, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            preds += model(**enc).logits.argmax(-1).cpu().tolist()
    return np.array(preds)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--aug_csv", required=True)
    ap.add_argument("--misses_csv", required=True)
    ap.add_argument("--ddg2p", required=True)
    ap.add_argument("--gene_info", required=True)
    ap.add_argument("--out_csv", default="revision/external_recall/screen_2x2_results.csv")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    from datasets import load_from_disk
    dd = pd.read_csv(args.ddg2p, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    g2gene = dict(zip(dd["g2p id"], dd["gene symbol"].str.strip()))
    g2prev = dict(zip(dd["g2p id"], dd["previous gene symbols"]))
    names = gene_fullnames(args.gene_info)

    def lgmde_cond(s):  # cond from a g2p_lgmde thread (gene=field1, prev=field4)
        p = s.split(" - ")
        return cond_of(p[1].strip() if len(p) > 1 else "", p[4] if len(p) > 4 else "", names)

    def build(ds):
        df = ds.to_pandas()
        df["cond"] = df["g2p_lgmde"].map(lgmde_cond)
        df["label"] = df["label"].astype(int)
        return df[["tiab", "cond", "label"]]
    train_base = build(load_from_disk(f"{args.data_dir}/ds_bert_train"))
    test_df = build(load_from_disk(f"{args.data_dir}/ds_test"))

    a = pd.read_csv(args.aug_csv, dtype=str).fillna("")
    cp = a["confirm_positive"].str.strip().str.lower()
    a = a[cp.isin(["0", "1", "yes", "no", "true", "false", "y", "n"])].copy()
    a["label"] = cp.loc[a.index].map(lambda v: 1 if v in ("1", "yes", "true", "y") else 0)
    a["tiab"] = (a["title"] + " " + a["abstract"]).str.strip()
    a["cond"] = [cond_of(g, g2prev.get(gid, ""), names) for g, gid in zip(a["gene"], a["g2p_id"])]
    aug = a[["tiab", "cond", "label"]]

    m = pd.read_csv(args.misses_csv, dtype=str).fillna("").drop_duplicates("pmid")
    m["tiab"] = (m["title"] + " " + m["abstract"]).str.strip()
    m["gene"] = m["g2p"].map(lambda g: g2gene.get(g, ""))
    m["cond"] = [cond_of(g2gene.get(g, ""), g2prev.get(g, ""), names) for g in m["g2p"]]
    m["foldg"] = m["gene"].map(fold)

    print(f"train_base {len(train_base)} | aug {len(aug)} (pos {int(aug.label.sum())}) | test {len(test_df)} | "
          f"misses {len(m)} (heldout {int((m.foldg=='heldout').sum())})")
    if args.dry_run:
        print("dry_run OK")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for gcond in (False, True):
        for use_aug in (False, True):
            tr_df = pd.concat([train_base, aug], ignore_index=True) if use_aug else train_base
            name = f"{'genecond' if gcond else 'plain'}{'+aug' if use_aug else ''}"
            print(f"\n=== {name}: train {len(tr_df)} | gene_cond={gcond} aug={use_aug} ===", flush=True)
            model, f1 = train_eval(tr_df, test_df, tokenizer, gcond, f"./_2x2_{name}")
            m["pred"] = predict_pos(model, tokenizer, m, gcond)
            rows.append({"cell": name, "gene_conditioned": gcond, "augmented": use_aug, **f1,
                         "miss_recovery_pct": round(100 * m["pred"].mean(), 1),
                         "miss_recovery_heldout_pct": round(100 * m.loc[m.foldg == "heldout", "pred"].mean(), 1),
                         "miss_recovery_trainfold_pct": round(100 * m.loc[m.foldg == "train", "pred"].mean(), 1)})
            print(rows[-1])
            import torch
            del model
            gc.collect()
            torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== 2x2 RESULTS ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
