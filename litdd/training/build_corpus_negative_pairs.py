#!/usr/bin/env python3
"""Add corpus-representative negative PAIRS to the cross-encoder training set, and build
the frozen-benchmark pair sets — the cross-encoder analogue of the screen's "add20k" recipe.

Why
---
The screen's in-domain test set could not see that it fired on 19.5% of PubMed; adding
20,000 random corpus abstracts as negatives fixed it, and corpus fire rate became a release
gate. The cross-encoder has the same blind spot in pair form: every training negative is a
(relevant paper, wrong thread) pair from gene-disease literature, so an off-topic abstract
that happens to mention a panel gene — exactly what the screen's false positives look like —
is never seen in training.

What the cross-encoder actually sees in deployment is narrower than the screen: only
abstracts that pass the gene gate, each paired with the G2P entries of its mentioned genes
(``litdd/pipeline/gene_candidates.py``). So the deployment-faithful negatives are the
**gate-passing subset of the SAME 20,000 corpus abstracts the screen trained on**, each
paired with its own gate candidates and labelled 0 (silver: unverified, ~1-2% expected label
noise — the same assumption the screen's negatives carry). Ungated abstracts never reach the
cross-encoder, so pairing them would teach nothing deployment-relevant.

Inputs are the gate output for every PMID set (``gate_input.parquet`` ->
``gate_candidates.parquet``, columns pmid/tiab/set/candidate_g2p_ids/candidate_sources) and
the arm's ``corpus.json`` thread map. Outputs (all HF ``save_to_disk``):

  <arm>/ds_cross_train_addneg   training pairs + gated corpus-negative pairs
  <arm>/ds_frozen_pos           truth pairs for the frozen 1,752 (label 1) + their gate
                                candidate universe per PMID (for data-driven top-k)
  <arm>/ds_frozen_neg_pairs     gated frozen-negative candidate pairs (label 0, silver)
  <arm>/ds_test_candidates      ds_cross_test rows + per-PMID gate candidate lists

Leakage assertions: corpus-negative PMIDs/TIABs are disjoint from ds_cross_test, the frozen
positives and the frozen negatives (the screen's provenance already guarantees this; it is
re-checked here on the datasets actually written).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm_dir", default="revision/crossencoder/flat")
    p.add_argument("--gate_input_parquet", default="revision/crossencoder/gate_input.parquet",
                   help="Every (pmid, tiab, set) row that was offered to the gate.")
    p.add_argument("--gate_parquet", default="revision/crossencoder/gate_candidates.parquet",
                   help="gene_candidates.py output (gated rows only).")
    p.add_argument("--truthsets_csv", default="revision/external_recall/truthsets.csv")
    p.add_argument("--frozen_pos_csv", default="revision/external_recall/frozen_positives_1752.csv")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_from_disk

    with open(os.path.join(args.arm_dir, "corpus.json")) as f:
        thread_map = {k: v for k, v in json.load(f).items() if not k.startswith("__")}

    # gene_candidates.py drops rows with no gene, so left-join its output back onto the
    # full input: ungated rows must exist with an empty universe (they are the
    # denominator of every pass rate, and ungated frozen positives are still truth).
    gated = pl.read_parquet(args.gate_parquet).to_pandas()[["pmid", "set", "candidate_g2p_ids"]]
    gate = pl.read_parquet(args.gate_input_parquet).to_pandas().merge(
        gated, on=["pmid", "set"], how="left")
    gate["candidate_g2p_ids"] = gate["candidate_g2p_ids"].map(
        lambda x: [] if x is None or isinstance(x, float) else list(x))
    gate["n_cand"] = gate["candidate_g2p_ids"].map(len)
    print("gate pass rate by set:")
    print(gate.assign(passed=gate.n_cand > 0).groupby("set")["passed"].agg(["mean", "sum", "size"])
          .round(4).to_string())

    def pairs_for(df: pd.DataFrame, label: int) -> pd.DataFrame:
        rows = []
        for pmid, tiab, ids in zip(df["pmid"], df["tiab"], df["candidate_g2p_ids"]):
            for gid in ids:
                if gid in thread_map:
                    rows.append((tiab, thread_map[gid], label, pmid, gid))
        return pd.DataFrame(rows, columns=["tiab", "g2p_lgmde", "label", "pmid", "g2p_id"])

    # --- training: base pairs + gated corpus-negative pairs ---------------------------
    train = load_from_disk(os.path.join(args.arm_dir, "ds_cross_train")).to_pandas()
    test = load_from_disk(os.path.join(args.arm_dir, "ds_cross_test")).to_pandas()
    cneg = gate[(gate.set == "corpusneg20k") & (gate.n_cand > 0)]
    cneg_pairs = pairs_for(cneg, 0)
    fpos_df = gate[gate.set == "pos1752"]
    fneg_df = gate[gate.set == "frozenneg"]
    assert not (set(cneg.tiab) & set(test.tiab)), "corpus negatives overlap ds_cross_test"
    assert not (set(cneg.pmid) & set(fpos_df.pmid)), "corpus negatives overlap frozen positives"
    assert not (set(cneg.pmid) & set(fneg_df.pmid)), "corpus negatives overlap frozen negatives"
    assert not (set(cneg.tiab) & set(train.tiab)), "corpus negatives already in train"
    print(f"\ncorpus negatives: {len(cneg)} gated abstracts of 20,000 -> {len(cneg_pairs)} "
          f"negative pairs (mean {len(cneg_pairs) / max(len(cneg), 1):.2f} candidates each)")
    train_add = pd.concat([train[["tiab", "g2p_lgmde", "label"]],
                           cneg_pairs[["tiab", "g2p_lgmde", "label"]]], ignore_index=True)
    print(f"ds_cross_train_addneg: {len(train_add)} rows, {int((train_add.label == 1).sum())} pos "
          f"({100 * (train_add.label == 1).mean():.1f}%) vs base {len(train)} rows "
          f"{100 * (train.label == 1).mean():.1f}% pos")

    # --- frozen positives: truth pairs + candidate universe -------------------------
    truth = pd.read_csv(args.truthsets_csv, dtype=str)
    fpos_meta = pd.read_csv(args.frozen_pos_csv, dtype=str)
    truth = truth[truth.pmid.isin(fpos_meta.pmid) & (truth.match_type == "g2p_id")]
    truth = truth.drop_duplicates(["pmid", "key"]).rename(columns={"key": "g2p_id"})
    truth = truth[truth.g2p_id.isin(thread_map)]
    cand_by_pmid = dict(zip(fpos_df.pmid, fpos_df.candidate_g2p_ids))
    tiab_by_pmid = dict(zip(fpos_df.pmid, fpos_df.tiab))
    scope_by_pmid = dict(zip(fpos_meta.pmid, fpos_meta.scope_category))
    frozen_pos = pd.DataFrame({
        "tiab": truth.pmid.map(tiab_by_pmid),
        "g2p_lgmde": truth.g2p_id.map(thread_map),
        "label": 1,
        "pmid": truth.pmid.values,
        "g2p_id": truth.g2p_id.values,
        "scope_category": truth.pmid.map(scope_by_pmid),
        "candidate_g2p_ids": truth.pmid.map(lambda p: list(cand_by_pmid.get(p, []))),
    })
    frozen_pos["truth_in_candidates"] = [g in c for g, c in
                                         zip(frozen_pos.g2p_id, frozen_pos.candidate_g2p_ids)]
    print(f"\nfrozen positives: {len(frozen_pos)} truth pairs over {frozen_pos.pmid.nunique()} "
          f"PMIDs; truth entry inside gate candidates: "
          f"{100 * frozen_pos.truth_in_candidates.mean():.1f}% of pairs")
    print(frozen_pos.groupby("scope_category").truth_in_candidates.mean().round(3).to_string())

    # --- frozen negatives: gated candidate pairs (silver label 0) --------------------
    fneg_pairs = pairs_for(fneg_df[fneg_df.n_cand > 0], 0)
    print(f"\nfrozen negatives: {int((fneg_df.n_cand > 0).sum())} gated of {len(fneg_df)} -> "
          f"{len(fneg_pairs)} candidate pairs")

    # --- ds_cross_test candidate lists (data-driven top-k universe) ------------------
    tcand = gate[gate.set == "dstest"]
    test_c = test.merge(tcand[["tiab", "pmid", "candidate_g2p_ids"]], on="tiab", how="left")
    # rows whose TIAB was not in the gate input (should be none) get an empty universe
    test_c["candidate_g2p_ids"] = test_c.candidate_g2p_ids.map(
        lambda x: [] if x is None or isinstance(x, float) else list(x))
    print(f"\nds_test: {len(test_c)} rows; rows whose TIAB passes the gate: "
          f"{100 * (test_c.candidate_g2p_ids.map(len) > 0).mean():.1f}%")

    if args.dry_run:
        return 0

    base_feats = {"tiab": Value("string"), "g2p_lgmde": Value("string"),
                  "label": ClassLabel(num_classes=2)}
    Dataset.from_pandas(train_add, preserve_index=False).cast(Features(base_feats)).save_to_disk(
        os.path.join(args.arm_dir, "ds_cross_train_addneg"))
    pos_feats = Features({**base_feats, "pmid": Value("string"), "g2p_id": Value("string"),
                          "scope_category": Value("string"),
                          "candidate_g2p_ids": Sequence(Value("string")),
                          "truth_in_candidates": Value("bool")})
    Dataset.from_pandas(frozen_pos, preserve_index=False).cast(pos_feats).save_to_disk(
        os.path.join(args.arm_dir, "ds_frozen_pos"))
    pair_feats = Features({**base_feats, "pmid": Value("string"), "g2p_id": Value("string")})
    Dataset.from_pandas(fneg_pairs, preserve_index=False).cast(pair_feats).save_to_disk(
        os.path.join(args.arm_dir, "ds_frozen_neg_pairs"))
    test_feats = Features({**base_feats, "pmid": Value("string"),
                           "candidate_g2p_ids": Sequence(Value("string"))})
    Dataset.from_pandas(test_c[["tiab", "g2p_lgmde", "label", "pmid", "candidate_g2p_ids"]],
                        preserve_index=False).cast(test_feats).save_to_disk(
        os.path.join(args.arm_dir, "ds_test_candidates"))

    prov = {
        "built": pd.Timestamp.now(tz="UTC").isoformat(), "argv": sys.argv[1:],
        "recipe": "screen add20k analogue: gate-passing subset of the SAME 20,000 corpus "
                  "abstracts (revision/external_recall/ladder/add20000), each paired with its "
                  "gene-gate candidate entries, label 0 (silver)",
        "gate": "litdd/pipeline/gene_candidates.py (PubTator3 symbols UNION HGNC names) -> "
                + args.gate_parquet,
        "corpus_negatives": {"abstracts_total": 20000, "abstracts_gated": int(len(cneg)),
                             "negative_pairs": int(len(cneg_pairs))},
        "train_addneg": {"rows": int(len(train_add)), "pos": int((train_add.label == 1).sum())},
        "frozen_pos": {"pairs": int(len(frozen_pos)), "pmids": int(frozen_pos.pmid.nunique()),
                       "truth_in_candidates_pct": round(100 * float(frozen_pos.truth_in_candidates.mean()), 2)},
        "frozen_neg": {"abstracts": int(len(fneg_df)), "gated": int((fneg_df.n_cand > 0).sum()),
                       "pairs": int(len(fneg_pairs))},
        "leakage_checks": "corpus negatives disjoint from ds_cross_train/ds_cross_test TIABs "
                          "and from frozen positive/negative PMIDs (asserted)",
    }
    with open(os.path.join(args.arm_dir, "addneg_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"\n[Info] saved -> {args.arm_dir}/{{ds_cross_train_addneg,ds_frozen_pos,"
          f"ds_frozen_neg_pairs,ds_test_candidates,addneg_provenance.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
