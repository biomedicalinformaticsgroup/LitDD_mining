#!/usr/bin/env python3
"""Build cross-encoder (tiab, g2p_lgmde, label) pair datasets with current thread text.

Why re-render
-------------
The stored pair datasets carry the thread text as it was rendered at annotation time.
G2P entries are re-curated between exports (confidence, mechanism fields, disease names),
so the stored strings drift from what ``litdd/pipeline/crossencode.py`` scores at
inference: against the 2026-06-24 export, fewer than 1% of stored threads still match the
deployment rendering byte-for-byte. Training on stale text and deploying on current text
is a silent train/serve skew, so this script re-renders every pair's thread from its
``g2p_id`` against the *current* export before fine-tuning. Rows whose entry was retired
from the panel are dropped (and counted in the provenance record).

Thread variants (the representation is a model input, so each variant is a separate
fine-tune arm with its own hard-negative mining):

  flat      -- the 15-field deployment rendering (``litdd.threads``)
  genename  -- flat + the full gene name (NCBI gene_info) inserted after
               ``previous gene symbols``, so a TIAB saying "arginase" has text to align
               against in the ``ARG1`` thread
  context   -- multi-line contextualised threads (MONDO/HPO/OMIM-enriched), supplied as a
               prebuilt ``{g2p_id: text}`` JSON

Augmentation-row filter: the augmentation positives were annotated at *gene* level
(1 = the gene causes some DD, even if the thread's specific disease differs), which is a
noisy positive for a disease-level ranker whenever the gene has several G2P entries.
``--aug_csv`` identifies those rows (via their PMIDs in ``--annotated_csv``) and keeps
augmentation positives only for genes with exactly one panel entry.

Leakage assertions (verified on output, not assumed): train/test TIAB sets disjoint, and
PMID sets disjoint for every TIAB that resolves to a PMID via ``--annotated_csv`` /
``--external_csv``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

SEP = " - "


def parse_g2p_id(thread: str) -> str:
    return str(thread).split(SEP, 1)[0].strip()


def panel_gene_info(g2p_csv: str) -> tuple[dict[str, str], dict[str, int]]:
    """({g2p_id: gene symbol}, {g2p_id: entry count of that gene}) for the current panel.

    Keyed by g2p_id, never parsed out of the thread string — thread variants (multi-line
    contextualised threads in particular) do not share the flat field layout.
    """
    df = pd.read_csv(g2p_csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates("g2p id")
    id2gene = dict(zip(df["g2p id"].str.strip(), df["gene symbol"].str.strip()))
    per_gene = df.groupby("gene symbol")["g2p id"].nunique()
    id2count = {gid: int(per_gene.get(g, 0)) for gid, g in id2gene.items()}
    return id2gene, id2count


def load_thread_map(variant: str, g2p_csv: str, gene_info: str | None,
                    context_json: str | None) -> dict[str, str]:
    from litdd.threads import build_lgmde_map, load_gene_names

    if variant == "flat":
        return build_lgmde_map(g2p_csv)
    if variant == "genename":
        if not gene_info:
            raise SystemExit("--variant genename requires --gene_info")
        return build_lgmde_map(g2p_csv, gene_names=load_gene_names(gene_info))
    if variant == "context":
        if not context_json:
            raise SystemExit("--variant context requires --context_threads_json")
        with open(context_json) as f:
            return {str(k): str(v) for k, v in json.load(f).items()
                    if not str(k).startswith("__")}  # skip the __provenance__ record
    raise SystemExit(f"unknown variant {variant!r}")


def rerender(df: pd.DataFrame, thread_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace each row's thread with the current rendering of its g2p_id."""
    df = df.copy()
    df["g2p_id"] = df["g2p_lgmde"].map(parse_g2p_id)
    resolved = df["g2p_id"].isin(thread_map)
    dropped = df.loc[~resolved].copy()
    df = df.loc[resolved].copy()
    df["g2p_lgmde"] = df["g2p_id"].map(thread_map)
    return df, dropped


def filter_aug_rows(df: pd.DataFrame, aug_keys: set[tuple[str, str]],
                    entry_counts: dict[str, int],
                    external_keys: set[tuple[str, str]] | None = None,
                    ) -> tuple[pd.DataFrame, int, int]:
    """Drop gene-level-annotated aug POSITIVES whose gene has >1 panel entry.

    Keyed on (tiab, g2p_id), not tiab alone: an augmentation paper can also appear as an
    external-truth row for its curated disease, and those rows are disease-level verified —
    only the pair the augmentation worksheet itself asserted is gene-level. A pair that
    external truth independently asserts (``external_keys``) is disease-level verified even
    if the aug worksheet also named it, so it is exempt from the drop.
    """
    ids = df["g2p_id"] if "g2p_id" in df.columns else df["g2p_lgmde"].map(parse_g2p_id)
    key = list(zip(df["tiab"], ids))
    is_aug_pos = pd.Series([k in aug_keys for k in key], index=df.index) & (df["label"] == 1)
    multi_entry = ids.map(lambda gid: entry_counts.get(gid, 0) != 1)
    drop = is_aug_pos & multi_entry
    if external_keys:
        verified = pd.Series([k in external_keys for k in key], index=df.index)
        drop &= ~verified
    return df.loc[~drop].copy(), int(is_aug_pos.sum()), int(drop.sum())


def dedupe_pairs(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse exact duplicate (tiab, thread, label) rows; drop label conflicts entirely.

    Duplicates arise when a paper enters via two routes (augmentation + external truth)
    asserting the same pair; keeping both double-weights it. A (tiab, thread) carrying BOTH
    labels is a source conflict (annotated-0 vs curated-positive); training must not take a
    side, so both rows are removed and the conflicts are returned for the provenance record.
    """
    before = len(df)
    df = df.drop_duplicates(["tiab", "g2p_lgmde", "label"])
    if before - len(df):
        print(f"[Info] {name}: collapsed {before - len(df)} exact duplicate pair rows")
    conf_mask = df.duplicated(["tiab", "g2p_lgmde"], keep=False)
    conflicts = df.loc[conf_mask].copy()
    if len(conflicts):
        print(f"[WARN] {name}: dropped {conflicts['g2p_lgmde'].nunique()} (tiab, thread) "
              f"pairs carrying BOTH labels (annotated-0 vs curated-positive); recorded in "
              f"provenance for source-level review")
    return df.loc[~conf_mask].copy(), conflicts


def tiab_to_pmids(source_csvs: list[str]) -> dict[str, set[str]]:
    t2p: dict[str, set[str]] = {}
    for path in source_csvs:
        src = pd.read_csv(path, dtype=str)
        if "tiab" not in src.columns or "pmid" not in src.columns:
            continue
        for t, p in zip(src["tiab"], src["pmid"]):
            if pd.notna(t) and pd.notna(p):
                t2p.setdefault(t, set()).add(str(p))
    return t2p


def pmid_set(tiabs: set[str], t2p: dict[str, set[str]]) -> tuple[set[str], int]:
    out: set[str] = set()
    unmapped = 0
    for t in tiabs:
        ps = t2p.get(t)
        if ps:
            out |= ps
        else:
            unmapped += 1
    return out, unmapped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train_ds_dir", default="revision/external_recall/ds_hirecall_train")
    p.add_argument("--test_ds_dir", default="revision/external_recall/split_tiab/ds_test")
    p.add_argument("--g2p_csv", default="revision/G2P_DD_2026-06-24.csv")
    p.add_argument("--variant", choices=["flat", "genename", "context"], default="flat")
    p.add_argument("--gene_info", default="revision/human_gene_info.gz",
                   help="NCBI gene_info dump (gzipped TSV); used by --variant genename.")
    p.add_argument("--context_threads_json", default=None,
                   help="{g2p_id: contextualised thread} JSON; used by --variant context.")
    p.add_argument("--aug_csv",
                   default="revision/external_recall/250_augmentation_candidates_to_annotate.csv",
                   help="Augmentation-annotation worksheet; its PMIDs mark the gene-level rows.")
    p.add_argument("--annotated_csv",
                   default="revision/external_recall/annotated_tiab_augmented.csv",
                   help="pmid,tiab source used to map aug PMIDs to TIABs and to verify "
                        "PMID-level train/test disjointness.")
    p.add_argument("--external_csv",
                   default="revision/external_recall/external_positives.csv",
                   help="Second pmid,tiab source for the PMID disjointness check.")
    p.add_argument("--out_dir", default=None,
                   help="Default: revision/crossencoder/<variant>")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from datasets import ClassLabel, Dataset, Features, Value, load_from_disk

    out_dir = args.out_dir or os.path.join("revision", "crossencoder", args.variant)
    thread_map = load_thread_map(args.variant, args.g2p_csv, args.gene_info,
                                 args.context_threads_json)
    print(f"[Info] variant={args.variant}: {len(thread_map)} panel threads")

    df_train = load_from_disk(args.train_ds_dir).to_pandas()
    df_test = load_from_disk(args.test_ds_dir).to_pandas()

    df_train, drop_train = rerender(df_train, thread_map)
    df_test, drop_test = rerender(df_test, thread_map)
    print(f"[Info] retired-entry rows dropped: train {len(drop_train)} "
          f"({sorted(drop_train.get('g2p_id', pd.Series(dtype=str)).unique())}), "
          f"test {len(drop_test)}")

    # Aug rows entered the training set with tiab = title + " " + abstract (the
    # merge_screen_annotations.py rule), so rebuild that string to identify them; key on
    # (tiab, g2p_id) so external-truth rows for the same paper are not touched.
    aug = pd.read_csv(args.aug_csv, dtype=str).fillna("")
    aug = aug[aug["confirm_positive"] != ""]
    aug_keys = set(zip((aug["title"] + " " + aug["abstract"]).str.strip(),
                       aug["g2p_id"].str.strip()))
    ext = pd.read_csv(args.external_csv, dtype=str).fillna("")
    external_keys = set(zip(ext["tiab"], ext["g2p_id"].str.strip()))
    id2gene, id2count = panel_gene_info(args.g2p_csv)
    df_train, n_aug_pos, n_aug_dropped = filter_aug_rows(
        df_train, aug_keys, id2count, external_keys)
    print(f"[Info] aug pair rows in train: {n_aug_pos} positives; "
          f"dropped {n_aug_dropped} multi-entry-gene positives (external-verified exempt)")
    df_train, train_conflicts = dedupe_pairs(df_train, "train")
    df_test, test_conflicts = dedupe_pairs(df_test, "test")

    # --- leakage assertions, on the datasets actually written -------------------
    tr_tiabs, te_tiabs = set(df_train["tiab"]), set(df_test["tiab"])
    assert not (tr_tiabs & te_tiabs), "TIAB leakage: train and test share abstracts"
    t2p = tiab_to_pmids([args.annotated_csv, args.external_csv])
    tr_pmids, tr_unmapped = pmid_set(tr_tiabs, t2p)
    te_pmids, te_unmapped = pmid_set(te_tiabs, t2p)
    assert not (tr_pmids & te_pmids), \
        f"PMID leakage: {sorted(tr_pmids & te_pmids)[:10]}"
    print(f"[Info] leakage checks passed: {len(tr_tiabs)} train / {len(te_tiabs)} test "
          f"TIABs disjoint; {len(tr_pmids)} / {len(te_pmids)} mapped PMIDs disjoint "
          f"({tr_unmapped} / {te_unmapped} TIABs unmapped to a PMID)")

    keep = ["tiab", "g2p_lgmde", "label"]
    df_train, df_test = df_train[keep], df_test[keep]
    print(f"[Info] rows: train={len(df_train)} (pos {int((df_train.label == 1).sum())}) "
          f"test={len(df_test)} (pos {int((df_test.label == 1).sum())})")

    if args.dry_run:
        print("[Info] --dry_run set; not writing.")
        return 0

    features = Features({"tiab": Value("string"), "g2p_lgmde": Value("string"),
                         "label": ClassLabel(num_classes=2)})
    os.makedirs(out_dir, exist_ok=True)
    Dataset.from_pandas(df_train, preserve_index=False).cast(features).save_to_disk(
        os.path.join(out_dir, "ds_cross_train"))
    Dataset.from_pandas(df_test, preserve_index=False).cast(features).save_to_disk(
        os.path.join(out_dir, "ds_cross_test"))
    with open(os.path.join(out_dir, "corpus.json"), "w") as f:
        json.dump(thread_map, f, indent=0)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    provenance = {
        "built": pd.Timestamp.now(tz="UTC").isoformat(),
        "script": "litdd/training/build_crossencoder_dataset.py",
        "git_commit": commit,
        "argv": sys.argv[1:],
        "variant": args.variant,
        "g2p_csv": args.g2p_csv,
        "inputs": {"train_ds_dir": args.train_ds_dir, "test_ds_dir": args.test_ds_dir},
        "rows": {"train": len(df_train), "test": len(df_test),
                 "train_pos": int((df_train.label == 1).sum()),
                 "test_pos": int((df_test.label == 1).sum())},
        "dropped_retired_entries": {"train": len(drop_train), "test": len(drop_test)},
        "aug_pair_positives_in_train": n_aug_pos,
        "aug_multi_entry_positives_dropped": n_aug_dropped,
        "label_conflict_pairs_dropped": {
            "train": sorted({f"{gid} | {id2gene.get(gid, '?')}"
                             for gid in train_conflicts["g2p_id"]}),
            "test": sorted({f"{gid} | {id2gene.get(gid, '?')}"
                            for gid in test_conflicts["g2p_id"]}),
        },
        "panel_threads": len(thread_map),
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[Info] saved -> {out_dir}/{{ds_cross_train,ds_cross_test,corpus.json,provenance.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
