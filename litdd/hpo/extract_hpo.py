#!/usr/bin/env python3
"""Annotate mapped full text with HPO terms via FastHPOCR, then roll up HPO terms per
G2P disease entry.

Upstream: full text is retrieved with Cadmus (run_cadmus.py ->
https://github.com/biomedicalinformaticsgroup/cadmus) and assembled into a parquet by
get_fulltext_df.py. The full text itself is not shared here (publisher permissions).

Two HPO representations are produced and carried through to the per-disease rollup:
  - ``weighted_hpo_ids``  : ';'-separated HPO ids with each term repeated according to
    its within-document frequency, then concatenated across all papers mapped to a G2P
    id. This frequency-weighted phenotype profile is what feeds the Exomiser experiment.
  - ``unweighted_hpo_ids``: the de-duplicated set of HPO ids.
(`hpo_ids` is kept as a backward-compatible alias of `weighted_hpo_ids`.)

Inputs (overridable via CLI):
  - the fulltext parquet (content_text + llm_dis_map columns) from get_fulltext_df.py
  - a FastHPOCR index (`hp.index`); build it once from `hp.obo` with --build_index
  - the G2P DD CSV (for the g2p_id -> disease MIM / MONDO join)

The large ontology/index/data files are not shipped in the repo; see README.
"""
import argparse

import pandas as pd
from FastHPOCR.HPOAnnotator import HPOAnnotator


def build_index(hp_obo: str, out_dir: str = ".") -> None:
    """Build the FastHPOCR index from an hp.obo file (run once)."""
    from FastHPOCR.IndexHPO import IndexHPO

    IndexHPO(hp_obo, out_dir).index()


def annotate_records(fulltext_df: pd.DataFrame, hp_index: str) -> pd.DataFrame:
    """Add HPO annotation columns from `content_text`.

    `weighted_hpo_ids` preserves duplicates (within-doc frequency);
    `unweighted_hpo_ids` is the sorted unique set.
    """
    annotator = HPOAnnotator(hp_index)

    all_hpo = []
    for txt in fulltext_df["content_text"]:
        if not isinstance(txt, str) or not txt.strip():
            all_hpo.append([])
            continue
        annotation_list = []
        for ao in annotator.annotate(txt) or []:
            parts = ao.toString().split("\t")
            if len(parts) >= 4:
                annotation_list.append(
                    {"endOffset": parts[0], "hpoUri": parts[1],
                     "hpoLabel": parts[2], "textSpan": parts[3]}
                )
        all_hpo.append(annotation_list)

    fulltext_df = fulltext_df.copy()
    fulltext_df["hpo_dicts"] = all_hpo
    # weighted: keep every occurrence (within-doc frequency)
    fulltext_df["weighted_hpo_ids"] = fulltext_df["hpo_dicts"].apply(
        lambda x: ";".join(d["hpoUri"] for d in x)
    )
    # unweighted: de-duplicated set
    fulltext_df["unweighted_hpo_ids"] = fulltext_df["hpo_dicts"].apply(
        lambda x: ";".join(sorted(set(d["hpoUri"] for d in x)))
    )
    # backward-compatible alias
    fulltext_df["hpo_ids"] = fulltext_df["weighted_hpo_ids"]
    return fulltext_df


def _agg_join(series, dedup: bool) -> str:
    """Concatenate ';'-joined HPO strings across rows; optionally de-duplicate."""
    parts = [p for s in series for p in str(s).split(";") if p]
    if dedup:
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        parts = out
    return ";".join(parts)


def g2p_to_hpo_df(ft_df: pd.DataFrame, g2p_col="llm_dis_map") -> pd.DataFrame:
    """Explode multi-G2P rows and aggregate weighted + unweighted HPO ids per G2P id.

    The weighted aggregation preserves duplicates across papers, so a term's count in
    the per-disease profile reflects how often it appears across the disease's literature.
    """
    df = ft_df[[g2p_col, "weighted_hpo_ids", "unweighted_hpo_ids"]].copy()
    df["_src_row"] = df.index  # track source row to count unique papers
    df["_g2p_list"] = (
        df[g2p_col].fillna("").astype(str)
        .apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
    )
    df["_weighted"] = df["weighted_hpo_ids"].fillna("").astype(str).str.strip(" ;")
    df["_unweighted"] = df["unweighted_hpo_ids"].fillna("").astype(str).str.strip(" ;")

    exploded = df.explode("_g2p_list", ignore_index=True)
    exploded = exploded[exploded["_g2p_list"].notna() & (exploded["_g2p_list"] != "")]

    out = (
        exploded.groupby("_g2p_list", sort=False)
        .agg(
            weighted_hpo_ids=("_weighted", lambda s: _agg_join(s, dedup=False)),
            unweighted_hpo_ids=("_unweighted", lambda s: _agg_join(s, dedup=True)),
            n_input_papers=("_src_row", "nunique"),
        )
        .reset_index()
        .rename(columns={"_g2p_list": "g2p_id"})
    )
    out["weighted_hpo_ids"] = out["weighted_hpo_ids"].str.strip(";")
    out["unweighted_hpo_ids"] = out["unweighted_hpo_ids"].str.strip(";")
    out["hpo_ids"] = out["weighted_hpo_ids"]  # backward-compatible alias
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fulltext_parquet", default="litdd/hpo/final_mappings_fulltext.parquet",
                    help="Parquet with content_text + llm_dis_map columns (from get_fulltext_df.py)")
    ap.add_argument("--hp_index", default="litdd/hpo/hp.index", help="FastHPOCR index file")
    ap.add_argument("--g2p_file", required=True, help="G2P DD CSV (for disease MIM/MONDO join)")
    ap.add_argument("--out_hpo_parquet", default="final_mappings_hpo.parquet")
    ap.add_argument("--out_g2p_csv", default="g2p_mapped_hpo.csv")
    ap.add_argument("--build_index", metavar="HP_OBO", default=None,
                    help="Build the FastHPOCR index from this hp.obo, then exit")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.build_index:
        build_index(args.build_index)
        return

    fulltext_df = pd.read_parquet(args.fulltext_parquet)
    fulltext_df = (
        fulltext_df[fulltext_df["content_text"].notna()]
        .drop_duplicates(subset=["llm_dis_map", "tiab"])
    )

    fulltext_df = annotate_records(fulltext_df, args.hp_index)
    fulltext_df.to_parquet(args.out_hpo_parquet)

    result = g2p_to_hpo_df(fulltext_df)

    g2p = pd.read_csv(args.g2p_file)[["g2p id", "disease mim", "disease MONDO"]]
    result = result.merge(g2p, left_on="g2p_id", right_on="g2p id", how="left").drop(columns="g2p id")
    result.to_csv(args.out_g2p_csv, index=False)
    print(f"Saved {args.out_hpo_parquet} and {args.out_g2p_csv} ({len(result)} G2P ids)")


if __name__ == "__main__":
    main()
