#!/usr/bin/env python3
"""Query the hosted AMELIE API for the external truth gene-diseases (Reviewer 3 R3.3).

AMELIE (amelie.stanford.edu) is a patient-diagnosis tool: given a patient's HPO phenotypes and a
candidate gene, its knowledgebase returns the TOP-5 supporting articles for that gene, ranked by
phenotype match (logistic-regression classifiers over PubMed). It is gene-level + phenotype-
conditioned and does NOT assign diseases (like EvAgg). The top-5 cap is a design choice (diagnosis,
not comprehensive mining), so we report gene-disease COVERAGE (does AMELIE surface any supporting
paper for the gene+phenotypes) alongside a caveated top-5 truth-overlap.

For each external truth disease (g2p_id -> gene + that OMIM disease's HPO terms + curated truth
PMIDs), POST to /api/gene_list_api/ and record the returned PMIDs+scores. Checkpoints incrementally
so it resumes after interruptions/rate-limits. No auth; verify=False per AMELIE's own API example.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
URL = "https://amelie.stanford.edu/api/gene_list_api/"


def query(gene: str, hpos: str, retries: int = 4) -> list | None:
    for attempt in range(retries):
        try:
            r = requests.post(URL, verify=False, timeout=90,
                              data={"patientName": "ext", "phenotypes": hpos, "genes": gene})
            if r.status_code == 200:
                return json.loads(r.text)
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query_table", default="revision/external_recall/amelie_query_table.csv")
    ap.add_argument("--out", default="revision/external_recall/amelie_results.csv")
    ap.add_argument("--limit", type=int, default=None, help="only first N diseases (testing)")
    ap.add_argument("--sleep", type=float, default=0.2)
    return ap.parse_args()


def main():
    args = parse_args()
    q = pd.read_csv(args.query_table, dtype=str).fillna("")
    q = q[(q["gene"] != "") & (q["n_hpo"].astype(int) > 0)]
    if args.limit:
        q = q.head(args.limit)
    done = set()
    out = Path(args.out)
    if out.exists():
        done = set(pd.read_csv(out, dtype=str)["g2p_id"])
        print(f"resuming: {len(done)} already done")
    rows = []
    for i, r in enumerate(q.itertuples(), 1):
        if r.g2p_id in done:
            continue
        res = query(r.gene, r.hpo)
        returned, top = [], 0.0
        if res:
            for g, arts in res:
                if g.upper() == r.gene.upper():
                    returned = [str(p) for p, s in arts]
                    top = max([s for _, s in arts], default=0.0)
        truth = set(str(r.truth_pmids).split(","))
        rows.append({"g2p_id": r.g2p_id, "gene": r.gene, "n_hpo": r.n_hpo,
                     "amelie_n": len(returned), "amelie_top_score": round(float(top), 1),
                     "covered": int(len(returned) > 0),
                     "truth_in_top5": int(bool(truth & set(returned))),
                     "amelie_pmids": ";".join(returned)})
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(out, mode="a" if out.exists() else "w",
                                      header=not out.exists(), index=False)
            print(f"  {i}/{len(q)} | covered {sum(x['covered'] for x in rows)}/{len(rows)}", flush=True)
            rows = []
        time.sleep(args.sleep)
    if rows:
        pd.DataFrame(rows).to_csv(out, mode="a" if out.exists() else "w", header=not out.exists(), index=False)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
