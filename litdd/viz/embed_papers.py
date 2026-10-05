"""Embed every paper of the released map with a text encoder, for the datamap figure.

Reads the released map CSV (``PMID``, ``G2P_IDs``) and a parquet holding each paper's title
and abstract (``pmid``, ``tiab``; the adjudication output serves), joins them, and encodes each
text with a transformer encoder (``--model``): the last hidden state is pooled (``--pooling``
``cls`` or ``mean`` over the attention mask) and L2-normalised. The figure uses
``abhinand/MedEmbed-large-v0.1`` with CLS pooling. Writes ``--out_parquet`` with
one row per paper: ``pmid``, ``tiab``, ``g2p_ids`` (list of the paper's entries) and
``embedding`` (list of floats). ``layout_clusters.py`` consumes it.

    python -m litdd.viz.embed_papers --map_csv results/litdd_pubmed2026_final_v6.csv \\
        --text_parquet llm_all.parquet --out_parquet embeddings.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "abhinand/MedEmbed-large-v0.1"


def load_papers(map_csv: str, text_parquet: str) -> pd.DataFrame:
    """One row per paper of the map with its G2P ids and its title and abstract."""
    m = pd.read_csv(map_csv, dtype=str)
    m["pmid"] = m["PMID"].astype(str).str.strip()
    codes = (m.groupby("pmid")["G2P_IDs"].apply(lambda s: sorted(set(s.astype(str).str.strip())))
             .rename("g2p_ids"))
    t = pd.read_parquet(text_parquet, columns=["pmid", "tiab"])
    t["pmid"] = t["pmid"].astype(str).str.strip()
    t = t.drop_duplicates("pmid").set_index("pmid")
    df = codes.to_frame().join(t, how="left").reset_index()
    missing = df["tiab"].isna() | (df["tiab"].fillna("").str.strip() == "")
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} papers of the map have no text in {text_parquet}")
    return df


def embed_texts(texts: list[str], model_id: str, batch_size: int, max_length: int,
                device: str | None = None, pooling: str = "cls") -> np.ndarray:
    """L2-normalised encoder embeddings of ``texts``, pooled by ``cls`` token or ``mean``."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out[idx] = pooled.float().cpu().numpy()
            if (start // batch_size) % 50 == 0:
                logger.info("embedded %d / %d texts (%.0f s)", min(start + batch_size, len(texts)),
                            len(texts), time.time() - t0)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map_csv", required=True, help="released map CSV (PMID, G2P_IDs)")
    ap.add_argument("--text_parquet", required=True, help="parquet with pmid and tiab")
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="encoder id or path")
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = parse_args()
    df = load_papers(args.map_csv, args.text_parquet)
    logger.info("%d papers, %d entries", len(df), len({g for gs in df["g2p_ids"] for g in gs}))
    X = embed_texts(df["tiab"].tolist(), args.model, args.batch_size, args.max_length, args.device,
                    args.pooling)
    df["embedding"] = list(X)
    df.to_parquet(args.out_parquet, index=False)
    logger.info("wrote %s (%d x %d)", args.out_parquet, *X.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
