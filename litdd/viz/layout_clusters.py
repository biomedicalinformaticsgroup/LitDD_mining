#!/usr/bin/env python3
"""Cluster paper embeddings and lay them out in two dimensions for the datamap figure.

Reads the parquet written by ``embed_papers.py`` (``--in_parquet``): an ``embedding`` column
(one vector per paper), ``tiab`` and ``g2p_ids`` (the G2P entries assigned to the paper).

Computes a UMAP projection to ``--umap_dim_clust`` dimensions, HDBSCAN clusters on that
projection, a second UMAP projection to two dimensions for display, and a label per cluster
made of its most frequent G2P code prefixes and its c-TF-IDF keywords over ``tiab``. cuML is
used for UMAP and HDBSCAN when it imports; otherwise ``umap-learn`` and ``hdbscan`` on the CPU.

Writes ``--out_parquet``: the input columns plus ``cluster_id`` (-1 for noise), ``viz_x``,
``viz_y``, ``codes_norm`` and ``cluster_label``. ``datamap_plot.py`` labels the clusters with MONDO terms and renders the result.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd


# ----------------------- Utilities --------------------------
def to_numpy(x):
    """Convert a CuPy array to NumPy; return anything else unchanged."""
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return x


def flatten_codes(val) -> list[str]:
    """Flatten a ``g2p_ids`` cell (string, list, set, tuple or array, nested) to a list of strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, str):
        return [val]
    try:
        iterator = iter(val)
    except TypeError:
        return [str(val)]
    out = []
    for item in iterator:
        if isinstance(item, np.ndarray):
            out.extend(flatten_codes(item.tolist()))
        elif isinstance(item, (list, tuple, set)):
            out.extend(flatten_codes(item))
        elif isinstance(item, str):
            out.append(item)
        else:
            out.append(str(item))
    return out


def code_id_prefix(c, prefix_len: int) -> str:
    """First ``prefix_len`` characters of a code; the whole code when ``prefix_len`` is 0 or the code is shorter."""
    if not isinstance(c, str):
        c = str(c)
    if prefix_len and len(c) >= prefix_len:
        return c[:prefix_len]
    return c


def preprocess_text(x) -> str:
    """Lower-case, keep letters, digits, whitespace and ``-+/``, collapse whitespace."""
    if not isinstance(x, str):
        return ""
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s\-+/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def top_codes_per_cluster(df: pd.DataFrame, top_k: int = 3, col: str = "codes_norm") -> dict[int, list[str]]:
    """The ``top_k`` most frequent codes of each non-noise cluster."""
    out = {}
    for cid, g in df[df.cluster_id >= 0].groupby("cluster_id"):
        codes = [c for lst in g[col] for c in lst if c]
        if not codes:
            out[cid] = []
            continue
        vc = pd.Series(codes, dtype="object").value_counts()
        out[cid] = [str(x) for x in vc.head(top_k).index.tolist()]
    return out


def ctfidf_keywords(df: pd.DataFrame, top_k: int = 4, max_feats: int = 30000, min_df: int = 5) -> dict[int, list[str]]:
    """The ``top_k`` c-TF-IDF keywords of each non-noise cluster, from the concatenated ``tiab`` text."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    mask = df["cluster_id"] >= 0
    if mask.sum() == 0:
        return {}
    docs = df.loc[mask, "tiab"].fillna("").astype(str).map(preprocess_text)
    cids = df.loc[mask, "cluster_id"].values
    groups = pd.Series(docs.values).groupby(cids).apply(lambda x: " ".join(x))
    cluster_texts = groups.sort_index()

    vectorizer = CountVectorizer(stop_words="english", max_features=max_feats, min_df=min_df)
    Xc = vectorizer.fit_transform(cluster_texts.values)
    if Xc.shape[1] == 0:
        return {}

    tf = normalize(Xc, norm="l1", axis=1)
    df_term = (Xc > 0).sum(axis=0).A1
    n_c = Xc.shape[0]
    idf = np.log((n_c + 1) / (df_term + 1)) + 1.0
    ctfidf = tf.multiply(idf)

    vocab = np.array(vectorizer.get_feature_names_out())
    keywords = {}
    for row_idx, cid in enumerate(cluster_texts.index):
        row = ctfidf.getrow(row_idx)
        if row.nnz == 0:
            keywords[cid] = []
            continue
        top = np.argsort(row.data)[::-1][:top_k]
        terms = vocab[row.indices[top]]
        keywords[cid] = [str(t) for t in terms.tolist()]
    return keywords


def compose_label(cid: int, code_labels: dict, kw_labels: dict) -> str:
    """``"code, code | kw / kw / kw"`` for a cluster, or ``"cluster <id>"`` when both are empty."""
    codes = [str(c) for c in code_labels.get(cid, []) if c]
    kws = [str(k) for k in kw_labels.get(cid, []) if k]
    parts = []
    if codes:
        parts.append(", ".join(codes))
    if kws:
        parts.append(" / ".join(kws[:3]))
    return " | ".join(parts) if parts else f"cluster {cid}"


# ----------------------- Projection and clustering --------------------------
def umap_project(X: np.ndarray, n_neighbors: int, min_dist: float, n_components: int,
                 deterministic: bool, verbose: bool) -> tuple[np.ndarray, str]:
    """UMAP with cosine metric; cuML when available, otherwise ``umap-learn``. Returns (coords, backend)."""
    try:
        from cuml.manifold import UMAP as cuUMAP
        kwargs = dict(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                      metric="cosine", verbose=verbose)
        if deterministic:
            kwargs["random_state"] = 42  # cuML then uses brute-force KNN, which is slower
        return to_numpy(cuUMAP(**kwargs).fit_transform(X)), "cuML-UMAP"
    except Exception as e_gpu:
        print("cuML UMAP unavailable, falling back to CPU UMAP:", e_gpu)
        import umap.umap_ as umap_cpu
        reducer = umap_cpu.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                                metric="cosine", random_state=(42 if deterministic else None), verbose=verbose)
        return reducer.fit_transform(X), "umap-learn (CPU)"


def hdbscan_cluster(X_low: np.ndarray, min_cluster_size: int, min_samples: int, eps: float,
                    selection_method: str) -> tuple[np.ndarray, str]:
    """HDBSCAN labels (-1 for noise); cuML when available, otherwise ``hdbscan``. Returns (labels, backend)."""
    try:
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        hdb = cuHDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                        cluster_selection_epsilon=float(eps), cluster_selection_method=selection_method)
        return to_numpy(hdb.fit_predict(X_low)), "cuML-HDBSCAN"
    except Exception as e_gpu:
        print("cuML HDBSCAN unavailable, falling back to CPU HDBSCAN:", e_gpu)
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                              cluster_selection_epsilon=float(eps), cluster_selection_method=selection_method,
                              metric="euclidean", core_dist_n_jobs=1)
        return hdb.fit_predict(X_low), "hdbscan (CPU)"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", required=True, help="parquet from embed_papers.py")
    ap.add_argument("--out_parquet", required=True,
                    help="output parquet with cluster_id, viz_x, viz_y and cluster_label")
    # UMAP projection used for clustering.
    ap.add_argument("--umap_neighbors_clust", type=int, default=80)
    ap.add_argument("--umap_min_dist_clust", type=float, default=0.3)
    ap.add_argument("--umap_dim_clust", type=int, default=15)
    # HDBSCAN on the clustering projection.
    ap.add_argument("--hdbscan_min_cluster_size", type=int, default=None,
                    help="default max(150, n_records // 400)")
    ap.add_argument("--hdbscan_min_samples", type=int, default=15)
    ap.add_argument("--hdbscan_eps", type=float, default=0.0, help="cluster_selection_epsilon")
    ap.add_argument("--hdbscan_selection_method", default="leaf")
    # Two-dimensional UMAP projection for display.
    ap.add_argument("--umap_neighbors_viz", type=int, default=200)
    ap.add_argument("--umap_min_dist_viz", type=float, default=0.7)
    ap.add_argument("--deterministic", action="store_true",
                    help="fix UMAP's random_state to 42 (slower with cuML)")
    ap.add_argument("--code_prefix_len", type=int, default=8,
                    help="characters of each G2P code kept for cluster labels; 0 keeps whole codes")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        df = pd.read_parquet(args.in_parquet)
    except FileNotFoundError:
        print(f"Input parquet not found: {args.in_parquet}", file=sys.stderr)
        return 1
    if "embedding" not in df.columns:
        print("Column 'embedding' not found in the input parquet.", file=sys.stderr)
        return 1

    X = np.array(df["embedding"].tolist(), dtype=np.float32)
    n = X.shape[0]
    print(f"Embeddings shape: {X.shape}")

    min_cluster_size = (args.hdbscan_min_cluster_size if args.hdbscan_min_cluster_size is not None
                        else max(150, n // 400))

    # Clustering projection and HDBSCAN.
    X_low, used_umap = umap_project(X, args.umap_neighbors_clust, args.umap_min_dist_clust,
                                    args.umap_dim_clust, args.deterministic, verbose=True)
    print("Low-D shape:", X_low.shape, "via", used_umap)
    labels, used_clusterer = hdbscan_cluster(X_low, min_cluster_size, args.hdbscan_min_samples,
                                             args.hdbscan_eps, args.hdbscan_selection_method)
    df["cluster_id"] = labels.astype(int)

    sizes = df[df.cluster_id >= 0].groupby("cluster_id").size().sort_values(ascending=False)
    n_noise = int((df.cluster_id == -1).sum())
    print(f"Clusters found: {len(sizes)}  (noise={n_noise}) via {used_clusterer}")
    print("Top 10 cluster sizes:", sizes.head(10).tolist())
    print("Median cluster size:", int(sizes.median()) if len(sizes) else 0)

    # Display projection.
    viz, viz_method = umap_project(X, args.umap_neighbors_viz, args.umap_min_dist_viz, 2,
                                   args.deterministic, verbose=False)
    df["viz_x"] = viz[:, 0].astype(np.float32)
    df["viz_y"] = viz[:, 1].astype(np.float32)
    print("2D viz via", viz_method)

    # Cluster labels from G2P code prefixes and c-TF-IDF keywords.
    if "g2p_ids" not in df.columns:
        print("Warning: 'g2p_ids' not found; code-based labels will be empty.", file=sys.stderr)
        df["codes_norm"] = [[] for _ in range(len(df))]
    else:
        df["codes_norm"] = df["g2p_ids"].apply(flatten_codes)
        if args.code_prefix_len:
            df["codes_norm"] = df["codes_norm"].apply(
                lambda lst: [code_id_prefix(x, args.code_prefix_len) for x in lst])

    code_labels = top_codes_per_cluster(df, top_k=10, col="codes_norm")
    try:
        kw_labels = ctfidf_keywords(df, top_k=10)
    except Exception as e_kw:
        print("Keyword labeling failed; proceeding with code labels only:", e_kw)
        kw_labels = {}

    cluster_label_map = {cid: compose_label(cid, code_labels, kw_labels)
                         for cid in sorted(set(df.cluster_id)) if cid >= 0}
    df["cluster_label"] = df["cluster_id"].map(cluster_label_map)

    df.to_parquet(args.out_parquet, index=False)
    print(f"Saved labeled clusters and 2D coords to {args.out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
