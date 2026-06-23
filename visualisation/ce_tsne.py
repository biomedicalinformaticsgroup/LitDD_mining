#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys

import numpy as np
import pandas as pd

# -------------------------- Config --------------------------
IN_PARQUET  = "filtered_clean_genes_bi_embeddings.parquet"   # Input with df['bi_embeddings']
OUT_PARQUET = "filtered_clean_genes_clusters_and_viz.parquet"
OUT_PNG     = "clusters_big_and_labeled.png"

# UMAP (15D) for clustering
UMAP_NEIGHBORS_CLUST = 80
UMAP_MIN_DIST_CLUST  = 0.3
UMAP_DIM_CLUST       = 15

# HDBSCAN (set min_cluster_size after we know n)
HDBSCAN_MIN_CLUSTER_SIZE = None  # if None, will compute from n
HDBSCAN_MIN_SAMPLES      = 15
HDBSCAN_EPS              = 0.0   # avoid merging
HDBSCAN_SELECTION_METHOD = "leaf"

# 2D visualization (larger neighbors/min_dist -> bigger-looking blobs)
UMAP_NEIGHBORS_VIZ = 200
UMAP_MIN_DIST_VIZ  = 0.7

# Determinism vs speed: cuML UMAP will switch to brute_force KNN if random_state is set
DETERMINISTIC = False

# Codes: shorten to first 8 chars (adjust if your IDs have a different prefix length)
SHORTEN_CODE_PREFIX = True
CODE_PREFIX_LEN = 8

# Annotate this many largest clusters on the plot
N_ANNOTATE = 64

# ----------------------- Utilities --------------------------
def to_numpy(x):
    """Convert CuPy arrays to NumPy if needed; otherwise return as-is."""
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return x

def flatten_codes(val):
    """Normalize llm_dis_map_lgmde entries to a flat list[str]. Handles sets, lists, tuples, numpy arrays, nested."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    import numpy as _np
    if isinstance(val, _np.ndarray):
        val = val.tolist()
    if isinstance(val, str):
        return [val]
    try:
        iterator = iter(val)
    except TypeError:
        return [str(val)]
    out = []
    for item in iterator:
        if isinstance(item, _np.ndarray):
            out.extend(flatten_codes(item.tolist()))
        elif isinstance(item, (list, tuple, set)):
            out.extend(flatten_codes(item))
        elif isinstance(item, str):
            out.append(item)
        else:
            out.append(str(item))
    return out

def code_id_prefix(c):
    if not isinstance(c, str):
        c = str(c)
    if SHORTEN_CODE_PREFIX and len(c) >= CODE_PREFIX_LEN:
        return c[:CODE_PREFIX_LEN]
    return c

def preprocess_text(x):
    if not isinstance(x, str):
        return ""
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s\-+/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def top_codes_per_cluster(df, top_k=3, col="codes_norm"):
    out = {}
    for cid, g in df[df.cluster_id >= 0].groupby("cluster_id"):
        codes = [c for lst in g[col] for c in lst if c]
        if not codes:
            out[cid] = []
            continue
        vc = pd.Series(codes, dtype="object").value_counts()
        out[cid] = [str(x) for x in vc.head(top_k).index.tolist()]
    return out

def ctfidf_keywords(df, top_k=4, max_feats=30000, min_df=5):
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

def compose_label(cid, code_labels, kw_labels):
    codes = [str(c) for c in code_labels.get(cid, []) if c]
    kws = [str(k) for k in kw_labels.get(cid, []) if k]
    parts = []
    if codes:
        parts.append(", ".join(codes))
    if kws:
        parts.append(" / ".join(kws[:3]))
    return " | ".join(parts) if parts else f"cluster {cid}"

# ----------------------- Load data --------------------------
if not os.path.exists(IN_PARQUET):
    print(f"Input parquet not found: {IN_PARQUET}", file=sys.stderr)
    sys.exit(1)

df = pd.read_parquet(IN_PARQUET)
if "bi_embeddings" not in df.columns:
    print("Column 'bi_embeddings' not found. Please run embedding step first.", file=sys.stderr)
    sys.exit(1)

X = np.array(df["bi_embeddings"].tolist(), dtype=np.float32)
n, d = X.shape
print(f"Embeddings shape: {X.shape}")

# Compute min_cluster_size now that we know n
min_cluster_size = (
    HDBSCAN_MIN_CLUSTER_SIZE
    if HDBSCAN_MIN_CLUSTER_SIZE is not None
    else max(150, n // 400)   # tweak this to steer the number of clusters
)

# ----------------- UMAP (15D) for clustering ----------------
X_low = None
used_umap = None
try:
    from cuml.manifold import UMAP as cuUMAP
    umap_kwargs = dict(
        n_neighbors=UMAP_NEIGHBORS_CLUST,
        min_dist=UMAP_MIN_DIST_CLUST,
        n_components=UMAP_DIM_CLUST,
        metric="cosine",
        verbose=True,
    )
    if DETERMINISTIC:
        umap_kwargs["random_state"] = 42  # Note: triggers brute_force KNN (slower) in cuML
    umap = cuUMAP(**umap_kwargs)
    X_low = umap.fit_transform(X)
    X_low = to_numpy(X_low)
    used_umap = "cuML-UMAP"
except Exception as e_gpu_umap:
    print("cuML UMAP unavailable, falling back to CPU UMAP:", e_gpu_umap)
    import umap.umap_ as umap_cpu
    umap = umap_cpu.UMAP(
        n_neighbors=UMAP_NEIGHBORS_CLUST,
        min_dist=UMAP_MIN_DIST_CLUST,
        n_components=UMAP_DIM_CLUST,
        metric="cosine",
        random_state=(42 if DETERMINISTIC else None),
        verbose=True,
    )
    X_low = umap.fit_transform(X)
    used_umap = "umap-learn (CPU)"

print("Low-D shape:", X_low.shape, "via", used_umap)

# -------------------------- HDBSCAN --------------------------
labels = None
used_clusterer = None
try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    hdb = cuHDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(HDBSCAN_MIN_SAMPLES),
        cluster_selection_epsilon=float(HDBSCAN_EPS),
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
    )
    labels = hdb.fit_predict(X_low)
    labels = to_numpy(labels)
    used_clusterer = "cuML-HDBSCAN"
except Exception as e_gpu_hdb:
    print("cuML HDBSCAN unavailable, falling back to CPU HDBSCAN:", e_gpu_hdb)
    import hdbscan
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(HDBSCAN_MIN_SAMPLES),
        cluster_selection_epsilon=float(HDBSCAN_EPS),
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
        metric="euclidean",
        core_dist_n_jobs=1,
    )
    labels = hdb.fit_predict(X_low)
    used_clusterer = "hdbscan (CPU)"

labels = labels.astype(int)
df["cluster_id"] = labels

# Report clusters
sizes = df[df.cluster_id >= 0].groupby("cluster_id").size().sort_values(ascending=False)
n_clusters = len(sizes)
n_noise = int((df.cluster_id == -1).sum())
print(f"Clusters found: {n_clusters}  (noise={n_noise})")
print("Top 10 cluster sizes:", sizes.head(10).tolist())
print("Median cluster size:", int(sizes.median()) if len(sizes) else 0)

# --------------------- 2D UMAP for viz ----------------------
viz = None
viz_method = None
try:
    from cuml.manifold import UMAP as cuUMAP
    umap2_kwargs = dict(
        n_neighbors=UMAP_NEIGHBORS_VIZ,
        min_dist=UMAP_MIN_DIST_VIZ,
        n_components=2,
        metric="cosine",
        verbose=False,
    )
    if DETERMINISTIC:
        umap2_kwargs["random_state"] = 42  # may switch to brute_force KNN
    umap2 = cuUMAP(**umap2_kwargs)
    viz = umap2.fit_transform(X)
    viz = to_numpy(viz)
    viz_method = "cuML-UMAP-2D"
except Exception:
    import umap.umap_ as umap_cpu
    umap2 = umap_cpu.UMAP(
        n_neighbors=UMAP_NEIGHBORS_VIZ,
        min_dist=UMAP_MIN_DIST_VIZ,
        n_components=2,
        metric="cosine",
        random_state=(42 if DETERMINISTIC else None),
        verbose=False,
    )
    viz = umap2.fit_transform(X)
    viz_method = "umap-learn-2D (CPU)"

df["viz_x"] = viz[:, 0].astype(np.float32)
df["viz_y"] = viz[:, 1].astype(np.float32)
print("2D viz via", viz_method)

# --------------- Normalize codes and label clusters ----------
codes_col = "llm_dis_map_lgmde" if "llm_dis_map_lgmde" in df.columns else None
if codes_col is None:
    print("Warning: 'llm_dis_map_lgmde' not found; code-based labels will be empty.", file=sys.stderr)
    df["codes_norm"] = [[] for _ in range(len(df))]
else:
    df["codes_norm"] = df[codes_col].apply(flatten_codes)
    if SHORTEN_CODE_PREFIX:
        df["codes_norm"] = df["codes_norm"].apply(lambda lst: [code_id_prefix(x) for x in lst])

code_labels = top_codes_per_cluster(df, top_k=10, col="codes_norm")

# c-TF-IDF keywords from TIAB (CPU scikit-learn)
try:
    kw_labels = ctfidf_keywords(df, top_k=10)
except Exception as e_kw:
    print("Keyword labeling failed; proceeding with code labels only:", e_kw)
    kw_labels = {}

cluster_label_map = {cid: compose_label(cid, code_labels, kw_labels)
                     for cid in sorted(set(df.cluster_id)) if cid >= 0}
df["cluster_label"] = df["cluster_id"].map(cluster_label_map)

# --------------------- Centroids for labels -----------------
centroids = (
    df[df.cluster_id >= 0]
    .groupby("cluster_id")[["viz_x", "viz_y"]]
    .mean()
    .reset_index()
)
centroids["label"] = centroids["cluster_id"].map(cluster_label_map)

# --------------------------- Save ---------------------------
df.to_parquet(OUT_PARQUET, index=False)
print(f"Saved labeled clusters and 2D coords to {OUT_PARQUET}")

# --------------------------- Plot (optional) ----------------
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 12), dpi=200)
# is_noise = df["cluster_id"] == -1
# plt.scatter(df.loc[is_noise, "viz_x"], df.loc[is_noise, "viz_y"],
#             s=2, c="#E0E0E0", alpha=0.35, linewidths=0)
# non_noise = df.loc[~is_noise]
# if len(non_noise) > 0:
#     plt.scatter(non_noise["viz_x"], non_noise["viz_y"],
#                 s=2, c=non_noise["cluster_id"], cmap="tab20", alpha=0.85, linewidths=0)
# largest = (
#     df[df.cluster_id >= 0]
#     .groupby("cluster_id")
#     .size()
#     .sort_values(ascending=False)
#     .head(N_ANNOTATE)
#     .index
# )
# for _, row in centroids[centroids.cluster_id.isin(largest)].iterrows():
#     plt.text(row.viz_x, row.viz_y, row.label,
#              fontsize=8, ha="center", va="center",
#              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85))
# plt.title(f"{used_umap} + {used_clusterer} | 2D via {viz_method} | n={n} | clusters={n_clusters} | noise={n_noise}")
# plt.axis("off")
# plt.tight_layout()
# plt.savefig(OUT_PNG)
# print(f"Saved plot to {OUT_PNG}")
