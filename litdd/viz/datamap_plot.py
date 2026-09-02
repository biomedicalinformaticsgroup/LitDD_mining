#!/usr/bin/env python3
"""Render the literature-space datamap figure (static PNG + interactive HTML).

Pipeline: ``ce_tsne.py`` produces ``filtered_clean_genes_clusters_and_viz.parquet``
(2D ``viz_x``/``viz_y`` coords, ``cluster_id`` and ``cluster_label`` per record). This
script labels clusters with MONDO disease terms and renders them with ``datamapplot``:

  - Static plot: per-cluster label from a two-group refined LCA of the cluster's
    G2P->MONDO diseases (split into two groups by ancestor Jaccard distance, with an
    LCA per group, re-splitting if the LCA collapses to a generic banned term); the
    "primary" (first) label is plotted, collapsed to the top-N most frequent.
  - Interactive plot: multi-scale label layers from the single deterministic LCA
    coarsened to increasing ontology depths.

This is a single-file, de-duplicated rewrite of the original exploratory notebook (which
redefined the same helpers three times); the labelling logic that produced the figures is
preserved. Large inputs (clusters parquet, ``mondo.owl``) are not shipped; ``mondo.owl`` is
downloaded on first run if absent. Requires owlready2 and datamapplot (see requirements).
"""
from __future__ import annotations

import argparse
import re
import urllib.request
from collections import deque
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from owlready2 import get_ontology

MONDO_OWL_URL = "https://purl.obolibrary.org/obo/mondo.owl"
MONDO_ROOT_IRI = "http://purl.obolibrary.org/obo/MONDO_0700096"  # human disease
NOISE_LABEL = "NOISE"
OTHER_LABEL = "MONDO:OTHER"
# Generic labels we refuse as a cluster name (re-split / go deeper instead).
BANNED_LABELS = {"syndromic disease", "hereditary disease", "human disease", "autosomal genetic disease"}

_MONDO_ID_RE = re.compile(r"(?:https?://purl\.obolibrary\.org/obo/)?MONDO[:_]\s*(\d+)", re.IGNORECASE)
_G2P_RE = re.compile(r"\bG2P[^\s,;|/]*", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# G2P -> MONDO mapping and cluster-label parsing
# --------------------------------------------------------------------------- #
def normalize_mondo_curie(s) -> Optional[str]:
    """Normalize any MONDO-like token to 'MONDO:NNNNNNN' (7 digits), else None."""
    if not isinstance(s, str) or not s.strip() or s.strip().upper() == "NO MATCH":
        return None
    m = _MONDO_ID_RE.search(s)
    return f"MONDO:{m.group(1).zfill(7)}" if m else None


def build_g2p_to_mondo(g2p_file: str) -> Dict[str, str]:
    """Map G2P id (upper) -> MONDO CURIE from the G2P DD CSV."""
    g2p = pd.read_csv(g2p_file).dropna(subset=["g2p id", "disease MONDO"])
    mapping: Dict[str, str] = {}
    for gid, mondo in zip(g2p["g2p id"].astype(str), g2p["disease MONDO"].astype(str)):
        curie = normalize_mondo_curie(mondo)
        gid = gid.strip().upper()
        if curie and gid not in mapping:
            mapping[gid] = curie
    return mapping


def parse_g2p_codes(label) -> List[str]:
    """Parse G2P ids from the codes portion (before first '|') of a cluster label."""
    if not isinstance(label, str) or not label.strip():
        return []
    return [h.strip() for h in _G2P_RE.findall(label.split("|", 1)[0]) if h.strip()]


# --------------------------------------------------------------------------- #
# MONDO ontology utilities (owlready2)
# --------------------------------------------------------------------------- #
class Mondo:
    """Thin wrapper over the MONDO ontology: depths, ancestors, descendants, LCA."""

    def __init__(self, owl_path: str):
        path = Path(owl_path)
        if not path.exists():
            print(f"Downloading MONDO OWL to {path} ...")
            urllib.request.urlretrieve(MONDO_OWL_URL, path)
        self.onto = get_ontology(path.resolve().as_uri()).load()
        root = self.onto.search(iri=MONDO_ROOT_IRI)
        if not root:
            raise RuntimeError(f"MONDO root not found at {MONDO_ROOT_IRI}")
        self.depths = self._compute_depths(root[0])
        self._nodes = set(self.depths)

    @staticmethod
    def _is_mondo(cls) -> bool:
        iri = getattr(cls, "iri", "")
        return isinstance(iri, str) and iri.startswith("http://purl.obolibrary.org/obo/MONDO_")

    @staticmethod
    def _name_to_curie(name: str) -> str:
        return name.replace("_", ":").strip()

    def _compute_depths(self, root_cls) -> Dict[str, int]:
        depths = {root_cls.name: 0}
        q = deque([root_cls])
        while q:
            u = q.popleft()
            for v in u.subclasses():
                if self._is_mondo(v) and (v.name not in depths or depths[u.name] + 1 < depths[v.name]):
                    depths[v.name] = depths[u.name] + 1
                    q.append(v)
        return {self._name_to_curie(n): d for n, d in depths.items()}

    @lru_cache(maxsize=200000)
    def _class(self, curie: str):
        try:
            ent = self.onto.world[f"http://purl.obolibrary.org/obo/{curie.replace(':', '_')}"]
            if ent is not None:
                return ent
        except Exception:
            pass
        try:
            return self.onto[curie.replace(":", "_")]
        except KeyError:
            return None

    def label(self, curie: str) -> str:
        cls = self._class(curie)
        labels = getattr(cls, "label", None) if cls is not None else None
        return str(labels[0]) if labels else curie

    @lru_cache(maxsize=200000)
    def ancestors(self, curie: str) -> frozenset:
        cls = self._class(curie)
        if cls is None or not self._is_mondo(cls):
            return frozenset()
        out = {self._name_to_curie(a.name) for a in cls.ancestors() if self._is_mondo(a)}
        return frozenset(out & self._nodes)

    @lru_cache(maxsize=200000)
    def n_descendants(self, curie: str) -> int:
        cls = self._class(curie)
        if cls is None:
            return 0
        return max(0, sum(1 for x in cls.descendants() if self._is_mondo(x)) - 1)

    def lca(self, curies: List[str]) -> Optional[str]:
        """Deterministic LCA: fewest descendants, then deepest, then lexicographic."""
        curies = [c for c in curies if c]
        if not curies:
            return None
        if len(curies) == 1:
            return curies[0]
        common = set.intersection(*(set(self.ancestors(c)) for c in curies))
        if not common:
            return None
        return min(common, key=lambda c: (self.n_descendants(c), -self.depths.get(c, -10**9), c))

    def coarsen(self, curie: str, depth_cap: int) -> str:
        """Lift a CURIE to its deepest ancestor with depth <= depth_cap."""
        cands = [(a, self.depths.get(a, -10**9)) for a in self.ancestors(curie)]
        cands = [(a, d) for a, d in cands if d <= depth_cap]
        return max(cands, key=lambda x: x[1])[0] if cands else curie


# --------------------------------------------------------------------------- #
# Two-group refined LCA labelling (drives the static plot)
# --------------------------------------------------------------------------- #
def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return 1.0 - (len(a & b) / union) if union else 1.0


def _agglomerative_split_two(curies: List[str], mondo: Mondo) -> List[List[str]]:
    """Average-linkage agglomeration (Jaccard of ancestor sets) down to two groups."""
    n = len(curies)
    if n <= 1:
        return [curies, []]
    if n == 2:
        return [[curies[0]], [curies[1]]]

    anc = [mondo.ancestors(c) for c in curies]
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist[i, j] = dist[j, i] = _jaccard(anc[i], anc[j])

    clusters = [[i] for i in range(n)]
    while len(clusters) > 2:
        best, best_key = None, None
        for a_idx in range(len(clusters)):
            for b_idx in range(a_idx + 1, len(clusters)):
                ca, cb = clusters[a_idx], clusters[b_idx]
                d = float(np.mean([dist[i, j] for i in ca for j in cb]))
                key = (d, tuple(sorted(ca)), tuple(sorted(cb)))
                if best_key is None or key < best_key:
                    best, best_key = (a_idx, b_idx), key
        ai, bi = best
        merged = sorted(clusters[ai] + clusters[bi])
        clusters = [c for k, c in enumerate(clusters) if k not in (ai, bi)] + [merged]

    groups = [[curies[i] for i in sorted(c)] for c in clusters]
    groups.sort(key=lambda g: (-len(g), tuple(g)))
    return groups


def _group_lcas_refined(curies: List[str], mondo: Mondo) -> List[dict]:
    """LCA(s) for a group; if the LCA is a banned generic term, split and recurse."""
    curies = [c for c in curies if c]
    if not curies:
        return []
    lca = mondo.lca(curies)
    if lca is None:
        return [{"curie": c, "label": mondo.label(c), "size": 1, "depth": mondo.depths.get(c, -10**9)}
                for c in curies]
    label = mondo.label(lca).strip()
    if label.lower() in BANNED_LABELS and len(curies) >= 2:
        out: List[dict] = []
        for g in _agglomerative_split_two(curies, mondo):
            if g:
                out.extend(_group_lcas_refined(g, mondo))
        return out
    return [{"curie": lca, "label": label, "size": len(curies), "depth": mondo.depths.get(lca, -10**9)}]


def two_lca_label(g2p_codes: List[str], g2p_to_mondo: Dict[str, str], mondo: Mondo) -> str:
    """'; '-joined refined two-group LCA label for a cluster's G2P codes."""
    curies, seen = [], set()
    for c in g2p_codes:
        m = g2p_to_mondo.get(c.upper())
        if m and m not in seen:
            seen.add(m)
            curies.append(m)
    if not curies:
        return OTHER_LABEL
    if len(curies) == 1:
        return mondo.label(curies[0]).strip() or OTHER_LABEL

    infos: List[dict] = []
    for g in _agglomerative_split_two(curies, mondo):
        if g:
            infos.extend(_group_lcas_refined(g, mondo))
    uniq = {i["curie"]: i for i in infos}.values()
    infos = sorted(uniq, key=lambda x: (-x["size"], -x["depth"], x["label"]))
    labels = [i["label"] for i in infos if i["label"].strip()]
    return "; ".join(labels) if labels else OTHER_LABEL


# --------------------------------------------------------------------------- #
# Single-LCA multi-scale layers (drive the interactive plot)
# --------------------------------------------------------------------------- #
def _coarsened_label(curie: Optional[str], mondo: Mondo, depth_cap: Optional[int]) -> str:
    if not curie:
        return OTHER_LABEL
    if depth_cap is not None:
        curie = mondo.coarsen(curie, depth_cap)
        lbl = mondo.label(curie).strip()
        if lbl.lower() in BANNED_LABELS:
            lbl = mondo.label(mondo.coarsen(curie, depth_cap + 2)).strip()
        return lbl or OTHER_LABEL
    return mondo.label(curie).strip() or OTHER_LABEL


def build_label_layers(df: pd.DataFrame, g2p_to_mondo: Dict[str, str], mondo: Mondo,
                       depth_caps=(3, 6)) -> List[np.ndarray]:
    """Coarse -> fine layers: depth-capped LCA, ..., point LCA, raw primary."""
    n = len(df)
    layers = [np.empty(n, dtype=object) for _ in range(len(depth_caps) + 2)]
    cid = df["cluster_id"].to_numpy()
    raw = df["cluster_label"].astype(str).to_numpy()

    lca_cache: Dict[str, Optional[str]] = {}
    for i in range(n):
        if cid[i] == -1:
            for layer in layers:
                layer[i] = NOISE_LABEL
            continue
        label = raw[i]
        if label not in lca_cache:
            mondos = [g2p_to_mondo.get(c.upper()) for c in parse_g2p_codes(label)]
            lca_cache[label] = mondo.lca([m for m in mondos if m])
        curie = lca_cache[label]
        for j, cap in enumerate(depth_caps):
            layers[j][i] = _coarsened_label(curie, mondo, cap)
        layers[-2][i] = _coarsened_label(curie, mondo, None)           # point-level LCA
        layers[-1][i] = label.split("|", 1)[0].strip() or OTHER_LABEL  # raw primary
    return layers


def static_cluster_labels(df: pd.DataFrame, g2p_to_mondo: Dict[str, str], mondo: Mondo,
                          top_n: int) -> np.ndarray:
    """Per-point primary two-LCA label collapsed to the top-N (noise -> NOISE)."""
    primary_by_cluster: Dict[int, str] = {}
    for cid, sub in df.groupby("cluster_id"):
        if cid == -1:
            primary_by_cluster[cid] = NOISE_LABEL
            continue
        label = next((x for x in sub["cluster_label"].dropna().astype(str) if x.strip()), "")
        full = two_lca_label(parse_g2p_codes(label), g2p_to_mondo, mondo)
        primary_by_cluster[cid] = full.split(";", 1)[0].strip() or OTHER_LABEL

    labels = df["cluster_id"].map(primary_by_cluster).to_numpy()
    is_noise = df["cluster_id"].to_numpy() == -1
    keep = set(pd.Series(labels[~is_noise]).value_counts().head(top_n).index)
    labels[~np.isin(labels, list(keep))] = OTHER_LABEL
    labels[is_noise] = NOISE_LABEL
    return labels


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters-parquet", default="filtered_clean_genes_clusters_and_viz.parquet",
                    help="Output of ce_tsne.py (viz_x, viz_y, cluster_id, cluster_label)")
    ap.add_argument("--g2p-file", required=True, help="G2P DD CSV (for G2P->MONDO mapping)")
    ap.add_argument("--mondo-owl", default="mondo.owl", help="MONDO OWL file (downloaded if absent)")
    ap.add_argument("--out-static", default="datamapplot_output.png")
    ap.add_argument("--out-interactive", default="mondo_interactive_datamap.html")
    ap.add_argument("--out-layers", default="datamap_with_mondo_layers.parquet")
    ap.add_argument("--top-n", type=int, default=30, help="Labels to keep in the static plot")
    return ap.parse_args()


def main():
    import datamapplot

    args = parse_args()
    df = pd.read_parquet(args.clusters_parquet)
    g2p_to_mondo = build_g2p_to_mondo(args.g2p_file)
    mondo = Mondo(args.mondo_owl)

    coords = df[["viz_x", "viz_y"]].to_numpy()

    # Interactive multi-scale layers (single-LCA, coarse -> fine) + export.
    layers = build_label_layers(df, g2p_to_mondo, mondo)
    out_df = df[["viz_x", "viz_y", "cluster_id", "cluster_label"]].copy()
    for col, layer in zip(["label_l0_coarse", "label_l1_medium", "label_l2_lca", "label_l3_raw"], layers):
        out_df[col] = layer
    out_df.to_parquet(args.out_layers, index=False)
    print(f"Wrote {args.out_layers}")

    # Static plot: refined two-LCA primary labels, top-N.
    fig, _ = datamapplot.create_plot(
        coords, static_cluster_labels(df, g2p_to_mondo, mondo, args.top_n),
        use_medoids=True, noise_label=NOISE_LABEL,
        point_size=1.5, alpha=0.7, label_font_size=14, label_over_points=True,
        dynamic_label_size=True,
    )
    fig.savefig(args.out_static, bbox_inches="tight")
    print(f"Wrote {args.out_static}")

    # Interactive plot: layers passed fine -> coarse.
    datamapplot.create_interactive_plot(
        coords, *reversed(layers), use_medoids=True, noise_label=NOISE_LABEL, enable_search=True,
    ).save(args.out_interactive)
    print(f"Wrote {args.out_interactive}")


if __name__ == "__main__":
    main()
