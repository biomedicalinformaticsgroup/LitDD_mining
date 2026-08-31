#!/usr/bin/env python3
"""Figure for the cross-encoder gate analysis (R3.6): what each threshold buys and costs.

Two small multiples over the score threshold (no dual axes):
  left  - the trade: in-scope external recall and test precision (0-1) with the corpus
          false-fire rate on its own panel below;
  right - the selection criteria: benchmark F1 and precision-weighted F0.5 (mean over
          seeds, +-1 sd band on F0.5), with the F0.5 plateau (within 1 sd of the maximum)
          shaded and the deployed 0.9 / argmax 0.95 marked.
Reads the fine-grid CSV from ``crossencoder_gate_analysis.py``. Colours are the first three
validated categorical slots of the reference palette (blue / orange / aqua); text uses ink
tokens, never series colour.
"""
from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", default="revision/crossencoder_gate_analysis_fine.csv")
    p.add_argument("--out", default="revision/figures/crossencoder_gate_analysis.png")
    p.add_argument("--deployed", type=float, default=0.9)
    return p.parse_args()


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)


def main() -> int:
    args = parse_args()
    d = pd.read_csv(args.csv, comment="#", index_col=0)
    t = d.index.values
    t_opt = d.bench_f05_lb.idxmax()
    mx, sd = d.bench_f05_lb.max(), d.loc[t_opt, "bench_f05_lb_sd"]
    plateau = d.index[d.bench_f05_lb >= mx - sd]

    fig = plt.figure(figsize=(11, 5.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.4], hspace=0.12, wspace=0.28)
    ax_tr = fig.add_subplot(gs[0, 0])
    ax_fire = fig.add_subplot(gs[1, 0], sharex=ax_tr)
    ax_cr = fig.add_subplot(gs[:, 1])
    for ax in (ax_tr, ax_fire, ax_cr):
        style(ax)

    # left-top: recall / precision trade
    ax_tr.plot(t, d.inscope_pair_recall, color=BLUE, lw=2, label="in-scope external recall")
    ax_tr.plot(t, d.test_precision, color=ORANGE, lw=2, label="test precision")
    ax_tr.plot(t, d.test_recall, color=AQUA, lw=2, label="test recall")
    ax_tr.set_ylim(0.6, 1.0)
    ax_tr.set_ylabel("proportion", color=INK2, fontsize=9)
    ax_tr.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    plt.setp(ax_tr.get_xticklabels(), visible=False)

    # left-bottom: false fires
    ax_fire.plot(t, d.fire_pct, color=BLUE, lw=2)
    ax_fire.set_ylabel("false fires\n(% of 87,600)", color=INK2, fontsize=9)
    ax_fire.set_xlabel("cross-encoder score threshold", color=INK2, fontsize=9)
    ax_fire.set_ylim(0, d.fire_pct.max() * 1.15)

    # right: criteria
    ax_cr.axvspan(plateau.min(), plateau.max(), color=GRID, alpha=0.6, lw=0,
                  label=f"F0.5 within 1 sd of max ({plateau.min():.2f}-{plateau.max():.2f})")
    ax_cr.fill_between(t, d.bench_f05_lb - d.bench_f05_lb_sd, d.bench_f05_lb + d.bench_f05_lb_sd,
                       color=ORANGE, alpha=0.15, lw=0)
    ax_cr.plot(t, d.bench_f05_lb, color=ORANGE, lw=2, label="benchmark F0.5 (precision-weighted)")
    ax_cr.plot(t, d.bench_f1_lb, color=BLUE, lw=2, label="benchmark F1")
    ax_cr.set_ylim(0.6, 0.82)
    ax_cr.set_xlabel("cross-encoder score threshold", color=INK2, fontsize=9)
    ax_cr.set_ylabel("score", color=INK2, fontsize=9)
    for ax in (ax_tr, ax_fire, ax_cr):
        ax.axvline(args.deployed, color=INK2, lw=1, ls="--")
        ax.axvline(t_opt, color=INK2, lw=1, ls=":")
    ax_cr.annotate(f"deployed {args.deployed:.2f}", (args.deployed, 0.615), color=INK2,
                   fontsize=8.5, ha="right", xytext=(-4, 0), textcoords="offset points")
    ax_cr.annotate(f"F0.5 max {t_opt:.2f}", (t_opt, 0.615), color=INK2, fontsize=8.5,
                   ha="left", xytext=(4, 0), textcoords="offset points")
    ax_cr.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)

    fig.suptitle("Cross-encoder gate: what each threshold buys and costs (3-seed means)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[Info] wrote {args.out}  (F0.5 max at {t_opt:.2f}, plateau "
          f"{plateau.min():.2f}-{plateau.max():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
