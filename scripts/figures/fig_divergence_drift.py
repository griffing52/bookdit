#!/usr/bin/env python
"""Drift figures for the LOB-Bench rollout comparisons.

--only horizon-matched -> horizon_bands.{pdf,png}
    Aggregate L1 over 18 statistics against rollout horizon, out to 16 000 generated messages,
    for BookDiT, LOB-S5, Cont-Stoikov-Talreja and a real-vs-real control, on a window set that
    is byte-identical across all four arms at every block. The ribbon is +/- 1.96 bootstrap
    standard errors from the paired window bootstrap; the legend carries each arm's slope.
    Reads bootstrap.json and curve_bands_l1_18.csv from --matched-src: by default
    runs/horizon/bands, where scripts/analysis/horizon_matched_bands.py writes them.

--only divergence -> divergence_drift.{pdf,png}
    (a) Growth of the LOB-Bench divergence L1 with prediction horizon at the benchmark default
        divergence_horizon = 100, relative to each arm's own first block. The arms do not share
        a level -- BookDiT's books are IDM-decoded to messages before LOB-Bench can score them,
        while LOB-S5 emits messages directly -- so only the SLOPE is like for like.
    (b) Drift slope per 1000 generated messages with its bootstrap interval. The real-vs-real
        control also drifts, so the quantity annotated per arm is the excess over the control.
    Reads headline.csv and curve_bands.csv from --div-src: by default runs/divergence, where
    scripts/baselines/slopes_divergence_official.py and scripts/figures/build_divergence_bands.py
    write them.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
DIV = ROOT / "runs/divergence"
MATCHED = ROOT / "runs/horizon/bands"
OUTDIR = ROOT / "runs/figures"

# MUTED Okabe-Ito -- the six hues the paper's figures share, kept at their Okabe-Ito OKLCH hue
# angles with chroma scaled to ~0.80-0.86 (vermillion additionally darkened) so every figure sits
# in one low-chroma register; the palette was checked for lightness banding and colour-vision-
# deficiency separation.  The control/grey is deliberately neutral ink, not a categorical hue.
C_BOOK, C_LOBS5, C_CTRL = "#2E75A9", "#BC5A19", "#5F5F5F"
C_MAR = "#CA80A9"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

ARMS = [  # (headline.csv arm name, display, color, marker)
    ("LOB-S5 (released samples)", "LOB-S5 (message model)", C_LOBS5, "s"),
    ("BookDiT", "BookDiT (ours)", C_BOOK, "o"),
    ("real-vs-real floor", "real vs. real control", C_CTRL, "^"),
]
# headline.csv `arm` label -> curve_bands.csv `model` id (the arm directory names that
# setup_divergence_official.py builds).
MODEL_OF = {"LOB-S5 (released samples)": "lobs5", "BookDiT": "bookdit",
            "real-vs-real floor": "realgen"}
# Short forms for the \columnwidth build: the display names above are ~22 characters, which at
# 6 pt eat 0.95 in of a 3.35 in canvas as panel-(b) tick labels and overrun panel (a)'s legend.
SHORT = {"LOB-S5 (released samples)": "LOB-S5", "BookDiT": "BookDiT",
         "real-vs-real floor": "real vs. real"}


def style(ax, title=None, fs_title=10, fs_tick=8.5, pad=7):
    """Shared axis styling.  The defaults suit a full-width figure; both figures here pass their
    own (smaller, because they are authored 1:1 at \\columnwidth and every point size is
    therefore the printed one)."""
    if title:
        ax.set_title(title, fontsize=fs_title, loc="left", color=INK, pad=pad)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=fs_tick, length=2.5, width=0.6)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_color(INK)


# ------------------------------------------------------------------- divergence growth
def fig_divergence(head: pd.DataFrame, bands: pd.DataFrame, out: Path) -> dict:
    # Authored at ACM sigconf \columnwidth = 241.15 pt = 3.35 in, the width the figure is included
    # at, so every point size below is the PRINTED point size.  The two panels are STACKED, not
    # side by side: 1.6 in per panel is too narrow for these annotations.
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7})
    FS_LAB, FS_TICK, FS_ANN, FS_TITLE = 6.6, 6.0, 6.0, 7.0
    # Panel (b) carries H = 100 only; the H = 50 robustness check is not drawn, but
    # fig_divergence() returns both horizons' ratios and main() prints them.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.35, 3.0),
                                 gridspec_kw={"height_ratios": [1.35, 1.0]})

    # ---- (a) drift curves at the benchmark default H = 100
    H0 = 100
    for arm, disp, col, mk in ARMS:
        b = bands[(bands.model == MODEL_OF[arm]) & (bands.divergence_horizon == H0)]
        h = head[(head.arm == arm) & (head.divergence_horizon == H0)].iloc[0]
        x = b.horizon_msgs.values
        a1.fill_between(x, b.rel_lo.values, b.rel_hi.values, color=col, alpha=0.16, lw=0)
        a1.plot(x, b.rel_point.values, marker=mk, ms=3.4, lw=1.4, color=col,
                mec="white", mew=0.5, zorder=3,
                label=f"{SHORT[arm]}  (abs. $L_1$ {h.l1_first:.3f})")
        yv = b.rel_point.values
        a1.annotate(f"{h.slope_per_1k:+.3f} / 1k", (x[-1], yv[-1]),
                    textcoords="offset points", xytext=(4, -1), fontsize=FS_ANN, color=col,
                    va="center", fontweight="bold")
    a1.axhline(0, color=GRID, lw=0.8, zorder=1)
    # Max over the PLOTTED arms only: any other model in curve_bands.csv may reach a different
    # depth, and the span shown must be the matched span headline.csv states.
    xmax = float(bands[(bands.divergence_horizon == H0)
                       & (bands.model.isin([MODEL_OF[arm] for arm, *_ in ARMS]))]
                 .horizon_msgs.max())
    assert xmax == float(head[(head.arm == "BookDiT")
                              & (head.divergence_horizon == H0)].span_msgs.iloc[0]), \
        f"plotted span {xmax} disagrees with headline.csv span_msgs"
    a1.set_xlim(H0 - 8, xmax + 46)                   # room for the end-of-curve slope labels
    # The data span including the bootstrap bands is [-0.0033, +0.0435]; this is that span plus a
    # hair, with the top 30% reserved for the 3-row legend.
    a1.set_ylim(-0.006, 0.062)
    a1.set_xlabel("prediction horizon [generated messages]", fontsize=FS_LAB, color=INK)
    a1.set_ylabel("divergence growth\n$\\Delta L_1$ since 1st block", fontsize=FS_LAB, color=INK)
    style(a1, "(a)  compounding error, $H{=}100$ (default)",
          fs_title=FS_TITLE, fs_tick=FS_TICK, pad=3)
    a1.legend(fontsize=FS_ANN, frameon=False, loc="upper left", labelcolor=INK,
              handlelength=1.3, borderpad=0.0, handletextpad=0.5, labelspacing=0.22)

    # ---- (b) slope with bootstrap CI at the benchmark default, excess over the control
    PLOT_H = (100,)
    info = {}
    y, ticks, labels, group_span = 0.0, [], [], {}
    for H in (50, 100):
        y0 = y
        for arm, disp, col, mk in reversed(ARMS):      # control at the bottom of each block
            h = head[(head.arm == arm) & (head.divergence_horizon == H)].iloc[0]
            info[(arm, H)] = h
            if H not in PLOT_H:
                continue
            a2.plot([h.slope_lo, h.slope_hi], [y, y], lw=1.0, color=col, solid_capstyle="butt",
                    alpha=0.85, zorder=3)
            a2.plot([h.slope_lo, h.slope_lo, np.nan, h.slope_hi, h.slope_hi],
                    [y - 0.13, y + 0.13, np.nan, y - 0.13, y + 0.13], lw=0.8, color=col, zorder=3)
            a2.plot([h.slope_per_1k], [y], marker=mk, ms=3.2, color=col, mec="white", mew=0.5,
                    zorder=4)
            ticks.append(y)
            labels.append(SHORT[arm])
            if arm != "real-vs-real floor":
                a2.annotate(f"{h.excess_slope_per_1k:+.4f}", (h.slope_hi, y),
                            textcoords="offset points", xytext=(4, 0), fontsize=FS_ANN,
                            color=col, va="center", fontweight="bold")
            y += 1.0
        if H in PLOT_H:
            group_span[H] = (y0 - 0.42, y - 0.58)
            y += 0.85

    # control band: everything to its right is drift the real-vs-real control does not explain
    for H, (lo_y, hi_y) in group_span.items():
        c = info[("real-vs-real floor", H)]
        a2.add_patch(Rectangle((c.slope_lo, lo_y), c.slope_hi - c.slope_lo, hi_y - lo_y,
                               facecolor=C_CTRL, alpha=0.10, lw=0, zorder=1))
        a2.plot([c.slope_per_1k, c.slope_per_1k], [lo_y, hi_y], ls=":", lw=0.8, color=C_CTRL,
                zorder=2)

    a2.axvline(0, color=GRID, lw=0.8, zorder=1)
    a2.set_yticks(ticks)
    a2.set_yticklabels(labels, fontsize=FS_TICK)
    a2.set_ylim(-0.75, max(ticks) + 0.55)
    a2.set_xlim(-0.012, 0.325)
    a2.set_xlabel("drift slope  [$L_1$ per 1000 generated messages]", fontsize=FS_LAB, color=INK)
    style(a2, "(b)  drift slope, 95% CI; bold = excess over control",
          fs_title=FS_TITLE, fs_tick=FS_TICK, pad=3)
    a2.grid(axis="y", visible=False)

    ratios = {}
    for H in (50, 100):
        b = info[("BookDiT", H)]
        l = info[("LOB-S5 (released samples)", H)]
        c = info[("real-vs-real floor", H)]
        # slope-free cross-check: cumulative drift at the deepest horizon block all three arms reach
        sub = {arm: bands[(bands.model == MODEL_OF[arm]) & (bands.divergence_horizon == H)]
               for arm, *_ in ARMS}
        common = int(min(s.horizon_msgs.max() for s in sub.values()))
        rel = {arm: float(s.loc[s.horizon_msgs == common, "rel_point"].iloc[0])
               for arm, s in sub.items()}
        cum_b = rel["BookDiT"] - rel["real-vs-real floor"]
        cum_l = rel["LOB-S5 (released samples)"] - rel["real-vs-real floor"]
        ratios[H] = dict(
            ratio=l.excess_slope_per_1k / b.excess_slope_per_1k,
            conservative=(l.slope_lo - c.slope_hi) / (b.slope_hi - c.slope_lo),
            book=b.excess_slope_per_1k, lobs5=l.excess_slope_per_1k,
            matched_msgs=common, cum_book=cum_b, cum_lobs5=cum_l, cum_ratio=cum_l / cum_b)
    assert ratios[50]["matched_msgs"] == ratios[100]["matched_msgs"]
    # Annotation sits in the empty strip right of the control's interval; the per-arm excesses
    # are already annotated in bold above.
    a2.text(0.985, 0.02,
            f"LOB-S5 : BookDiT excess  {ratios[100]['ratio']:.1f}$\\times$",
            transform=a2.transAxes, ha="right", va="bottom", fontsize=FS_ANN, color=INK,
            fontweight="bold")

    fig.tight_layout(pad=0.35, h_pad=0.9)
    # tight_layout, not bbox_inches="tight": the latter grows the canvas past the authored
    # 3.35 in and \includegraphics[width=\columnwidth] then downscales it.
    save(fig, out)
    return ratios


# ------------------------------------------------- matched window set + real-vs-real floor
def fig_horizon_matched(out: Path, src: Path) -> dict:
    """Banded horizon profile on a fixed window set shared by all arms, plus a real-vs-real floor.

    Both inputs come from scripts/analysis/horizon_matched_bands.py.

    (1) Population.  The population is the intersection over every block of every arm, so all
        four arms see byte-identical windows at all 32 blocks.  Letting each arm score whichever
        windows it has at that block -- the composition rule the published profiles use -- leaves
        some of the block-to-block scatter as composition rather than model, because the number
        of LOB-S5 windows that pass the official level-ten filter changes from block to block.
        The intersection costs sample size and widens the CI.

    (2) Real-vs-real floor.  LOB-Bench's L1 is a plug-in total variation between two binned
        histograms, so two finite samples of real data score well above zero.  The grey line is
        that number: the population's windows split into two disjoint halves, the real score
        arrays of each half pooled, and the identical L1 taken between them (24 random splits per
        block, fixed seed).  It says how much of every curve's level is the estimator rather than
        the model, and -- being real data on both sides -- its slope is the drift a perfectly
        stationary generator would still show.

        Its level is an upper bound: each side holds half the windows, and the plug-in TV falls
        with sample size.  Its slope is unaffected -- every block uses the same window count --
        and the slope is what the figure compares.

    Grey is the control, here as in the divergence figure.  CST takes the palette's
    reddish-purple rather than grey so the two figures can be read with one legend convention.
    """
    bands = pd.read_csv(src / "curve_bands_l1_18.csv")
    boot = json.loads((src / "bootstrap.json").read_text())

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7})
    FS_LAB, FS_TICK, FS_ANN, FS_TITLE = 6.6, 6.0, 6.0, 7.0
    fig, a1 = plt.subplots(1, 1, figsize=(3.35, 2.25))

    x = bands.horizon.values
    band = {}
    for key, disp, col, mk, lw, z in (
            ("real",    "real vs. real control", C_CTRL,  "^", 0.9, 2),
            ("cst",     "Cont–Stoikov–Talreja",  C_MAR,   "d", 0.9, 2),
            ("lobs5",   "LOB-S5 (compute-matched)", C_LOBS5, "s", 1.2, 3),
            ("bookdit", "BookDiT (0.95M, ours)", C_BOOK,  "o", 1.4, 4)):
        s = boot["arms"][key]
        y, se = bands[f"{key}_point"].values, bands[f"{key}_se"].values
        band[key] = (y - 1.96 * se, y + 1.96 * se)
        a1.fill_between(x, band[key][0], band[key][1], color=col, alpha=0.16, lw=0, zorder=z - 1.5)
        # The CI is in the caption, not the legend: four rows do not fit \columnwidth at 6 pt.
        a1.plot(x, y, marker=mk, ms=2.4, lw=lw, color=col, mec="white",
                mew=0.4, alpha=1.0 if key not in ("cst", "real") else 0.85, zorder=z,
                label=f"{disp}  {s['slope_per_1k']*1000:+.2f}"
                      r"$\times$10$^{-3}$/1k")
    lo = min(b[0].min() for b in band.values())
    hi = max(b[1].max() for b in band.values())
    a1.set_ylim(lo - 0.030, hi + 0.135)          # top margin holds the 4-row legend
    a1.set_xlim(0, 16600)
    a1.set_xlabel("rollout horizon [generated messages]", fontsize=FS_LAB, color=INK)
    a1.set_ylabel("aggregate $L_1^{18}$\n(official protocol)", fontsize=FS_LAB, color=INK)
    a1.set_xticks([0, 4000, 8000, 12000, 16000])
    a1.set_xticklabels(["0", "4k", "8k", "12k", "16k"])
    cl = boot["contrasts"]["lobs5"]
    style(a1, f"BookDiT drifts {cl['ratio']:.1f}$\\times$ slower than LOB-S5",
          fs_title=FS_TITLE, fs_tick=FS_TICK)
    a1.legend(fontsize=FS_ANN, frameon=False, loc="upper left", labelcolor="linecolor",
              handlelength=1.5, borderpad=0.0, handletextpad=0.5, labelspacing=0.22)
    # The window count, B, and the slope-gap interval live in the caption, not on the canvas.

    fig.tight_layout(pad=0.35)
    save(fig, out)
    return boot


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print("wrote", f"{out}.pdf / .png")
    plt.close(fig)


def require(src: Path, names: tuple, how: str, flag: str) -> None:
    """Fail with a readable message, not a traceback, when an input dir is incomplete."""
    missing = [n for n in names if not (src / n).exists()]
    if missing:
        raise SystemExit(f"{src} lacks {', '.join(missing)}; {how} (docs/REPRODUCE.md §3.9).\n"
                         f"  Run that first, or point {flag} at the directory it wrote.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(OUTDIR),
                    help="where the PDF and PNG are written (default runs/figures)")
    ap.add_argument("--matched-src", default=str(MATCHED),
                    help="source dir for --only horizon-matched: bootstrap.json and "
                         "curve_bands_l1_18.csv (default runs/horizon/bands, the "
                         "horizon_matched_bands.py output)")
    ap.add_argument("--div-src", default=str(DIV),
                    help="source dir for --only divergence: headline.csv and curve_bands.csv "
                         "(default runs/divergence, the slopes_divergence_official.py and "
                         "build_divergence_bands.py output)")
    ap.add_argument("--only", choices=["divergence", "horizon-matched"],
                    default="horizon-matched",
                    help="which figure to (re)write; the other is left untouched on disk.")
    a = ap.parse_args()
    outdir = Path(a.outdir)

    if a.only == "horizon-matched":
        src = Path(a.matched_src)
        require(src, ("bootstrap.json", "curve_bands_l1_18.csv"),
                "both are written by scripts/analysis/horizon_matched_bands.py",
                "--matched-src")
        hb = fig_horizon_matched(outdir / "horizon_bands", src)
        print(f"\n--- matched window bootstrap ({hb['n_windows']} windows, "
              f"population={hb['population_mode']}), L1^18 drift slope per 1000 messages ---")
        for arm, s in hb["arms"].items():
            print(f"  {arm:8s} {s['first']:.4f} -> {s['last']:.4f}  "
                  f"{s['slope_per_1k']*1000:+.3f}e-3  "
                  f"95% CI [{s['lo']*1000:+.3f}e-3, {s['hi']*1000:+.3f}e-3]")
        print("--- slope differences, arm - reference ---")
        for key, c in hb["contrasts"].items():
            ref = c["reference"]
            arm = key[:-len(f"_vs_{ref}")] if key.endswith(f"_vs_{ref}") else key
            # A ratio is meaningful only against BookDiT's clearly positive slope; the real
            # control's slope is ~0, so a ratio over it (or of it) is noise.
            ratio = (f"ratio {c['ratio']:.2f}x" if ref == "bookdit" and arm != "real"
                     else "ratio n/a")
            print(f"  {arm:7s} - {ref:7s} {c['diff_per_1k']*1000:+.3f}e-3 "
                  f"[{c['diff_lo']*1000:+.3f}, {c['diff_hi']*1000:+.3f}]e-3  "
                  f"{ratio:13s} P(diff <= 0) = {c['p_diff_le0']:.4f}")
        return

    div_src = Path(a.div_src)
    require(div_src, ("headline.csv", "curve_bands.csv"),
            "they are written by scripts/baselines/slopes_divergence_official.py and "
            "scripts/figures/build_divergence_bands.py", "--div-src")
    head = pd.read_csv(div_src / "headline.csv")
    bands = pd.read_csv(div_src / "curve_bands.csv")

    ratios = fig_divergence(head, bands, outdir / "divergence_drift")
    print("\n--- recomputed excess drift over the real-vs-real control ---")
    for H, d in ratios.items():
        print(f"H={H:3d}  slope   BookDiT {d['book']:+.5f}   LOB-S5 {d['lobs5']:+.5f}   "
              f"ratio {d['ratio']:.3f}x   (CI-worst-case lower bound "
              f"{d['conservative']:.2f}x)")
        print(f"      cum@{d['matched_msgs']}  BookDiT {d['cum_book']:+.5f}   "
              f"LOB-S5 {d['cum_lobs5']:+.5f}   ratio {d['cum_ratio']:.3f}x")


if __name__ == "__main__":
    main()
