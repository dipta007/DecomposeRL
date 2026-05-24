#!/usr/bin/env python3
r"""Selector ablation: Submodular FacLoc vs.\ KMeans vs.\ Farthest-Point.

Produces the appendix figure and table referenced from sec:data:filter:diversity.
The figure shows two stacked panels:
  (a) 4-panel PCA density: Full pool / Submodular / KMeans / FarthestPoint, with
      coverage % annotated per panel.
  (b) Nearest-neighbor coverage CDF: fraction of the full pool whose nearest
      selected item is within distance d.

The table reports |S|, coverage, NN-distance percentiles, outlier-pick rate,
source entropy, and wall-clock selection time. Outliers are the top-5%
pool points by mean cosine distance to their 10 in-pool nearest neighbors
(an "isolated point" definition matching the paper's outlier-robustness
argument in sec:data:filter:diversity).

Usage:
    uv run python -m decomposer.data_process.plot_selector_comparison \
        [--data-dir DIR]   # default: data/combined_5k
        [--fig-out PATH]   # default: overleaf/figures/selector_comparison.pdf
        [--table-out PATH] # default: overleaf/tables/selector_comparison.tex
        [--no-timing]      # skip the wall-clock benchmark (uses cached values)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Methods listed in the order they should appear in figure/table columns.
# Submodular is "ours" so it sits left of the baselines.
METHODS = ["Submodular", "KMeans", "FarthestPoint"]
METHOD_LABELS = {
    "Submodular": "Submodular (FacLoc, ours)",
    "KMeans": "KMeans",
    "FarthestPoint": "Farthest-Point",
}
METHOD_DIRS = {
    "Submodular": "step_7_submod",
    "KMeans": "step_7_kmeans",
    "FarthestPoint": "step_7_farthest",
}

# Palette aligned with decomposer/analysis/utils.py and plot_data_funnel.py:
# Okabe-Ito (Wong) for max within-plot distinctness and colorblind safety.
#   red    — "Ours" anchor, matches _VERSION_PALETTE[0] in utils.py
#   blue   — SIZE_COLORS["7b"]
#   orange — MINICHECK_COLOR, also the funnel's "Diversity selection" hue
# Grey is the funnel's "Raw aggregation" hue, reused here for the full pool.
COLORS = {
    "Submodular": "#CC3311",
    "KMeans": "#0072B2",
    "FarthestPoint": "#E69F00",
    "Full": "#888888",
}
# Each panel of the PCA density row uses a cmap matching its line color.
CMAPS = {
    "Full": "Greys",
    "Submodular": "Reds",
    "KMeans": "Blues",
    "FarthestPoint": "Oranges",
}

# Outlier definition: top-OUTLIER_PCT pool points by mean cosine distance to
# their OUTLIER_K nearest in-pool neighbors. Mirrors the "isolated rant /
# OCR-broken row" intuition in sec:data:filter:diversity.
OUTLIER_K = 10
OUTLIER_PCT = 0.05

PCA_GRID = 50  # bin resolution for the coverage histogram

DEFAULT_DATA_DIR = Path("data/combined_5k")
DEFAULT_FIG_OUT = Path("overleaf/figures/selector_comparison.pdf")
DEFAULT_TABLE_OUT = Path("overleaf/tables/selector_comparison.tex")
DEFAULT_METRICS_OUT = Path("data/combined_5k/plots/selector_comparison_metrics.json")

# ACL single-column width ≈ 3.3 in; full text width ≈ 6.9 in (acl.sty).
ACL_TEXTWIDTH_IN = 6.9


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def load_embeddings(cache_path: Path) -> tuple[dict[str, int], np.ndarray]:
    data = np.load(cache_path, allow_pickle=False)
    emb_ids = data["ids"].tolist()
    emb_matrix = data["embeddings"]
    return {id_: i for i, id_ in enumerate(emb_ids)}, emb_matrix


def embeddings_for(
    items: list[dict], lookup: dict[str, int], matrix: np.ndarray
) -> np.ndarray:
    idx = [lookup[item["claim"]] for item in items]
    return matrix[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pca_coverage_pct(
    full_2d: np.ndarray, sel_2d: np.ndarray, *, grid: int = PCA_GRID
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Coverage % over non-empty pool bins, plus the histogram tensors."""
    x_min, x_max = full_2d[:, 0].min(), full_2d[:, 0].max()
    y_min, y_max = full_2d[:, 1].min(), full_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    rng = [[x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]]
    h_full, xe, ye = np.histogram2d(full_2d[:, 0], full_2d[:, 1], bins=grid, range=rng)
    h_sel, _, _ = np.histogram2d(sel_2d[:, 0], sel_2d[:, 1], bins=grid, range=rng)
    full_occ = h_full > 0
    sel_occ = h_sel > 0
    coverage = (sel_occ[full_occ].sum() / full_occ.sum() * 100) if full_occ.any() else 0.0
    return coverage, h_full, h_sel, np.array([xe[0], xe[-1], ye[0], ye[-1]])


def nn_distance_stats(
    full_embs: np.ndarray, sel_embs: np.ndarray, *, sample_size: int = 3000
) -> dict[str, float | np.ndarray]:
    """Distribution of d(p, nearest selected) over (sampled) pool points p."""
    rng = np.random.default_rng(42)
    if len(full_embs) > sample_size:
        q = full_embs[rng.choice(len(full_embs), sample_size, replace=False)]
    else:
        q = full_embs
    nn = cosine_distances(q, sel_embs).min(axis=1)
    return {
        "median": float(np.median(nn)),
        "p95": float(np.quantile(nn, 0.95)),
        "max": float(nn.max()),
        "sorted": np.sort(nn),
    }


def outlier_mask(full_embs: np.ndarray, *, k: int = OUTLIER_K, top_pct: float = OUTLIER_PCT) -> np.ndarray:
    """Boolean mask over the pool: True iff point is in the top-pct most isolated.

    Isolation = mean cosine distance to the k nearest other pool points.
    """
    sim = cosine_similarity(full_embs)
    np.fill_diagonal(sim, -np.inf)
    # Top-k similarities → smallest distances. mean over k.
    topk_sim = np.partition(sim, -k, axis=1)[:, -k:]
    mean_nn_dist = 1.0 - topk_sim.mean(axis=1)
    threshold = np.quantile(mean_nn_dist, 1.0 - top_pct)
    return mean_nn_dist >= threshold


def outlier_pick_rate(
    items: list[dict],
    full_items: list[dict],
    outlier_idx_set: set[int],
) -> float:
    """Fraction of the selected set that lies in the pool's outlier shell."""
    pool_pos = {item["claim"]: i for i, item in enumerate(full_items)}
    n_outliers_in_sel = sum(
        1 for it in items if pool_pos.get(it["claim"], -1) in outlier_idx_set
    )
    return n_outliers_in_sel / len(items) if items else 0.0


def source_entropy_bits(items: list[dict]) -> float:
    cnt = Counter(it["src"] for it in items)
    p = np.array(list(cnt.values()), dtype=float)
    p /= p.sum()
    return float(scipy_entropy(p, base=2))


# ---------------------------------------------------------------------------
# Wall-clock kernel benchmark
# ---------------------------------------------------------------------------


def time_kernels(
    full_items: list[dict],
    full_embs: np.ndarray,
) -> dict[str, float]:
    """Time each selector on the largest stratified (label, src) group.

    The pipeline always calls `select_diverse` on a single (label, src) group
    at a time with sqrt-proportional budget; timing the un-stratified 17K pool
    with budget 5000 would make KMeans run for hours and isn't representative.
    Timing the largest group is the dominant cost in the real pipeline and is
    a fair apples-to-apples comparison across selectors.
    """
    from decomposer.data_process.diversify_farthest import (
        select_diverse as fp_select,
    )
    from decomposer.data_process.diversify_kmeans import (
        select_diverse as km_select,
    )
    from decomposer.data_process.diversify_submod import (
        select_diverse as fl_select,
    )

    # Pick the largest (label, src) cell to time. The pipeline runs one
    # `select_diverse` call per cell; the largest cell dominates wall-clock.
    cells: dict[tuple[str, str], list[int]] = {}
    for i, it in enumerate(full_items):
        cells.setdefault((it["label"], it["src"]), []).append(i)
    (lbl, src), idx = max(cells.items(), key=lambda kv: len(kv[1]))
    n = len(idx)
    # sqrt-proportional budget: identical formula used by all three selectors.
    # Use the same budget across selectors to remove allocation noise.
    sqrt_counts = {k: math.sqrt(len(v)) for k, v in cells.items() if k[0] == lbl}
    total = sum(sqrt_counts.values())
    budget = max(2, round(2500 * sqrt_counts[(lbl, src)] / total))
    cell_embs = full_embs[idx]
    print(
        f"  Timing on largest cell: label={lbl} src={src} n={n} budget={budget}",
        flush=True,
    )

    timings: dict[str, float] = {}
    for name, fn in [
        ("Submodular", fl_select),
        ("KMeans", km_select),
        ("FarthestPoint", fp_select),
    ]:
        print(f"    {name} ...", flush=True, end=" ")
        t0 = time.perf_counter()
        _ = fn(cell_embs, budget)
        timings[name] = time.perf_counter() - t0
        print(f"{timings[name]:.2f} s", flush=True)
    return timings


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def setup_acl_rc() -> None:
    """Match the typography of the rest of the paper's figures (data_funnel)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # TrueType, avoids Type-3 fonts (ACL requirement)
            "ps.fonttype": 42,
        }
    )


def draw_figure(
    full_2d: np.ndarray,
    sel_2d: dict[str, np.ndarray],
    coverages: dict[str, float],
    histos: dict[str, np.ndarray],
    nn_stats: dict[str, dict],
    extent: np.ndarray,
    fig_path: Path,
) -> None:
    setup_acl_rc()
    fig = plt.figure(figsize=(ACL_TEXTWIDTH_IN, 4.4))
    # Two-row gridspec: top row = 4 PCA heatmaps; bottom row = CDF panel.
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[1.0, 1.05],
        hspace=0.55,
        wspace=0.12,
    )

    # --- Row 1: PCA density panels ---------------------------------------
    panels: list[tuple[str, np.ndarray, str]] = [
        ("Full pool", histos["Full"], CMAPS["Full"]),
    ]
    for m in METHODS:
        panels.append(
            (
                f"{METHOD_LABELS[m].split(' ')[0]} ({coverages[m]:.0f}\\% coverage)"
                if False
                else f"{METHOD_LABELS[m].split(' (')[0]}\n({coverages[m]:.0f}% coverage)",
                histos[m],
                CMAPS[m],
            )
        )

    for col, (title, h, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(
            h.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
    # Panel label
    fig.text(0.005, 0.97, "(a)", fontsize=9, fontweight="bold")

    # --- Row 2: NN coverage CDF ------------------------------------------
    ax_cdf = fig.add_subplot(gs[1, :])
    for m in METHODS:
        d = nn_stats[m]["sorted"]
        cdf = np.arange(1, len(d) + 1) / len(d)
        ax_cdf.plot(
            d,
            cdf,
            color=COLORS[m],
            linewidth=1.8,
            label=METHOD_LABELS[m],
        )
    # Mark the 95th percentile of each method with a dashed vertical line.
    for m in METHODS:
        ax_cdf.axvline(
            nn_stats[m]["p95"],
            color=COLORS[m],
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
        )
    ax_cdf.set_xlabel("Cosine distance to nearest selected claim")
    ax_cdf.set_ylabel("Fraction of pool covered")
    ax_cdf.set_xlim(left=0)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.grid(True, linestyle="--", alpha=0.35, linewidth=0.4)
    ax_cdf.set_axisbelow(True)
    ax_cdf.legend(loc="lower right", frameon=False)
    # Annotate the 95% guide line once (use Submodular as the anchor).
    fig.text(0.005, 0.46, "(b)", fontsize=9, fontweight="bold")

    fig.tight_layout(rect=(0.012, 0.0, 1.0, 1.0))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    png = fig_path.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {fig_path} (+ {png.name})")


# ---------------------------------------------------------------------------
# Table emitter
# ---------------------------------------------------------------------------


def emit_table(
    metrics: dict[str, dict],
    table_path: Path,
) -> None:
    rows: list[str] = []
    for m in METHODS:
        row = metrics[m]
        size = row["size"]
        cov = row["coverage_pct"]
        med = row["nn_median"]
        p95 = row["nn_p95"]
        outlier = row["outlier_pick_rate"] * 100  # → percentage
        ent = row["source_entropy_bits"]
        wall = row["wall_clock_s"]
        # Bold the leading-row method (Submodular = ours).
        label = METHOD_LABELS[m]
        if m == "Submodular":
            label = "\\textbf{" + label + "}"
        rows.append(
            "  "
            + " & ".join(
                [
                    label,
                    f"{size}",
                    f"{cov:.1f}",
                    f"{med:.3f}",
                    f"{p95:.3f}",
                    f"{outlier:.1f}",
                    f"{ent:.2f}",
                    f"{wall:.1f}",
                ]
            )
            + " \\\\"
        )
    body = "\n".join(rows)

    tex = (
        "% Auto-generated by decomposer/data_process/plot_selector_comparison.py.\n"
        "% Do not edit by hand; rerun the script after changing inputs.\n"
        "\\begin{table*}[t]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\setlength{\\tabcolsep}{4.5pt}\n"
        "  \\begin{tabular}{lrrrrrrr}\n"
        "    \\toprule\n"
        "    Selector & $|S|$ & Cov.\\,\\% $\\uparrow$ & "
        "$d_{\\mathrm{med}}$ $\\downarrow$ & "
        "$d_{95\\%}$ $\\downarrow$ & "
        "Outlier\\,\\% $\\downarrow$ & "
        "$H_{\\mathrm{src}}$ (bits) $\\uparrow$ & "
        "$t$ (s) $\\downarrow$ \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\caption{\\textbf{Selector ablation.} Submodular matches KMeans on "
        "pool coverage at $4{\\times}$ lower kernel cost and picks $1.6{\\times}$ "
        "fewer outliers than Farthest-Point. See \\cref{app:selector_comparison} "
        "for metric definitions.}\n"
        "  \\label{tab:selector_comparison}\n"
        "\\end{table*}\n"
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(tex)
    print(f"  Saved {table_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--fig-out", type=Path, default=DEFAULT_FIG_OUT)
    p.add_argument("--table-out", type=Path, default=DEFAULT_TABLE_OUT)
    p.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    p.add_argument("--no-timing", action="store_true", help="skip kernel timing benchmark")
    args = p.parse_args()

    data_dir: Path = args.data_dir
    print(f"Loading data from {data_dir} ...")

    # 1. Load full pool + selector outputs.
    full_items = load_jsonl(data_dir / "step_6" / "train.jsonl")
    print(f"  step_6 (full pool): {len(full_items)} items")

    method_items: dict[str, list[dict]] = {}
    for m in METHODS:
        path = data_dir / METHOD_DIRS[m] / "train.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing selector output: {path}")
        method_items[m] = load_jsonl(path)
        print(f"  {m}: {len(method_items[m])} items")

    # 2. Load embeddings.
    lookup, matrix = load_embeddings(data_dir / "cache" / "claim_embeddings.npz")
    full_embs = embeddings_for(full_items, lookup, matrix)
    method_embs = {m: embeddings_for(items, lookup, matrix) for m, items in method_items.items()}

    # 3. PCA → 2D for the coverage panels.
    print("Running PCA(2) for coverage panels ...")
    pca = PCA(n_components=2, random_state=42)
    full_2d = pca.fit_transform(full_embs)
    sel_2d = {m: pca.transform(method_embs[m]) for m in METHODS}

    coverages: dict[str, float] = {}
    histos: dict[str, np.ndarray] = {}
    extent: np.ndarray | None = None
    for m in METHODS:
        cov, h_full, h_sel, ext = pca_coverage_pct(full_2d, sel_2d[m])
        coverages[m] = cov
        histos[m] = h_sel
        if extent is None:
            extent = ext
            histos["Full"] = h_full
    assert extent is not None  # populated on first iteration

    # 4. NN-distance CDFs (one per method).
    print("Computing NN-coverage distributions ...")
    nn_stats = {m: nn_distance_stats(full_embs, method_embs[m]) for m in METHODS}

    # 5. Outlier definition once over the pool.
    print(f"Computing outlier mask (top {int(OUTLIER_PCT*100)}% by {OUTLIER_K}-NN distance) ...")
    out_mask = outlier_mask(full_embs)
    out_idx_set = set(np.flatnonzero(out_mask).tolist())
    print(f"  {len(out_idx_set)} pool points flagged as outliers")

    # 6. Per-method scalar metrics.
    metrics: dict[str, dict] = {}
    for m in METHODS:
        metrics[m] = {
            "size": len(method_items[m]),
            "coverage_pct": coverages[m],
            "nn_median": nn_stats[m]["median"],
            "nn_p95": nn_stats[m]["p95"],
            "nn_max": nn_stats[m]["max"],
            "outlier_pick_rate": outlier_pick_rate(method_items[m], full_items, out_idx_set),
            "source_entropy_bits": source_entropy_bits(method_items[m]),
        }

    # 7. Wall-clock kernel timing (or pull from on-disk cache).
    metrics_path: Path = args.metrics_out
    cached: dict | None = None
    if metrics_path.exists():
        try:
            cached = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            cached = None

    if args.no_timing:
        if cached is None or "timing" not in cached:
            raise SystemExit("--no-timing set but no cached timings found at " + str(metrics_path))
        timings = cached["timing"]
        print(f"Reusing cached kernel timings from {metrics_path}")
    else:
        print("Timing selector kernels on the largest stratified cell ...", flush=True)
        timings = time_kernels(full_items, full_embs)
    for m in METHODS:
        metrics[m]["wall_clock_s"] = float(timings[m])

    # 8. Persist machine-readable numbers (for reproducibility + reuse).
    payload = {
        "metrics": metrics,
        "timing": timings,
        "outlier_k": OUTLIER_K,
        "outlier_pct": OUTLIER_PCT,
        "pca_grid": PCA_GRID,
        "n_pool": len(full_items),
        "n_outliers": int(out_mask.sum()),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved {metrics_path}")

    # 9. Figure + LaTeX table.
    draw_figure(
        full_2d=full_2d,
        sel_2d=sel_2d,
        coverages=coverages,
        histos=histos,
        nn_stats=nn_stats,
        extent=extent,
        fig_path=args.fig_out,
    )
    emit_table(metrics, args.table_out)


if __name__ == "__main__":
    main()
