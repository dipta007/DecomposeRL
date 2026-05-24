"""Compare outputs of diversity methods: Submodular, KMeans, KMeans+DPP.

Prints quick stats (source/label distributions, pairwise overlap, diversity
metrics) and generates comparison plots.

Run:  uv run python -m decomposer.data_process.compare_diversity

Outputs plots to: data/combined/step_7/plots/
"""

import itertools
import math
from collections import Counter
from pathlib import Path

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALL_METHODS = {
    "Submodular": Path("data/combined/step_7_submod"),
    "KMeans": Path("data/combined/step_7_kmeans"),
    "KMeans+DPP": Path("data/combined/step_7_kmeans_dpp"),
    "FarthestPoint": Path("data/combined/step_7_farthest"),
}
# Filter to methods whose directories (and train.jsonl) actually exist
METHODS = {
    name: path
    for name, path in ALL_METHODS.items()
    if (path / "train.jsonl").exists()
}
INPUT_DIR = Path("data/combined/step_6")  # full dataset (pre-diversity)
EMBEDDING_CACHE_PATH = "data/combined/cache/claim_embeddings.npz"
PLOT_DIR = Path("data/combined/plots")

TSNE_SAMPLE = 3000  # subsample for t-SNE (full set can be slow)
COVERAGE_GRID = 50  # grid resolution for PCA coverage heatmap
_COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
COLORS = {name: _COLOR_PALETTE[i % len(_COLOR_PALETTE)] for i, name in enumerate(METHODS)}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_train(path: Path) -> list[dict]:
    with jsonlines.open(path / "train.jsonl", "r") as reader:
        return list(reader)


def load_embeddings():
    data = np.load(EMBEDDING_CACHE_PATH, allow_pickle=False)
    emb_ids = data["ids"].tolist()
    emb_matrix = data["embeddings"]
    emb_lookup = {id_: i for i, id_ in enumerate(emb_ids)}
    return emb_lookup, emb_matrix


def get_embeddings(items, emb_lookup, emb_matrix):
    indices = [emb_lookup[item["claim"]] for item in items]
    return emb_matrix[indices].astype(np.float32)


def avg_pairwise_cosine_distance(embs: np.ndarray, sample_size: int = 2000) -> float:
    """Average pairwise cosine distance (1 - similarity). Subsamples if large."""
    if len(embs) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embs), sample_size, replace=False)
        embs = embs[idx]
    sim = cosine_similarity(embs)
    n = len(sim)
    triu_idx = np.triu_indices(n, k=1)
    distances = 1 - sim[triu_idx]
    return float(distances.mean())


def vendi_score(embs: np.ndarray, sample_size: int = 2000) -> float:
    """Compute Vendi score (exponential of von Neumann entropy of similarity matrix)."""
    if len(embs) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embs), sample_size, replace=False)
        embs = embs[idx]
    sim = cosine_similarity(embs)
    sim = (sim + 1) / 2  # shift from [-1,1] to [0,1]
    eigenvalues = np.linalg.eigvalsh(sim)
    eigenvalues = eigenvalues[eigenvalues > 0]
    eigenvalues = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return float(math.exp(entropy))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_distribution(name: str, items: list[dict]):
    print(f"\n  [{name}] Total: {len(items)}")

    label_counts = Counter(item["label"] for item in items)
    print(f"  Labels:")
    for lbl, cnt in sorted(label_counts.items()):
        pct = cnt / len(items) * 100
        print(f"    {lbl}: {cnt} ({pct:.1f}%)")

    src_counts = Counter(item["src"] for item in items)
    print(f"  Sources:")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(items) * 100
        print(f"    {src:<25} {cnt:>6} ({pct:.1f}%)")


def print_pairwise_overlap(all_items: dict[str, list[dict]]):
    claim_sets = {
        name: {item["claim"] for item in items} for name, items in all_items.items()
    }
    names = list(claim_sets.keys())

    print(f"\n  Pairwise overlap:")
    for a, b in itertools.combinations(names, 2):
        overlap = claim_sets[a] & claim_sets[b]
        union = claim_sets[a] | claim_sets[b]
        jaccard = len(overlap) / len(union) if union else 0
        only_a = len(claim_sets[a] - claim_sets[b])
        only_b = len(claim_sets[b] - claim_sets[a])
        print(f"    {a} vs {b}:")
        print(
            f"      {a} only: {only_a}  |  {b} only: {only_b}  |  Both: {len(overlap)}  |  Jaccard: {jaccard:.3f}"
        )

    # Items in all methods
    all_common = set.intersection(*claim_sets.values())
    print(f"    Common to all {len(names)} methods: {len(all_common)}")


def print_diversity_metrics(all_embs: dict[str, np.ndarray]):
    print(f"\n  Diversity metrics:")

    header = f"    {'Method':<20} {'Avg Cos Dist':>14} {'Vendi Score':>13}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, embs in all_embs.items():
        dist = avg_pairwise_cosine_distance(embs)
        vs = vendi_score(embs)
        print(f"    {name:<20} {dist:>14.4f} {vs:>13.2f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_source_comparison(all_items: dict[str, list[dict]]):
    names = list(all_items.keys())
    src_counters = {
        name: Counter(item["src"] for item in items)
        for name, items in all_items.items()
    }
    all_srcs = sorted(set().union(*src_counters.values()))

    n_methods = len(names)
    x = np.arange(len(all_srcs))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(names):
        offset = (i - (n_methods - 1) / 2) * width
        counts = [src_counters[name].get(s, 0) for s in all_srcs]
        ax.bar(x + offset, counts, width, label=name, color=COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(all_srcs, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Source Distribution Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "source_comparison.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'source_comparison.png'}")
    plt.close(fig)


def plot_label_comparison(all_items: dict[str, list[dict]]):
    names = list(all_items.keys())
    lbl_counters = {
        name: Counter(item["label"] for item in items)
        for name, items in all_items.items()
    }
    all_labels = sorted(set().union(*lbl_counters.values()))

    n_methods = len(names)
    x = np.arange(len(all_labels))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in enumerate(names):
        offset = (i - (n_methods - 1) / 2) * width
        counts = [lbl_counters[name].get(l, 0) for l in all_labels]
        ax.bar(x + offset, counts, width, label=name, color=COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "label_comparison.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'label_comparison.png'}")
    plt.close(fig)


def plot_source_label_heatmaps(all_items: dict[str, list[dict]]):
    """Side-by-side heatmaps of (source x label) counts."""
    names = list(all_items.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    # Collect all sources/labels across methods for consistent axes
    all_srcs = sorted(
        set().union(
            *(set(item["src"] for item in items) for items in all_items.values())
        )
    )
    all_labels = sorted(
        set().union(
            *(set(item["label"] for item in items) for items in all_items.values())
        )
    )

    for ax, name in zip(axes, names):
        items = all_items[name]
        counts = Counter((item["src"], item["label"]) for item in items)
        matrix = np.array(
            [[counts.get((s, l), 0) for l in all_labels] for s in all_srcs]
        )

        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt="d",
            xticklabels=all_labels,
            yticklabels=all_srcs,
            cmap="YlOrRd",
        )
        ax.set_title(name)

    fig.suptitle("Source x Label Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "source_label_heatmap.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'source_label_heatmap.png'}")
    plt.close(fig)


def plot_tsne(all_items: dict[str, list[dict]], emb_lookup, emb_matrix):
    """t-SNE visualization colored by which method(s) selected each point."""
    names = list(all_items.keys())
    claim_sets = {
        name: {item["claim"] for item in items} for name, items in all_items.items()
    }

    # Assign each claim a label based on which methods selected it
    all_claims = set().union(*claim_sets.values())
    groups = []
    for claim in all_claims:
        selected_by = [name for name in names if claim in claim_sets[name]]
        label = " + ".join(selected_by) if len(selected_by) < len(names) else "All"
        groups.append((claim, label))

    # Subsample if needed
    rng = np.random.default_rng(42)
    if len(groups) > TSNE_SAMPLE:
        indices = rng.choice(len(groups), TSNE_SAMPLE, replace=False)
        groups = [groups[i] for i in indices]

    claims = [g[0] for g in groups]
    labels = [g[1] for g in groups]
    emb_indices = [emb_lookup[c] for c in claims]
    embs = emb_matrix[emb_indices].astype(np.float32)

    print(f"  Running t-SNE on {len(embs)} points...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embs) - 1))
    coords = tsne.fit_transform(embs)

    # Build color map: single-method gets its color, combos get gray shades, All gets green
    unique_labels = sorted(set(labels))
    color_map = {}
    for lbl in unique_labels:
        if lbl == "All":
            color_map[lbl] = "#2ca02c"
        elif lbl in COLORS:
            color_map[lbl] = COLORS[lbl]
        else:
            color_map[lbl] = "#999999"

    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw "All" first (background), then combos, then single-method
    draw_order = sorted(unique_labels, key=lambda l: (l != "All", l.count("+"), l))
    for lbl in draw_order:
        mask = np.array([l == lbl for l in labels])
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color_map[lbl],
            label=f"{lbl} ({mask.sum()})",
            alpha=0.5,
            s=8,
        )
    ax.set_title("t-SNE: Diversity Method Comparison")
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tsne_comparison.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'tsne_comparison.png'}")
    plt.close(fig)


def plot_diversity_bar(all_embs: dict[str, np.ndarray]):
    """Bar chart comparing diversity metrics across methods."""
    names = list(all_embs.keys())
    cos_dists = [avg_pairwise_cosine_distance(all_embs[n]) for n in names]
    vendis = [vendi_score(all_embs[n]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(names))
    bars = axes[0].bar(x, cos_dists, color=[COLORS[n] for n in names])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("Avg Pairwise Cosine Distance")
    axes[0].set_title("Embedding Diversity")
    for bar, val in zip(bars, cos_dists):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars = axes[1].bar(x, vendis, color=[COLORS[n] for n in names])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("Vendi Score")
    axes[1].set_title("Vendi Diversity Score")
    for bar, val in zip(bars, vendis):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "diversity_metrics.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'diversity_metrics.png'}")
    plt.close(fig)


def plot_pairwise_distance_kde(all_embs: dict[str, np.ndarray], sample_size: int = 2000):
    """KDE of pairwise cosine distances per method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)

    for name, embs in all_embs.items():
        if len(embs) > sample_size:
            idx = rng.choice(len(embs), sample_size, replace=False)
            embs = embs[idx]
        sim = cosine_similarity(embs)
        triu_idx = np.triu_indices(len(sim), k=1)
        distances = 1 - sim[triu_idx]
        # Subsample distances for KDE if too many pairs
        if len(distances) > 100_000:
            distances = rng.choice(distances, 100_000, replace=False)
        sns.kdeplot(distances, ax=ax, label=name, color=COLORS[name], fill=True, alpha=0.2)

    ax.set_xlabel("Pairwise Cosine Distance")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise Distance Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pairwise_distance_kde.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'pairwise_distance_kde.png'}")
    plt.close(fig)


def plot_coverage_heatmap(
    all_items: dict[str, list[dict]],
    full_embs: np.ndarray,
    all_embs: dict[str, np.ndarray],
):
    """PCA coverage heatmap: how each method covers the full embedding space."""
    print("  Running PCA for coverage heatmap...")
    pca = PCA(n_components=2, random_state=42)
    full_2d = pca.fit_transform(full_embs)

    names = list(all_items.keys())
    n = len(names)

    # Compute grid bounds from full dataset
    x_min, x_max = full_2d[:, 0].min(), full_2d[:, 0].max()
    y_min, y_max = full_2d[:, 1].min(), full_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 4))

    # Full dataset density
    h_full, xedges, yedges = np.histogram2d(
        full_2d[:, 0], full_2d[:, 1], bins=COVERAGE_GRID,
        range=[[x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]],
    )
    axes[0].imshow(
        h_full.T, origin="lower", aspect="auto", cmap="Greys",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    axes[0].set_title(f"Full dataset ({len(full_embs)})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for i, name in enumerate(names):
        method_2d = pca.transform(all_embs[name])
        h, _, _ = np.histogram2d(
            method_2d[:, 0], method_2d[:, 1], bins=COVERAGE_GRID,
            range=[[x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]],
        )
        axes[i + 1].imshow(
            h.T, origin="lower", aspect="auto", cmap="YlOrRd",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        # Coverage: fraction of non-empty bins that are covered
        full_occupied = h_full > 0
        method_occupied = h[full_occupied] > 0
        coverage = method_occupied.sum() / full_occupied.sum() * 100 if full_occupied.sum() else 0
        axes[i + 1].set_title(f"{name} ({coverage:.0f}% coverage)")
        axes[i + 1].set_xticks([])
        axes[i + 1].set_yticks([])

    fig.suptitle("PCA Coverage: Selected vs Full Dataset", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "coverage_heatmap.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'coverage_heatmap.png'}")
    plt.close(fig)


def plot_nn_coverage_curve(
    all_embs: dict[str, np.ndarray],
    full_embs: np.ndarray,
    sample_size: int = 3000,
):
    """CDF of nearest-neighbor distance from full dataset to each method's selection.

    Lower curve = better coverage (more full-dataset items are close to a selected item).
    """
    rng = np.random.default_rng(42)
    # Subsample full dataset for tractability
    if len(full_embs) > sample_size:
        idx = rng.choice(len(full_embs), sample_size, replace=False)
        query_embs = full_embs[idx]
    else:
        query_embs = full_embs

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, method_embs in all_embs.items():
        dists = cosine_distances(query_embs, method_embs)
        nn_dists = dists.min(axis=1)
        sorted_dists = np.sort(nn_dists)
        cdf = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
        ax.plot(sorted_dists, cdf, label=name, color=COLORS[name], linewidth=2)

    ax.set_xlabel("Cosine Distance to Nearest Selected Item")
    ax.set_ylabel("Fraction of Full Dataset")
    ax.set_title("Nearest-Neighbor Coverage Curve (lower-left = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "nn_coverage_curve.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'nn_coverage_curve.png'}")
    plt.close(fig)


def plot_radar(all_items: dict[str, list[dict]], all_embs: dict[str, np.ndarray]):
    """Radar chart comparing methods across multiple metrics."""
    names = list(all_items.keys())

    # Compute metrics
    metrics = {}
    for name in names:
        items = all_items[name]
        embs = all_embs[name]

        # Source entropy (higher = more balanced across sources)
        src_counts = Counter(item["src"] for item in items)
        src_probs = np.array(list(src_counts.values()), dtype=float)
        src_probs /= src_probs.sum()
        src_ent = float(scipy_entropy(src_probs, base=2))

        # Label balance (closer to 1 = more balanced)
        lbl_counts = Counter(item["label"] for item in items)
        lbl_probs = np.array(list(lbl_counts.values()), dtype=float)
        lbl_probs /= lbl_probs.sum()
        lbl_balance = 1.0 - float(np.abs(lbl_probs - 1 / len(lbl_probs)).sum() / 2)

        metrics[name] = {
            "Avg Cos Dist": avg_pairwise_cosine_distance(embs),
            "Vendi Score": vendi_score(embs),
            "Source Entropy": src_ent,
            "Label Balance": lbl_balance,
            "Total Items": len(items),
        }

    # Normalize each metric to [0, 1] for radar
    metric_names = list(next(iter(metrics.values())).keys())
    raw = {m: [metrics[n][m] for n in names] for m in metric_names}
    normalized = {}
    for m in metric_names:
        vals = raw[m]
        mn, mx = min(vals), max(vals)
        if mx > mn:
            normalized[m] = [(v - mn) / (mx - mn) for v in vals]
        else:
            normalized[m] = [1.0 for _ in vals]

    # Plot
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for i, name in enumerate(names):
        values = [normalized[m][i] for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=COLORS[name])
        ax.fill(angles, values, alpha=0.1, color=COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Method Comparison (normalized)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "radar_comparison.png", dpi=150)
    print(f"  Saved {PLOT_DIR / 'radar_comparison.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if len(METHODS) < 2:
        available = list(METHODS.keys()) or ["(none)"]
        missing = [n for n in ALL_METHODS if n not in METHODS]
        print(f"ERROR: Need at least 2 methods to compare, but only found: {', '.join(available)}")
        print(f"  Missing: {', '.join(missing)}")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data ({len(METHODS)} methods available)...")
    all_items: dict[str, list[dict]] = {}
    for name, dir_path in METHODS.items():
        all_items[name] = load_train(dir_path)
        print(f"  {name}: {len(all_items[name])} items from {dir_path}")

    # Load full dataset (for coverage plots)
    print("Loading full dataset...")
    full_items = load_train(INPUT_DIR)
    print(f"  Full dataset: {len(full_items)} items from {INPUT_DIR}")

    print("Loading embeddings...")
    emb_lookup, emb_matrix = load_embeddings()
    all_embs = {
        name: get_embeddings(items, emb_lookup, emb_matrix)
        for name, items in all_items.items()
    }
    full_embs = get_embeddings(full_items, emb_lookup, emb_matrix)

    # --- Stats ---
    print("\n" + "=" * 60)
    print("COMPARISON: " + " vs ".join(METHODS.keys()))
    print("=" * 60)

    for name, items in all_items.items():
        print_distribution(name, items)

    print_pairwise_overlap(all_items)
    print_diversity_metrics(all_embs)

    # --- Plots ---
    print("\nGenerating plots...")
    plot_source_comparison(all_items)
    plot_label_comparison(all_items)
    plot_source_label_heatmaps(all_items)
    plot_diversity_bar(all_embs)
    plot_pairwise_distance_kde(all_embs)
    plot_coverage_heatmap(all_items, full_embs, all_embs)
    plot_nn_coverage_curve(all_embs, full_embs)
    plot_radar(all_items, all_embs)
    plot_tsne(all_items, emb_lookup, emb_matrix)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
