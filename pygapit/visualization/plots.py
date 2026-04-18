"""
Visualization module.
Translates GAPIT.Manhattan.R, GAPIT.QQ.R, GAPIT.PCA.R,
GAPIT.GS.Visualization.R, GAPIT.Phenotype.View.R

All plots are publication-ready and match GAPIT's visual style.
Static plots use matplotlib/seaborn.
Interactive plots use plotly (same package as GAPIT's plotly R).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ── Color palette matching GAPIT's default ───────────────────────────────
CHR_COLORS = [
    "#3C5587", "#89A8D0",   # alternating blue shades for chromosomes
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
    "#3C5587", "#89A8D0",
]
SIG_COLOR = "#E41A1C"       # red for significant hits
SUGGEST_COLOR = "#FF7F00"   # orange for suggestive


def manhattan_plot(
    snp_names: np.ndarray,
    chromosomes: np.ndarray,
    positions: np.ndarray,
    p_values: np.ndarray,
    title: str = "Manhattan Plot",
    significance_threshold: float = None,
    suggestive_threshold: float = None,
    highlight_snps: np.ndarray = None,
    save_path: str = None,
    figsize: tuple = (14, 5),
    point_size: float = 1.5,
) -> plt.Figure:
    """
    Manhattan plot.
    Translates GAPIT.Manhattan.R

    Parameters
    ----------
    snp_names            : SNP identifiers
    chromosomes          : chromosome labels
    positions            : genomic positions (bp)
    p_values             : association p-values
    significance_threshold : genome-wide significance line (default: Bonferroni)
    suggestive_threshold : suggestive line (default: 1e-5)
    highlight_snps       : indices of SNPs to highlight red
    save_path            : if provided, save to this path
    """
    # ── Data prep ────────────────────────────────────────────────────────
    valid = ~np.isnan(p_values) & (p_values > 0) & (p_values <= 1)
    p_vals = np.where(valid, p_values, 1.0)
    log_p = -np.log10(np.where(p_vals > 0, p_vals, 1e-300))

    m = len(p_values)
    if significance_threshold is None:
        significance_threshold = 0.05 / m
    if suggestive_threshold is None:
        suggestive_threshold = 1.0 / m

    sig_line = -np.log10(significance_threshold)
    sug_line = -np.log10(suggestive_threshold)

    # ── Compute cumulative positions ─────────────────────────────────────
    chroms = np.array([str(c) for c in chromosomes])
    unique_chroms = []
    seen = set()
    for c in chroms:
        if c not in seen:
            unique_chroms.append(c)
            seen.add(c)

    chrom_offset = {}
    cumulative = 0
    chrom_centers = {}
    for chrom in unique_chroms:
        mask = chroms == chrom
        max_pos = np.nanmax(positions[mask]) if mask.any() else 0
        chrom_offset[chrom] = cumulative
        chrom_centers[chrom] = cumulative + max_pos / 2
        cumulative += max_pos + 5_000_000  # gap between chromosomes

    x_vals = np.array([
        float(positions[i]) + chrom_offset.get(chroms[i], 0)
        for i in range(m)
    ])

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    for ci, chrom in enumerate(unique_chroms):
        mask = chroms == chrom
        color = CHR_COLORS[ci % len(CHR_COLORS)]
        ax.scatter(
            x_vals[mask], log_p[mask],
            c=color, s=point_size, linewidths=0, rasterized=True, alpha=0.8,
        )

    # Highlight significant SNPs
    if highlight_snps is not None and len(highlight_snps) > 0:
        ax.scatter(
            x_vals[highlight_snps], log_p[highlight_snps],
            c=SIG_COLOR, s=point_size * 4, linewidths=0, zorder=5,
        )

    # Threshold lines
    ax.axhline(y=sig_line, color=SIG_COLOR, linestyle="--", linewidth=0.8, alpha=0.9)
    ax.axhline(y=sug_line, color=SUGGEST_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)

    # Axis formatting
    ax.set_xlim(0, x_vals.max() * 1.01)
    ax.set_ylim(0, max(log_p.max() * 1.1, sig_line * 1.2))
    ax.set_xticks([chrom_centers[c] for c in unique_chroms])
    ax.set_xticklabels(unique_chroms, fontsize=7)
    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def qq_plot(
    p_values: np.ndarray,
    title: str = "QQ Plot",
    save_path: str = None,
    figsize: tuple = (5, 5),
) -> plt.Figure:
    """
    Quantile-Quantile plot with genomic inflation factor.
    Translates GAPIT.QQ.R

    Diagonal = expected under null hypothesis (no association).
    Deviation upward at right tail = true associations.
    Uniform upward deviation = population stratification (λ > 1).
    """
    from ..stats.testing import genomic_inflation_factor

    valid = ~np.isnan(p_values) & (p_values > 0) & (p_values <= 1)
    p_obs = np.sort(p_values[valid])
    n = len(p_obs)

    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No valid p-values")
        return fig

    # Expected quantiles
    expected = -np.log10(np.arange(1, n + 1) / n)
    observed = -np.log10(p_obs[::-1])

    # Confidence band (95% CI)
    ci_upper = -np.log10(
        np.clip(np.array([
            max(1e-300, p_obs[int(i * n)] if int(i * n) < n else 1.0)
            for i in np.linspace(0, 1, n, endpoint=False)[::-1]
        ]), 1e-300, 1.0)
    )

    # Lambda
    lam = genomic_inflation_factor(p_values)

    fig, ax = plt.subplots(figsize=figsize)

    # Confidence band
    upper_ci = np.sort(expected)[::-1] * 1.3  # simple approximation
    ax.fill_between(
        np.sort(expected)[::-1],
        np.sort(expected)[::-1],
        np.sort(expected)[::-1] * 1.3,
        alpha=0.15, color="gray",
    )

    # Diagonal
    max_val = max(observed.max(), expected.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.7)

    # Points
    ax.scatter(
        np.sort(expected)[::-1],
        observed,
        c="#3C5587", s=4, linewidths=0, alpha=0.7,
    )

    ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=10)
    ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=10)
    ax.set_title(f"{title}\n(λ = {lam:.3f})", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def kinship_heatmap(
    K: np.ndarray,
    taxa: np.ndarray = None,
    title: str = "Kinship Matrix",
    save_path: str = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """
    Heatmap of genomic kinship matrix.
    Translates GAPIT.Genotype.View.R (kinship heatmap section)

    Color scale: blue=low kinship, red=high kinship
    Sorted by hierarchical clustering (like GAPIT's heatmap.2)
    """
    from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
    from scipy.spatial.distance import squareform

    n = K.shape[0]
    dist = np.clip(1.0 - K, 0, None)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    try:
        condensed = squareform(dist)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    except Exception:
        order = np.arange(n)

    K_sorted = K[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(K_sorted, aspect="auto", cmap="RdBu_r",
                   vmin=K.min(), vmax=K.max())
    plt.colorbar(im, ax=ax, shrink=0.8, label="Kinship")

    if taxa is not None and n <= 50:
        taxa_sorted = np.array(taxa)[order]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(taxa_sorted, rotation=90, fontsize=6)
        ax.set_yticklabels(taxa_sorted, fontsize=6)
    else:
        ax.set_xlabel("Individuals", fontsize=10)
        ax.set_ylabel("Individuals", fontsize=10)

    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def pca_plot_2d(
    scores: np.ndarray,
    var_explained: np.ndarray,
    taxa: np.ndarray = None,
    groups: np.ndarray = None,
    title: str = "PCA Plot",
    save_path: str = None,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """
    2D PCA scatter plot (PC1 vs PC2).
    Translates GAPIT.PCA.R static plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    pc1, pc2 = scores[:, 0], scores[:, 1]

    if groups is not None:
        unique_groups = np.unique(groups)
        palette = plt.cm.tab10(np.linspace(0, 0.9, len(unique_groups)))
        for gi, g in enumerate(unique_groups):
            mask = groups == g
            ax.scatter(pc1[mask], pc2[mask], s=15, alpha=0.8,
                       color=palette[gi], label=str(g), linewidths=0)
        ax.legend(fontsize=8, markerscale=2, framealpha=0.5)
    else:
        ax.scatter(pc1, pc2, s=10, alpha=0.7, color="#3C5587", linewidths=0)

    pct1 = var_explained[0] * 100 if len(var_explained) > 0 else 0
    pct2 = var_explained[1] * 100 if len(var_explained) > 1 else 0
    ax.set_xlabel(f"PC1 ({pct1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pct2:.1f}%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def pca_plot_3d_interactive(
    scores: np.ndarray,
    var_explained: np.ndarray,
    taxa: np.ndarray = None,
    groups: np.ndarray = None,
    title: str = "3D PCA",
    save_path: str = None,
) -> object:
    """
    Interactive 3D PCA using Plotly.
    Translates GAPIT.3D.PCA.python.R — same plotly calls, Python syntax.
    Identical to GAPIT's interactive HTML output.
    """
    if not HAS_PLOTLY:
        warnings.warn("plotly not installed; skipping 3D interactive PCA.")
        return None

    pc1 = scores[:, 0] if scores.shape[1] > 0 else np.zeros(len(scores))
    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
    pc3 = scores[:, 2] if scores.shape[1] > 2 else np.zeros(len(scores))

    pct = [v * 100 for v in var_explained[:3]] if len(var_explained) >= 3 else [0, 0, 0]

    hover_text = taxa.tolist() if taxa is not None else [str(i) for i in range(len(pc1))]

    fig = go.Figure(data=[go.Scatter3d(
        x=pc1, y=pc2, z=pc3,
        mode="markers",
        marker=dict(
            size=4,
            color=pc1,
            colorscale="Viridis",
            opacity=0.85,
        ),
        text=hover_text,
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({pct[0]:.1f}%)",
            yaxis_title=f"PC2 ({pct[1]:.1f}%)",
            zaxis_title=f"PC3 ({pct[2]:.1f}%)",
        ),
        width=700, height=600,
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def manhattan_interactive(
    snp_names: np.ndarray,
    chromosomes: np.ndarray,
    positions: np.ndarray,
    p_values: np.ndarray,
    effects: np.ndarray = None,
    maf: np.ndarray = None,
    title: str = "Interactive Manhattan",
    save_path: str = None,
) -> object:
    """
    Interactive Manhattan plot with hover info.
    Translates GAPIT.Interactive.Manhattan.R
    Hover shows: SNP name, chromosome, position, MAF, p-value, effect.
    """
    if not HAS_PLOTLY:
        warnings.warn("plotly not installed; skipping interactive Manhattan.")
        return None

    valid = ~np.isnan(p_values) & (p_values > 0)
    m = len(p_values)

    chroms = np.array([str(c) for c in chromosomes])
    unique_chroms = []
    seen = set()
    for c in chroms:
        if c not in seen:
            unique_chroms.append(c)
            seen.add(c)

    chrom_offset = {}
    cumulative = 0
    for chrom in unique_chroms:
        mask = chroms == chrom
        max_pos = np.nanmax(positions[mask]) if mask.any() else 0
        chrom_offset[chrom] = cumulative
        cumulative += max_pos + 5_000_000

    x_vals = np.array([
        float(positions[i]) + chrom_offset.get(chroms[i], 0)
        for i in range(m)
    ])
    log_p = -np.log10(np.where(valid, np.maximum(p_values, 1e-300), 1.0))

    # Build hover text
    hover = []
    for i in range(m):
        txt = (
            f"<b>{snp_names[i]}</b><br>"
            f"Chr: {chromosomes[i]}, Pos: {int(positions[i]):,}<br>"
            f"P-value: {p_values[i]:.2e}<br>"
        )
        if effects is not None:
            txt += f"Effect: {effects[i]:.4f}<br>"
        if maf is not None:
            txt += f"MAF: {maf[i]:.3f}"
        hover.append(txt)

    sig_threshold = 0.05 / m
    sig_line = -np.log10(sig_threshold)

    fig = go.Figure()
    for ci, chrom in enumerate(unique_chroms):
        mask = chroms == chrom
        color = CHR_COLORS[ci % len(CHR_COLORS)]
        fig.add_trace(go.Scatter(
            x=x_vals[mask], y=log_p[mask],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.7),
            text=np.array(hover)[mask],
            hovertemplate="%{text}<extra></extra>",
            name=f"Chr {chrom}",
            showlegend=False,
        ))

    fig.add_hline(y=sig_line, line_dash="dash", line_color=SIG_COLOR,
                  annotation_text="Bonferroni", annotation_position="right")

    fig.update_layout(
        title=title,
        xaxis_title="Genomic Position",
        yaxis_title="-log₁₀(p)",
        hovermode="closest",
        width=900, height=400,
        plot_bgcolor="white",
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def gs_scatter(
    observed: np.ndarray,
    predicted: np.ndarray,
    taxa: np.ndarray = None,
    trait_name: str = "Trait",
    save_path: str = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """
    Genomic Selection scatter: predicted vs observed.
    Translates GAPIT.GS.Visualization.R
    Pearson r = prediction accuracy.
    """
    valid = ~(np.isnan(observed) | np.isnan(predicted))
    obs_v = observed[valid]
    pred_v = predicted[valid]

    if len(obs_v) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Insufficient data for GS scatter")
        return fig

    r = np.corrcoef(obs_v, pred_v)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(obs_v, pred_v, s=15, alpha=0.6, color="#3C5587", linewidths=0)

    # Regression line
    m_coef = np.polyfit(obs_v, pred_v, 1)
    x_line = np.linspace(obs_v.min(), obs_v.max(), 100)
    ax.plot(x_line, np.polyval(m_coef, x_line), "r-", linewidth=1.2, alpha=0.8)

    ax.set_xlabel(f"Observed {trait_name}", fontsize=10)
    ax.set_ylabel(f"Predicted {trait_name}", fontsize=10)
    ax.set_title(f"GS Accuracy (r = {r:.3f})", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def phenotype_distribution(
    y: np.ndarray,
    trait_name: str = "Trait",
    significant_snp_geno: np.ndarray = None,
    save_path: str = None,
    figsize: tuple = (6, 4),
) -> plt.Figure:
    """
    Phenotype distribution histogram.
    Translates GAPIT.Phenotype.View.R
    Optionally split by genotype at top significant SNP.
    """
    fig, ax = plt.subplots(figsize=figsize)
    valid_y = y[~np.isnan(y)]

    ax.hist(valid_y, bins=30, color="#3C5587", alpha=0.75, edgecolor="white", linewidth=0.3)

    if significant_snp_geno is not None:
        for geno_val, label, color in [(0, "Ref/Ref", "#2166AC"),
                                        (1, "Ref/Alt", "#74ADD1"),
                                        (2, "Alt/Alt", "#D73027")]:
            mask = (significant_snp_geno == geno_val) & ~np.isnan(y)
            if mask.any():
                ax.hist(y[mask], bins=20, alpha=0.5, color=color, label=label, edgecolor="none")
        ax.legend(fontsize=8)

    ax.set_xlabel(trait_name, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Distribution of {trait_name}", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
