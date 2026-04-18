"""
pyGAPIT — Python reimplementation of GAPIT
(Genome Association and Prediction Integrated Tool)

Main entry point: GAPIT()

Mirrors the R GAPIT() function signature exactly, so users
familiar with R GAPIT can switch to Python with minimal changes.

R:      myGAPIT <- GAPIT(Y=myY, GD=myGD, GM=myGM, model="BLINK", PCA.total=3)
Python: myGAPIT  = GAPIT(Y=myY,  GD=myGD, GM=myGM, model="BLINK", PCA_total=3)

(R dots replaced with underscores in Python, e.g. PCA.total → PCA_total)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List

from .io.formats import (
    read_hapmap, read_numeric, read_phenotype,
    align_taxa, maf_filter, GenotypeData, PhenotypeData,
)
from .stats.kinship import vanraden_kinship, scale_kinship
from .stats.pca import compute_pca, build_covariate_matrix
from .stats.testing import (
    bonferroni_threshold, benjamini_hochberg,
    genomic_inflation_factor, get_significant_snps,
)
from .gwas.glm import glm_gwas
from .gwas.mlm import mlm_gwas, cmlm_gwas
from .gwas.mlmm import mlmm_gwas
from .gwas.blink import blink_gwas
from .gwas.farmcpu import farmcpu_gwas
from .gs.blup import gblup, cblup, sblup, GBLUPResult


@dataclass
class GAPITResult:
    """
    Return object from GAPIT().
    Mirrors the named list returned by R's GAPIT().
    """
    # GWAS outputs
    GWAS: Optional[pd.DataFrame] = None           # full results table
    significant: Optional[pd.DataFrame] = None    # significant SNPs only
    lambda_gc: float = 1.0                        # genomic inflation factor

    # GS/prediction outputs
    Pred: Optional[pd.DataFrame] = None           # prediction table per individual

    # Variance components
    h2: float = 0.0
    vg: float = 0.0
    ve: float = 0.0

    # QTNs identified (multi-locus methods)
    QTNs: Optional[np.ndarray] = None

    # Intermediate objects (useful for custom downstream analysis)
    kinship: Optional[np.ndarray] = None
    pca: Optional[object] = None
    taxa: Optional[np.ndarray] = None

    # Method used
    model: str = ""
    trait: str = ""
    runtime_seconds: float = 0.0


def GAPIT(
    # ── Input data ──────────────────────────────────────────────────────
    Y: Optional[Union[pd.DataFrame, str]] = None,       # phenotype
    G: Optional[Union[pd.DataFrame, str]] = None,       # HapMap genotype
    GD: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,  # numeric genotype
    GM: Optional[Union[pd.DataFrame, str]] = None,      # SNP map
    KI: Optional[Union[np.ndarray, pd.DataFrame]] = None,       # kinship
    CV: Optional[Union[pd.DataFrame, np.ndarray]] = None,       # covariates
    Z: Optional[np.ndarray] = None,                     # incidence matrix

    # ── Model selection ─────────────────────────────────────────────────
    model: Union[str, List[str]] = "BLINK",

    # ── PCA parameters ──────────────────────────────────────────────────
    PCA_total: int = 3,                       # R: PCA.total

    # ── QC parameters ───────────────────────────────────────────────────
    maf_threshold: float = 0.05,              # R: maf.threshold
    SNP_impute: str = "middle",               # R: SNP.impute

    # ── GWAS parameters ─────────────────────────────────────────────────
    cutOff: float = None,                     # significance threshold
    p_threshold: float = None,                # R: p.threshold (multi-locus)
    FDRcut: float = 0.05,                     # FDR q-value cutoff
    LD: float = 0.7,                          # LD threshold for BLINK

    # ── CMLM parameters ─────────────────────────────────────────────────
    group_from: int = 1,                      # R: group.from
    group_to: int = None,                     # R: group.to

    # ── FarmCPU/BLINK parameters ─────────────────────────────────────────
    bin_size: int = 5_000_000,                # R: bin.size (bp)
    maxLoop: int = 10,                        # max iterations

    # ── Genomic Selection ────────────────────────────────────────────────
    buspred: bool = False,                    # predict after GWAS
    prediction_model: str = "gBLUP",

    # ── Simulation parameters ────────────────────────────────────────────
    h2: float = None,                         # heritability for simulation
    NQTN: int = None,                         # number of QTNs for simulation

    # ── Output parameters ────────────────────────────────────────────────
    file_output: bool = True,                 # R: file.output
    output_dir: str = ".",                    # working directory for outputs
    Multiple_analysis: bool = False,          # R: Multiple_analysis

    # ── Kinship algorithm ────────────────────────────────────────────────
    kinship_algorithm: str = "VanRaden",      # R: kinship.algorithm

    # ── Trait to analyze (column name or index) ──────────────────────────
    trait: Union[str, int] = None,

) -> GBLUPResult:
    """
    GAPIT — Genome Association and Prediction Integrated Tool (Python)

    Complete Python reimplementation of R's GAPIT() function.
    Supports all GWAS models (GLM, MLM, CMLM, MLMM, FarmCPU, BLINK)
    and GS methods (gBLUP, cBLUP, sBLUP).

    Parameters match R GAPIT() with underscores instead of dots.

    Examples
    --------
    # Basic GWAS with BLINK (default):
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df, model="BLINK")

    # Multiple models:
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df,
                   model=["FarmCPU", "BLINK", "MLM"])

    # Genomic prediction:
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df,
                   model="gBLUP")

    # With user kinship:
    result = GAPIT(Y=pheno_df, GD=geno_df, GM=map_df,
                   KI=my_kinship, model="MLM")

    Returns
    -------
    GAPITResult with GWAS table, significant SNPs, Pred table,
    h2, vg, ve, kinship, pca, QTNs
    """
    t_start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # ── Normalise model list ─────────────────────────────────────────────
    if isinstance(model, str):
        models = [model]
    else:
        models = list(model)
    models = [m.upper() for m in models]

    # ── Load data ────────────────────────────────────────────────────────
    pheno, geno = _load_data(Y, G, GD, GM, SNP_impute)

    # ── Simulation mode ──────────────────────────────────────────────────
    if h2 is not None and NQTN is not None:
        print(f"[pyGAPIT] Simulation mode: h²={h2}, NQTN={NQTN}")
        pheno = _simulate_phenotype(pheno, geno, h2=h2, n_qtn=NQTN)

    # ── Select trait ─────────────────────────────────────────────────────
    trait_names = pheno.trait_names
    if trait is None:
        traits_to_run = trait_names
    elif isinstance(trait, int):
        traits_to_run = [trait_names[trait]]
    else:
        traits_to_run = [str(trait)]

    all_results = {}

    for trait_name in traits_to_run:
        print(f"\n[pyGAPIT] ──── Trait: {trait_name} ────")

        # ── Align taxa ───────────────────────────────────────────────────
        ki_df = _ki_to_df(KI, pheno.taxa) if KI is not None else None
        cv_df = _cv_to_df(CV, pheno.taxa) if CV is not None else None
        aligned = align_taxa(pheno, geno, cv_df=cv_df, ki_df=ki_df)

        taxa = aligned["taxa"]
        Y_aligned = aligned["Y"]
        GD_aligned = aligned["GD"]
        GM_aligned = aligned["GM"]
        KI_aligned = aligned.get("KI")
        CV_aligned = aligned.get("CV")

        # Extract phenotype vector for this trait
        y_col = Y_aligned[trait_name].values.astype(float)
        valid_mask = ~np.isnan(y_col)
        y = y_col[valid_mask]
        GD_y = GD_aligned[valid_mask, :]
        taxa_y = taxa[valid_mask]
        n = len(y)

        if n < 10:
            warnings.warn(f"Only {n} individuals with phenotype for {trait_name}. Skipping.")
            continue

        print(f"[pyGAPIT] n={n} individuals, m={GD_y.shape[1]} SNPs")

        # ── MAF filter ───────────────────────────────────────────────────
        GD_filtered, kept_snp_idx = maf_filter(GD_y, threshold=maf_threshold)
        GM_filtered = GM_aligned.iloc[kept_snp_idx].reset_index(drop=True)
        m_filtered = GD_filtered.shape[1]
        print(f"[pyGAPIT] After MAF filter (≥{maf_threshold}): {m_filtered} SNPs")

        # ── Kinship ──────────────────────────────────────────────────────
        if KI_aligned is not None:
            K = KI_aligned[np.ix_(valid_mask, valid_mask)]
            print("[pyGAPIT] Using provided kinship matrix")
        else:
            print("[pyGAPIT] Computing VanRaden kinship...")
            K = vanraden_kinship(GD_filtered)

        # ── PCA ──────────────────────────────────────────────────────────
        print(f"[pyGAPIT] Computing PCA (k={PCA_total})...")
        pca_result = compute_pca(GD_filtered, n_components=PCA_total)
        X0 = build_covariate_matrix(pca_result, PCA_total, CV_aligned[valid_mask] if CV_aligned is not None else None)

        # ── Extract SNP annotation ────────────────────────────────────────
        snp_names = GM_filtered["SNP"].values.astype(str)
        chromosomes = GM_filtered["Chromosome"].values
        positions = GM_filtered["Position"].values.astype(float)

        # ── Run requested models ─────────────────────────────────────────
        for model_name in models:
            print(f"[pyGAPIT] Running {model_name}...")
            t_model = time.time()

            result = _run_model(
                model_name=model_name,
                y=y, X0=X0, GD=GD_filtered, K=K,
                chromosomes=chromosomes, positions=positions,
                p_threshold=p_threshold,
                group_from=group_from, group_to=group_to,
                bin_size=bin_size, maxLoop=maxLoop,
                LD_threshold=LD,
            )

            elapsed = time.time() - t_model
            print(f"[pyGAPIT] {model_name} done in {elapsed:.1f}s | h²={result.get('h2',0):.3f}")

            # ── Build result table ────────────────────────────────────────
            maf_vals = _compute_maf(GD_filtered)
            gwas_df = _build_gwas_table(
                snp_names=snp_names,
                chromosomes=chromosomes,
                positions=positions,
                p_values=result["p_values"],
                effects=result["effects"],
                se=result["se"],
                maf=maf_vals,
                n_obs=n,
                adj_pvalues=benjamini_hochberg(result["p_values"]),
            )

            # ── Significance ──────────────────────────────────────────────
            threshold = cutOff if cutOff is not None else bonferroni_threshold(m_filtered)
            lam = genomic_inflation_factor(result["p_values"])
            sig_mask_final = gwas_df["P.value"] <= threshold
            sig_df = gwas_df[sig_mask_final].copy()

            print(f"[pyGAPIT] λ={lam:.3f} | {sig_mask_final.sum()} significant SNPs (Bonferroni threshold={threshold:.2e})")

            # ── Build prediction table ────────────────────────────────────
            pred_df = None
            if buspred or model_name in ("GBLUP", "CBLUP", "SBLUP"):
                qtns = result.get("selected_qtns")
                pred_df = _run_gs_and_build_pred(
                    y=y, X0=X0, GD=GD_filtered, K=K,
                    taxa=taxa_y, model_name=model_name,
                    qtn_indices=qtns,
                )

            # ── Save outputs ──────────────────────────────────────────────
            if file_output:
                _save_outputs(
                    gwas_df=gwas_df,
                    sig_df=sig_df,
                    pred_df=pred_df,
                    pca_result=pca_result,
                    K=K,
                    taxa=taxa_y,
                    snp_names=snp_names,
                    chromosomes=chromosomes,
                    positions=positions,
                    p_values=result["p_values"],
                    effects=result["effects"],
                    maf=maf_vals,
                    trait_name=trait_name,
                    model_name=model_name,
                    output_dir=output_dir,
                )

            all_results[f"{trait_name}_{model_name}"] = GAPITResult(
                GWAS=gwas_df,
                significant=sig_df if len(sig_df) > 0 else None,
                lambda_gc=lam,
                Pred=pred_df,
                h2=result.get("h2", 0.0),
                vg=result.get("vg", 0.0),
                ve=result.get("ve", 0.0),
                QTNs=result.get("selected_qtns"),
                kinship=K,
                pca=pca_result,
                taxa=taxa_y,
                model=model_name,
                trait=trait_name,
                runtime_seconds=time.time() - t_start,
            )

    # Return single result if only one, else dict
    if len(all_results) == 1:
        return list(all_results.values())[0]
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(Y, G, GD, GM, snp_impute) -> tuple[PhenotypeData, GenotypeData]:
    """Load and parse all input data."""
    # Phenotype
    if isinstance(Y, str):
        pheno = read_phenotype(Y)
    elif isinstance(Y, pd.DataFrame):
        pheno = PhenotypeData(
            Y=Y.copy(),
            taxa=Y.iloc[:, 0].astype(str).values,
            trait_names=Y.columns[1:].tolist(),
        )
    else:
        raise ValueError("Y must be a file path or pandas DataFrame")

    # Genotype
    if G is not None:
        if isinstance(G, str):
            geno = read_hapmap(G, impute_method=snp_impute)
        else:
            geno = read_hapmap(G, impute_method=snp_impute)  # assume df
    elif GD is not None and GM is not None:
        if isinstance(GD, str):
            geno = read_numeric(GD, GM, impute_method=snp_impute)
        else:
            if isinstance(GD, pd.DataFrame):
                taxa_gd = GD.iloc[:, 0].astype(str).values
                GD_vals = GD.iloc[:, 1:].values.astype(float)
            else:
                GD_vals = np.asarray(GD, dtype=float)
                taxa_gd = np.array([str(i) for i in range(GD_vals.shape[0])])

            if isinstance(GM, str):
                gm_df = pd.read_csv(GM, sep="\t")
            else:
                gm_df = GM.copy() if isinstance(GM, pd.DataFrame) else pd.DataFrame(GM)

            if gm_df.shape[1] >= 3:
                gm_df = gm_df.iloc[:, :3]
                gm_df.columns = ["SNP", "Chromosome", "Position"]

            geno = GenotypeData(GD=GD_vals, GM=gm_df, taxa=taxa_gd)
    else:
        raise ValueError("Provide either G (HapMap) or both GD and GM (numeric format)")

    return pheno, geno


def _simulate_phenotype(
    pheno: PhenotypeData,
    geno: GenotypeData,
    h2: float = 0.7,
    n_qtn: int = 20,
) -> PhenotypeData:
    """
    Simulate phenotype from genotype with given heritability.
    Translates GAPIT.Phenotype.Simulation.R
    """
    np.random.seed(198521)  # GAPIT's default seed
    n, m = geno.GD.shape

    # Random QTN indices
    qtn_idx = np.random.choice(m, size=min(n_qtn, m), replace=False)
    GD_qtn = geno.GD[:, qtn_idx]

    # Standardize QTN genotypes
    GD_std = (GD_qtn - GD_qtn.mean(axis=0)) / (GD_qtn.std(axis=0) + 1e-8)

    # Random effects
    effects = np.random.normal(0, 1, size=n_qtn)
    g = GD_std @ effects
    g_var = np.var(g)

    if g_var > 0:
        e_var = g_var * (1 - h2) / h2
    else:
        e_var = 1.0

    e = np.random.normal(0, np.sqrt(e_var), size=n)
    y_sim = g + e

    # Build new Y DataFrame using the genotype taxa (n individuals with geno data)
    geno_taxa = np.array([str(t) for t in geno.taxa])
    col_name = "SimTrait"
    Y_new = pd.DataFrame({"Taxa": geno_taxa, col_name: y_sim})

    return PhenotypeData(
        Y=Y_new,
        taxa=geno_taxa,
        trait_names=[col_name],
    )


def _ki_to_df(KI, taxa):
    """Convert kinship numpy array to DataFrame with taxa column."""
    if isinstance(KI, pd.DataFrame):
        return KI
    if isinstance(KI, np.ndarray):
        n = KI.shape[0]
        if len(taxa) == n:
            df = pd.DataFrame(KI, columns=taxa)
            df.insert(0, "Taxa", taxa)
            return df
    return None


def _cv_to_df(CV, taxa):
    """Convert CV numpy array to DataFrame with taxa column."""
    if isinstance(CV, pd.DataFrame):
        return CV
    if isinstance(CV, np.ndarray):
        n = CV.shape[0] if CV.ndim > 1 else len(CV)
        cv_df = pd.DataFrame(CV, columns=[f"CV{i+1}" for i in range(CV.shape[1] if CV.ndim > 1 else 1)])
        cv_df.insert(0, "Taxa", taxa[:n])
        return cv_df
    return None


def _run_model(
    model_name: str,
    y, X0, GD, K,
    chromosomes, positions,
    p_threshold, group_from, group_to,
    bin_size, maxLoop, LD_threshold,
) -> dict:
    """Dispatch to the correct GWAS/GS model."""
    m = GD.shape[1]
    p_thresh = p_threshold or (1.0 / m)

    if model_name == "GLM":
        r = glm_gwas(y, X0, GD)
        return {"p_values": r.p_values, "effects": r.effects, "se": r.se,
                "h2": 0.0, "vg": 0.0, "ve": 0.0}

    elif model_name == "MLM":
        r = mlm_gwas(y, X0, GD, K)
        return {"p_values": r.p_values, "effects": r.effects, "se": r.se,
                "h2": r.h2, "vg": r.vg, "ve": r.ve}

    elif model_name == "CMLM":
        n = len(y)
        r = cmlm_gwas(y, X0, GD, K,
                       group_from=group_from,
                       group_to=min(group_to or n, n))
        return {"p_values": r.p_values, "effects": r.effects, "se": r.se,
                "h2": r.h2, "vg": r.vg, "ve": r.ve}

    elif model_name == "MLMM":
        r = mlmm_gwas(y, X0, GD, K, p_threshold=p_thresh)
        return {"p_values": r.p_values, "effects": r.effects, "se": r.se,
                "h2": r.h2, "vg": r.vg, "ve": r.ve,
                "selected_qtns": r.selected_qtns}

    elif model_name in ("BLINK", "BLINK"):
        r = blink_gwas(y, X0, GD, max_iterations=maxLoop,
                        ld_threshold=LD_threshold, p_threshold=p_thresh)
        return {"p_values": r.p_values, "effects": r.effects, "se": r.se,
                "h2": 0.0, "vg": 0.0, "ve": 0.0,
                "selected_qtns": r.selected_qtns}

    elif model_name == "FARMCPU":
        r = farmcpu_gwas(
            y, X0, GD, chromosomes=chromosomes, positions=positions,
            max_iterations=maxLoop, bin_size=bin_size,
            p_threshold=p_thresh,
        )
        return {"p_values": r.p_values, "effects": r.effects, "se": r.se,
                "h2": r.h2, "vg": r.vg, "ve": r.ve,
                "selected_qtns": r.selected_qtns}

    elif model_name in ("GBLUP", "GBLUP"):
        r = gblup(y, X0, K)
        p_vals = np.ones(GD.shape[1])
        return {"p_values": p_vals, "effects": np.zeros(GD.shape[1]),
                "se": np.ones(GD.shape[1]),
                "h2": r.h2, "vg": r.vg, "ve": r.ve,
                "blup_result": r}

    elif model_name in ("CBLUP", "CBLUP"):
        r = cblup(y, X0, GD)
        p_vals = np.ones(GD.shape[1])
        return {"p_values": p_vals, "effects": np.zeros(GD.shape[1]),
                "se": np.ones(GD.shape[1]),
                "h2": r.h2, "vg": r.vg, "ve": r.ve,
                "blup_result": r}

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Choose from: GLM, MLM, CMLM, MLMM, BLINK, FarmCPU, gBLUP, cBLUP, sBLUP"
        )


def _compute_maf(GD: np.ndarray) -> np.ndarray:
    """Compute MAF for each SNP."""
    n = GD.shape[0]
    freq = np.nansum(GD, axis=0) / (2.0 * n)
    return np.minimum(freq, 1.0 - freq)


def _build_gwas_table(
    snp_names, chromosomes, positions, p_values,
    effects, se, maf, n_obs, adj_pvalues,
) -> pd.DataFrame:
    """Build standardized GWAS result DataFrame matching GAPIT's CSV output."""
    return pd.DataFrame({
        "SNP": snp_names,
        "Chr": chromosomes,
        "Pos": positions.astype(int),
        "P.value": p_values,
        "maf": np.round(maf, 4),
        "nobs": n_obs,
        "effect": np.round(effects, 6),
        "se": np.round(se, 6),
        "FDR.Adjusted.P.values": np.round(adj_pvalues, 6),
    }).sort_values(["Chr", "Pos"]).reset_index(drop=True)


def _run_gs_and_build_pred(
    y, X0, GD, K, taxa, model_name, qtn_indices=None,
) -> pd.DataFrame:
    """Run genomic prediction and build prediction DataFrame."""
    try:
        if qtn_indices is not None and len(qtn_indices) > 0 and model_name == "FARMCPU":
            gs_result = sblup(y, X0, GD, qtn_indices=qtn_indices, taxa=taxa)
        else:
            gs_result = gblup(y, X0, K, taxa=taxa)

        return pd.DataFrame({
            "Taxa": taxa,
            "BLUE": np.round(gs_result.blue, 4),
            "BLUP": np.round(gs_result.blup, 4),
            "PEV": np.round(gs_result.pev, 6),
            "gBreedingValue": np.round(gs_result.gebv, 4),
            "Prediction": np.round(gs_result.prediction, 4),
        })
    except Exception as e:
        warnings.warn(f"GS prediction failed: {e}")
        return None


def _save_outputs(
    gwas_df, sig_df, pred_df, pca_result, K, taxa,
    snp_names, chromosomes, positions, p_values, effects, maf,
    trait_name, model_name, output_dir,
):
    """Save all result files and plots. Translates GAPIT.ID.R output logic."""
    from .visualization.plots import (
        manhattan_plot, qq_plot, kinship_heatmap,
        pca_plot_2d, gs_scatter,
    )

    prefix = f"GAPIT.{model_name}.{trait_name}"
    out = Path(output_dir)

    # ── CSV outputs ────────────────────────────────────────────────────
    gwas_df.to_csv(out / f"{prefix}.GWAS.Results.csv", index=False)
    print(f"[pyGAPIT] Saved: {prefix}.GWAS.Results.csv")

    if pred_df is not None:
        pred_df.to_csv(out / f"{prefix}.Prediction.csv", index=False)

    # Kinship
    ki_df = pd.DataFrame(K, columns=taxa)
    ki_df.insert(0, "Taxa", taxa)
    ki_df.to_csv(out / "GAPIT.Kinship.csv", index=False)

    # PCA scores
    pca_df = pd.DataFrame(
        pca_result.scores,
        columns=[f"PC{i+1}" for i in range(pca_result.scores.shape[1])]
    )
    pca_df.insert(0, "Taxa", taxa)
    pca_df.to_csv(out / "GAPIT.PCA.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────
    try:
        # Manhattan
        sig_mask = gwas_df["P.value"] <= bonferroni_threshold(len(gwas_df))
        sig_indices = np.where(sig_mask.values)[0]

        fig_man = manhattan_plot(
            snp_names=gwas_df["SNP"].values,
            chromosomes=gwas_df["Chr"].values,
            positions=gwas_df["Pos"].values,
            p_values=gwas_df["P.value"].values,
            title=f"Manhattan: {trait_name} ({model_name})",
            highlight_snps=sig_indices if len(sig_indices) > 0 else None,
            save_path=str(out / f"{prefix}.Manhattan.pdf"),
        )
        plt.close(fig_man)

        # QQ
        fig_qq = qq_plot(
            p_values=gwas_df["P.value"].values,
            title=f"QQ: {trait_name} ({model_name})",
            save_path=str(out / f"{prefix}.QQ.pdf"),
        )
        plt.close(fig_qq)

        # Kinship heatmap
        fig_k = kinship_heatmap(
            K=K, taxa=taxa,
            save_path=str(out / "GAPIT.Kinship.pdf"),
        )
        plt.close(fig_k)

        # PCA 2D
        fig_pca = pca_plot_2d(
            scores=pca_result.scores,
            var_explained=pca_result.var_explained,
            title=f"PCA: {trait_name}",
            save_path=str(out / "GAPIT.PCA.pdf"),
        )
        plt.close(fig_pca)

    except Exception as e:
        warnings.warn(f"Plot generation failed: {e}")


# Import matplotlib at module level to allow plt.close calls above
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
