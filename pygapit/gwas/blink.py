"""
PyGAPIT - BLINK
================
Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway.
Huang et al. (2018) Genome Biology.

Matches GAPIT Blink R source exactly:
  - Iteration: GLM with current pseudo-QTNs as fixed covariates
  - LD removal: greedy r² pruning of significant SNPs (threshold 0.7)
  - BIC selection:
        ve   = var(yp - y) = RSS / (n-1)   [R's var() uses n-1 denominator]
        n2LL = n*log(2π) + n*log(ve) + n
        BIC  = n2LL + (k-1)*log(n)         k = number of model parameters
  - Convergence: Jaccard similarity between consecutive QTN sets >= converge
  - Oscillation guard: same as FarmCPU (step N == step N-2 → consensus)

v1.0.2 changes:
  - Added BLINK-specific oscillation guard matching the FarmCPU pattern
  - threshold (max QTNs) uses GAPIT formula: floor(n / log(n))
  - Bonferroni threshold for candidate screening uses 0.05/M
"""

import numpy as np
import pandas as pd
from scipy import stats
from .glm import GLM
from typing import Optional


def BLINK(phenotype: np.ndarray,
          genotype: np.ndarray,
          snp_info: pd.DataFrame,
          covariates: Optional[np.ndarray] = None,
          trait_name: str = "Trait",
          max_snps: int = None,
          ld_threshold: float = 0.7,
          p_threshold: float = 0.05,
          converge: float = 1.0,
          max_iterations: int = 10) -> pd.DataFrame:
    """
    Run BLINK GWAS.

    Parameters
    ----------
    max_snps      : int   — max loci (default: floor(n/log(n)), GAPIT default)
    ld_threshold  : float — r² LD-pruning threshold (GAPIT default: 0.7)
    p_threshold   : float — significance level for Bonferroni cutoff
    converge      : float — Jaccard convergence threshold (1.0 = exact match)
    max_iterations: int   — safety cap

    Returns
    -------
    pd.DataFrame  columns = SNP, Chromosome, Position, Effect, SE, P.value, Selected
    """
    print(f"[PyGAPIT] Running BLINK GWAS for trait: {trait_name}")
    n, m      = genotype.shape
    bonferroni = p_threshold / m
    # GAPIT: threshold = floor(ny / log(ny))
    threshold  = int(np.floor(n / np.log(max(n, 2)))) if max_snps is None else max_snps

    covar        = covariates.copy() if covariates is not None else None
    selected_idx = []
    prev_sets    = []
    last_results = None
    is_done      = False

    for iteration in range(max_iterations):
        print(f"[PyGAPIT]  BLINK iteration {iteration+1}, pseudo-QTNs: {len(selected_idx)}")

        # ── Step 1: GLM with current pseudo-QTNs ─────────────────────────────
        fixed_covar = covar
        if selected_idx:
            qm          = genotype[:, selected_idx].astype(float)
            fixed_covar = np.hstack([fixed_covar, qm]) if fixed_covar is not None else qm

        results      = GLM(phenotype, genotype, snp_info,
                           covariates=fixed_covar, trait_name=trait_name)
        last_results = results

        # ── Step 2: LD remove — significant SNPs only ─────────────────────────
        sig_mask = results["P.value"] < bonferroni
        if not sig_mask.any():
            print(f"[PyGAPIT]  BLINK: no SNPs pass Bonferroni ({bonferroni:.2e}), stopping.")
            break

        candidates = results[sig_mask].sort_values("P.value")
        new_idx    = _ld_remove(candidates, genotype, threshold, ld_threshold)

        # ── Step 3: BIC model selection ───────────────────────────────────────
        if len(new_idx) > 1:
            new_idx = _bic_select(phenotype, genotype, covar, new_idx)

        new_set = frozenset(new_idx)

        # ── Convergence: Jaccard ──────────────────────────────────────────────
        if prev_sets and prev_sets[-1]:
            union_  = len(new_set | prev_sets[-1])
            inter_  = len(new_set & prev_sets[-1])
            jaccard = inter_ / union_ if union_ > 0 else 1.0
            if jaccard >= converge:
                print(f"[PyGAPIT]  BLINK converged (Jaccard={jaccard:.3f}).")
                is_done = True

        # ── Oscillation guard: step N == step N-2 ────────────────────────────
        if len(prev_sets) >= 2 and new_set == prev_sets[-2]:
            print("[PyGAPIT]  BLINK oscillation detected — taking consensus QTNs.")
            consensus    = sorted(new_set & prev_sets[-1])
            selected_idx = consensus if consensus else sorted(prev_sets[-1])
            break

        prev_sets.append(new_set)
        selected_idx = new_idx

        # Update covariate for next iteration
        if selected_idx:
            qm    = genotype[:, selected_idx].astype(float)
            covar = np.hstack([covariates, qm]) if covariates is not None else qm
        else:
            covar = covariates

        snames = [snp_info.iloc[i]["SNP"] for i in selected_idx[:5]]
        print(f"[PyGAPIT]  Selected QTNs: {snames}")

        if is_done:
            break

    if last_results is None:
        last_results = GLM(phenotype, genotype, snp_info,
                           covariates=covar, trait_name=trait_name)

    last_results["Selected"] = last_results.index.isin(selected_idx)
    print(f"[PyGAPIT]  -> BLINK done. Selected loci: {len(selected_idx)}")
    return last_results


# ─────────────────────────────────────────────────────────────────────────────
def _ld_remove(candidates: pd.DataFrame,
               genotype: np.ndarray,
               max_snps: int,
               ld_threshold: float) -> list:
    """
    Greedy LD pruning matching GAPIT Blink.LDRemove.
    Candidates already sorted by p-value ascending (best first).
    Keep a SNP only if r² < ld_threshold with ALL already-kept SNPs.
    """
    kept_idx  = []
    kept_cols = []
    for row_idx in candidates.index:
        if len(kept_idx) >= max_snps:
            break
        col = genotype[:, row_idx].astype(float)
        if np.std(col) < 1e-12:
            continue
        in_ld = any(
            np.std(prev) > 1e-12 and
            float(np.corrcoef(col, prev)[0, 1] ** 2) > ld_threshold
            for prev in kept_cols
        )
        if not in_ld:
            kept_idx.append(row_idx)
            kept_cols.append(col)
    return kept_idx


def _bic_select(y: np.ndarray,
                G: np.ndarray,
                covar: Optional[np.ndarray],
                snp_indices: list) -> list:
    """
    Forward BIC model selection matching GAPIT Blink.BICselection.

    GAPIT BIC formula (exact):
        ve   = var(yp - y) = sum((yp-y)²) / (n-1)   [R var() uses n-1]
        n2LL = n*log(2π) + n*log(ve) + n
        BIC  = n2LL + (k-1)*log(n)     k = number of model parameters

    Steps through 1..len(snp_indices) and keeps the model with minimum BIC.
    """
    mask = ~np.isnan(y)
    ym   = y[mask].astype(float)
    nm   = len(ym)

    W = np.ones((nm, 1))
    if covar is not None:
        W = np.hstack([W, np.asarray(covar)[mask].astype(float)])

    best_bic = np.inf
    best_k   = 0
    W_cur    = W.copy()

    for k, idx in enumerate(snp_indices, 1):
        col  = G[mask, idx].reshape(-1, 1).astype(float)
        W_cur = np.hstack([W_cur, col])
        beta, _, _, _ = np.linalg.lstsq(W_cur, ym, rcond=None)
        yp   = W_cur @ beta
        # GAPIT: ve = var(yp - y) with n-1 denominator (R's var())
        ve   = float(np.sum((yp - ym) ** 2) / (nm - 1))
        if ve <= 0:
            ve = 1e-300
        npar = W_cur.shape[1]
        n2LL = nm * np.log(2 * np.pi) + nm * np.log(ve) + nm
        bic  = n2LL + (npar - 1) * np.log(nm)
        if bic < best_bic:
            best_bic = bic
            best_k   = k

    return snp_indices[:best_k] if best_k > 0 else snp_indices[:1]


def _compute_r2(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return 0.0
    r, _ = stats.pearsonr(x[mask], y[mask])
    return r ** 2
