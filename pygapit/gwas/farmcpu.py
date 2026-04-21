"""
PyGAPIT - FarmCPU
==================
Fixed and random model Circulating Probability Unification.
Liu et al. (2016) PLOS Genetics.

Matches GAPIT FarmCPU R source:
  Iteration loop:
    Step 1 (Fixed)  : GLM scan with current pseudo-QTNs as fixed covariates
    Step 2 (Random) : MLM scan to compute 'Prior' probability per SNP
                      (used for QTN selection in iterations 2+)
    Step 3 (BIN)    : LD-pruned QTN selection from Prior p-values
  Convergence       : Jaccard(new_set, prev_set) >= converge (default 1.0)
  Oscillation guard : if set at step N equals set at step N-2, take
                      consensus (intersection) and stop

Accepts kinship as `kinship=` or `K=`.

v1.0.2 changes vs v1.0.1:
  - Oscillation guard label corrected to use GAPIT Jaccard criterion
  - Convergence message includes Jaccard value
  - bin_size default changed from 10 to match GAPIT default bound=sqrt(n)/sqrt(log10(n))
"""

import numpy as np
import pandas as pd
from .glm import GLM
from .mlm import MLM
from ..kinship.kinship import VanRaden_kinship
from typing import Optional


def FarmCPU(phenotype: np.ndarray,
            genotype: np.ndarray,
            snp_info: pd.DataFrame,
            kinship: Optional[np.ndarray] = None,
            K: Optional[np.ndarray] = None,
            covariates: Optional[np.ndarray] = None,
            trait_name: str = "Trait",
            max_iterations: int = 10,
            bin_size: int = None,
            ld_threshold: float = 0.7,
            converge: float = 1.0) -> pd.DataFrame:
    """
    Run FarmCPU GWAS.

    Parameters
    ----------
    kinship       : (n,n) kinship matrix (computed if None). Also accepted as `K=`.
    max_iterations: int   — maximum EM-style iterations (GAPIT default: 10)
    bin_size      : int   — max pseudo-QTNs (GAPIT: floor(sqrt(n)/sqrt(log10(n))))
    ld_threshold  : float — r² LD-pruning threshold (GAPIT default: 0.7)
    converge      : float — Jaccard convergence threshold (1.0 = exact match)

    Returns
    -------
    pd.DataFrame  (final GLM results + 'Pseudo_QTN' boolean column)
    """
    if kinship is None and K is not None:
        kinship = K
    print(f"[PyGAPIT] Running FarmCPU GWAS for trait: {trait_name}")

    if kinship is None:
        kinship = VanRaden_kinship(genotype)

    n = phenotype.shape[0]
    # GAPIT default bound: round(sqrt(n) / sqrt(log10(n)))
    if bin_size is None:
        bin_size = max(1, int(round(np.sqrt(n) / np.sqrt(np.log10(max(n, 2))))))

    covar        = covariates.copy() if covariates is not None else None
    pseudo_qtns  = []       # list of integer column indices
    prev_sets    = []       # frozenset history for oscillation detection
    last_results = None

    for iteration in range(max_iterations):
        print(f"[PyGAPIT]  FarmCPU iter {iteration+1}, pseudo-QTNs: {len(pseudo_qtns)}")

        # ── Step 1 (Fixed): GLM with pseudo-QTNs as fixed covariates ─────────
        fixed_covar = covar
        if pseudo_qtns:
            qm          = genotype[:, pseudo_qtns].astype(float)
            fixed_covar = np.hstack([fixed_covar, qm]) if fixed_covar is not None else qm

        glm_res      = GLM(phenotype, genotype, snp_info,
                           covariates=fixed_covar, trait_name=trait_name)
        last_results = glm_res

        # ── Step 2 (Random): MLM for Prior probabilities ──────────────────────
        # GAPIT uses MLM p-values as prior for QTN selection when QTNs exist.
        # On iteration 1 (no QTNs yet), use GLM p-values directly.
        if pseudo_qtns:
            try:
                mlm_res = MLM(phenotype, genotype, snp_info,
                               kinship=kinship, covariates=covar,
                               trait_name=trait_name)
                prior_p = mlm_res["P.value"].values
            except Exception:
                prior_p = glm_res["P.value"].values
        else:
            prior_p = glm_res["P.value"].values

        # ── Step 3 (BIN): LD-pruned QTN selection from Prior ─────────────────
        new_qtns = _select_qtns_ld_pruned(prior_p, genotype, bin_size, ld_threshold)
        new_set  = frozenset(new_qtns)

        # ── Convergence: Jaccard similarity ───────────────────────────────────
        if prev_sets and prev_sets[-1]:
            union_   = len(new_set | prev_sets[-1])
            inter_   = len(new_set & prev_sets[-1])
            jaccard  = inter_ / union_ if union_ > 0 else 1.0
            if jaccard >= converge:
                print(f"[PyGAPIT]  FarmCPU converged (Jaccard={jaccard:.3f}).")
                break

        # ── Oscillation: step N == step N-2 ───────────────────────────────────
        if len(prev_sets) >= 2 and new_set == prev_sets[-2]:
            print("[PyGAPIT]  FarmCPU oscillation detected — taking consensus QTNs.")
            consensus   = sorted(new_set & prev_sets[-1])
            pseudo_qtns = consensus if consensus else sorted(prev_sets[-1])
            break

        prev_sets.append(new_set)
        pseudo_qtns = new_qtns

    if last_results is None:
        last_results = GLM(phenotype, genotype, snp_info,
                           covariates=covar, trait_name=trait_name)

    last_results["Pseudo_QTN"] = last_results.index.isin(pseudo_qtns)
    print(f"[PyGAPIT]  -> FarmCPU done. Pseudo-QTNs: {len(pseudo_qtns)}")
    return last_results


# ─────────────────────────────────────────────────────────────────────────────
def _select_qtns_ld_pruned(p_values: np.ndarray,
                            genotype: np.ndarray,
                            bin_size: int,
                            ld_threshold: float) -> list:
    """
    Select pseudo-QTNs matching GAPIT FarmCPU.BIN / Blink.LDRemove:
      1. Rank SNPs by p-value (ascending = most significant first).
      2. Greedily keep a SNP only if r² < ld_threshold with all already-kept.
      3. Cap at bin_size.

    Invariant SNPs (std < 1e-12) are silently skipped.
    """
    order     = np.argsort(p_values)
    kept_idx  = []
    kept_cols = []

    for idx in order:
        if len(kept_idx) >= bin_size:
            break
        col = genotype[:, idx].astype(float)
        if np.std(col) < 1e-12:
            continue
        in_ld = any(
            np.std(prev) > 1e-12
            and float(np.corrcoef(col, prev)[0, 1] ** 2) > ld_threshold
            for prev in kept_cols
        )
        if not in_ld:
            kept_idx.append(int(idx))
            kept_cols.append(col)

    return kept_idx
