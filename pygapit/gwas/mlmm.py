"""
PyGAPIT - Multi-Locus Mixed Model (MLMM)
=========================================
Forward stepwise MLM using Bonferroni threshold to control entry.
Matches GAPIT MLMM / Segura et al. (2012) Nature Genetics.

Accepts kinship as `kinship=` or `K=`.
Raises ValueError if no kinship provided.

v1.0.2 changes:
  - Default p_threshold now uses Bonferroni(M) rather than hardcoded 5e-6
  - Kinship alias K= supported
"""

import numpy as np
import pandas as pd
from .mlm import MLM
from typing import Optional


def MLMM(phenotype: np.ndarray,
         genotype: np.ndarray,
         snp_info: pd.DataFrame,
         kinship: np.ndarray = None,
         covariates: Optional[np.ndarray] = None,
         trait_name: str = "Trait",
         max_cofactors: int = 10,
         p_threshold: float = None,
         K: np.ndarray = None) -> pd.DataFrame:
    """
    Multi-Locus Mixed Model (MLMM) GWAS — forward stepwise cofactor selection.

    Parameters
    ----------
    kinship      : (n,n) kinship matrix — also accepted as `K=`
    max_cofactors: int   maximum cofactors to add
    p_threshold  : float entry threshold (default = Bonferroni 0.05/M)

    Returns
    -------
    pd.DataFrame — final MLM results with 'Cofactor' boolean column
    """
    if kinship is None and K is not None:
        kinship = K
    if kinship is None:
        raise ValueError(
            "A kinship matrix must be provided via `kinship=` or `K=`.")

    M = genotype.shape[1]
    if p_threshold is None:
        p_threshold = 0.05 / M    # Bonferroni

    print(f"[PyGAPIT] Running MLMM GWAS for trait: {trait_name}")
    cofactor_idx = []
    covar        = covariates.copy() if covariates is not None else None

    for step in range(max_cofactors):
        print(f"[PyGAPIT]  MLMM step {step+1}: {len(cofactor_idx)} cofactor(s)")
        results = MLM(phenotype, genotype, snp_info, kinship=kinship,
                      covariates=covar, trait_name=trait_name)

        # Exclude already-selected cofactors
        candidates = results.drop(index=cofactor_idx, errors="ignore")
        best_idx   = candidates["P.value"].idxmin()
        best_p     = float(candidates.loc[best_idx, "P.value"])

        if best_p > p_threshold:
            print(f"[PyGAPIT]  MLMM stopping: best p={best_p:.2e} > threshold={p_threshold:.2e}")
            break

        cofactor_idx.append(best_idx)
        new_snp = genotype[:, best_idx].reshape(-1, 1).astype(float)
        covar   = np.hstack([covar, new_snp]) if covar is not None else new_snp
        print(f"[PyGAPIT]  Added cofactor: {snp_info.iloc[best_idx]['SNP']} "
              f"(p={best_p:.2e})")

    results["Cofactor"] = results.index.isin(cofactor_idx)
    print(f"[PyGAPIT]  -> MLMM done with {len(cofactor_idx)} cofactor(s).")
    return results
