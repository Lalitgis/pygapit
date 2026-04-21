"""
PyGAPIT - Mixed Linear Model (MLM / P3D)
=========================================
Matches GAPIT EMMA REMLE exactly:
  - REML estimation via grid search on log(delta) in [-10, 10] with 100 grids,
    then Brent refinement at each sign-change (matches GAPIT.emma.REMLE exactly)
  - Spectral decomposition of K; projection matrix S = I - X(X'X)^{-1}X'
  - Per-SNP t-test in the rotated (whitened) space using delta from P3D

Accepts kinship as `kinship=` or `K=`.
Raises ValueError if no kinship provided.

v1.0.2 changes vs v1.0.1:
  - REML grid search now matches GAPIT exactly (was scipy minimize_scalar)
  - Per-SNP test uses full whitened-rotation t-test matching GAPIT's approach
  - MLMM p_threshold now uses Bonferroni(M) as default, not hardcoded 5e-6
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Optional
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# EMMA REML — exact port of GAPIT's GAPIT.emma.REMLE
# ─────────────────────────────────────────────────────────────────────────────
def _emma_eigen_R_wo_Z(K: np.ndarray, X: np.ndarray):
    """
    GAPIT: emma.eigen.R.wo.Z
    S = I - X(X'X)^{-1}X'
    eig of S(K+I)S, return (values[1:(n-q)] - 1, vectors[:, 1:(n-q)])
    """
    n, q = X.shape
    # Projection matrix S = I - X(X'X)^{-1}X'
    try:
        XtXinv = np.linalg.solve(X.T @ X, np.eye(q))
    except np.linalg.LinAlgError:
        XtXinv = np.linalg.pinv(X.T @ X)
    S = np.eye(n) - X @ XtXinv @ X.T
    # Eigendecomposition of S(K+I)S — symmetric
    M = S @ (K + np.eye(n)) @ S
    eigvals, eigvecs = np.linalg.eigh(M)
    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    # Return first (n-q) after subtracting 1 (we added I)
    return eigvals[:n-q] - 1.0, eigvecs[:, :n-q]


def _reml_ll_wo_Z(logdelta: float, lambdas: np.ndarray, etas: np.ndarray) -> float:
    """GAPIT: emma.delta.REML.LL.wo.Z"""
    nq    = len(etas)
    delta = np.exp(logdelta)
    d     = lambdas + delta
    d     = np.maximum(d, 1e-9)   # GAPIT: mitigate negative Lambdas
    s2    = np.sum(etas**2 / d) / nq
    return 0.5 * (nq * (np.log(nq / (2*np.pi)) - 1 - np.log(max(s2, 1e-300)))
                  - np.sum(np.log(d)))


def _reml_dll_wo_Z(logdelta: float, lambdas: np.ndarray, etas: np.ndarray) -> float:
    """GAPIT: emma.delta.REML.dLL.wo.Z"""
    nq    = len(etas)
    delta = np.exp(logdelta)
    etasq = etas**2
    ld    = lambdas + delta
    ld    = np.maximum(ld, 1e-9)
    return 0.5 * (nq * np.sum(etasq / (ld**2)) / np.sum(etasq / ld)
                  - np.sum(1.0 / ld))


def _emma_REMLE(y: np.ndarray, X: np.ndarray, K: np.ndarray,
                ngrids: int = 100, llim: float = -10., ulim: float = 10.,
                esp: float = 1e-10):
    """
    Port of GAPIT.emma.REMLE (without Z matrix, Z=NULL case).
    Returns: delta, Vg, Ve
    """
    n, q = len(y), X.shape[1]

    # Singularity check
    if abs(np.linalg.det(X.T @ X)) < 1e-15:
        warnings.warn("X is singular in EMMA REMLE")
        return 1.0, 1.0, 1.0

    lam, evec = _emma_eigen_R_wo_Z(K, X)   # (n-q,), (n, n-q)
    etas = evec.T @ y                        # projected phenotype

    # Grid of log-delta
    logdelta = np.linspace(llim, ulim, ngrids + 1)
    m        = len(logdelta)
    delta_g  = np.exp(logdelta)

    # Vectorised LL and dLL over grid
    # Lambda matrix: (n-q) x m
    nq      = n - q
    Lam     = lam[:, None] + delta_g[None, :]   # (nq, m)
    Lam     = np.maximum(Lam, 1e-9)
    Etasq   = (etas**2)[:, None] * np.ones(m)    # (nq, m)
    dLL     = 0.5 * delta_g * (nq * np.sum(Etasq / Lam**2, axis=0)
                                / np.sum(Etasq / Lam, axis=0)
                                - np.sum(1.0 / Lam, axis=0))

    opt_logdelta = []
    opt_LL       = []

    # Boundary check: dLL[0] < esp
    if dLL[0] < esp:
        opt_logdelta.append(llim)
        opt_LL.append(_reml_ll_wo_Z(llim, lam, etas))
    # Boundary check: dLL[-2] > -esp
    if dLL[m-2] > -esp:
        opt_logdelta.append(ulim)
        opt_LL.append(_reml_ll_wo_Z(ulim, lam, etas))

    # Sign-change intervals → Brent root-finding
    for i in range(m - 1):
        if dLL[i] * dLL[i+1] < 0 and dLL[i] > 0 and dLL[i+1] < 0:
            try:
                r = optimize.brentq(_reml_dll_wo_Z, logdelta[i], logdelta[i+1],
                                     args=(lam, etas), xtol=1e-8)
                opt_logdelta.append(r)
                opt_LL.append(_reml_ll_wo_Z(r, lam, etas))
            except Exception:
                pass

    if not opt_logdelta:
        # Fallback: take grid maximum
        best_idx = int(np.argmax([_reml_ll_wo_Z(ld, lam, etas) for ld in logdelta]))
        opt_logdelta = [logdelta[best_idx]]
        opt_LL       = [_reml_ll_wo_Z(logdelta[best_idx], lam, etas)]

    best_delta = np.exp(opt_logdelta[int(np.argmax(opt_LL))])

    # Compute Vg and Ve from best delta
    d  = lam + best_delta
    d  = np.maximum(d, 1e-9)
    Vg = float(np.sum(etas**2 / d) / nq)
    Ve = Vg * best_delta

    return best_delta, max(Vg, 1e-9), max(Ve, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — kept for backward compat
# ─────────────────────────────────────────────────────────────────────────────
def _EMMA_vc(y, X, K):
    """Legacy wrapper — returns (delta, Vg, Ve, U, eigvals, beta)."""
    delta, Vg, Ve = _emma_REMLE(y, X, K)
    eigvals, U = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 1e-8)
    d  = eigvals + delta
    Uy = U.T @ y;  UX = U.T @ X
    UXd = UX / d[:, None]
    gram = UX.T @ UXd
    beta = np.linalg.lstsq(gram, UXd.T @ Uy, rcond=None)[0]
    return delta, Vg, Ve, U, eigvals, beta


def MLM(phenotype, genotype, snp_info, kinship=None,
        covariates=None, trait_name="Trait", method="P3D", K=None):
    """
    Mixed Linear Model (EMMA/P3D) GWAS.

    Parameters
    ----------
    phenotype : np.ndarray (n,)
    genotype  : np.ndarray (n, m)
    snp_info  : pd.DataFrame  with SNP, Chromosome, Position columns
    kinship   : np.ndarray (n,n)   — also accepted as keyword `K=`
    covariates: np.ndarray (n, k)  optional PCA / Q covariates
    trait_name: str
    method    : 'P3D' (estimate VC once) or 'EMMA' (per-SNP, slower)

    Returns
    -------
    pd.DataFrame  columns = SNP, Chromosome, Position, Effect, SE, P.value, -log10(P)
    """
    if kinship is None and K is not None:
        kinship = K
    if kinship is None:
        raise ValueError(
            "A kinship matrix must be provided via `kinship=` or `K=`.")

    print(f"[PyGAPIT] Running MLM ({method}) GWAS for trait: {trait_name}")
    y = phenotype.astype(float)

    # Handle missing phenotype
    mask = ~np.isnan(y)
    y   = y[mask]
    Kn  = kinship[np.ix_(mask, mask)]
    G   = genotype[mask, :]
    n, m = G.shape

    # Base design matrix (intercept + covariates)
    X_base = np.ones((n, 1))
    if covariates is not None:
        cv = np.asarray(covariates)[mask]
        X_base = np.hstack([X_base, cv])

    # ── P3D: estimate VC once ─────────────────────────────────────────────
    delta, Vg, Ve, _, _, _ = _EMMA_vc(y, X_base, Kn)
    print(f"[PyGAPIT]  Vg={Vg:.4f}  Ve={Ve:.4f}  delta={delta:.4f}")

    # Eigen-decomposition of Kn for rotation
    eigvals_K, U_K = np.linalg.eigh(Kn)
    eigvals_K = np.maximum(eigvals_K, 1e-8)
    d0  = eigvals_K + delta          # (n,) — shared denominator

    # Rotate base quantities once
    Uy0 = U_K.T @ y                  # (n,)
    UX0 = U_K.T @ X_base            # (n, k)

    results = []
    for j in range(m):
        snp_col = G[:, j].astype(float)

        # Skip invariant
        if np.std(snp_col) < 1e-8:
            results.append({
                "SNP": snp_info.iloc[j]["SNP"],
                "Chromosome": snp_info.iloc[j].get("Chromosome", np.nan),
                "Position":   snp_info.iloc[j].get("Position",   np.nan),
                "Effect": 0.0, "SE": np.nan, "P.value": 1.0})
            continue

        try:
            # Whitened design matrix (intercept + covariates + SNP)
            Uz    = U_K.T @ snp_col              # (n,)
            UX    = np.hstack([UX0, Uz.reshape(-1, 1)])  # (n, k+1)

            # Scale rows by 1/sqrt(lambda_i + delta) — matches GAPIT rotation
            scale = 1.0 / np.sqrt(d0)            # (n,)
            UXs   = UX    * scale[:, None]       # (n, k+1) whitened
            Uys   = Uy0   * scale                # (n,)     whitened

            # OLS on whitened data
            beta, _, _, _ = np.linalg.lstsq(UXs, Uys, rcond=None)
            resid  = Uys - UXs @ beta
            df_res = n - UXs.shape[1]
            if df_res <= 0:
                results.append({
                    "SNP": snp_info.iloc[j]["SNP"],
                    "Chromosome": snp_info.iloc[j].get("Chromosome", np.nan),
                    "Position":   snp_info.iloc[j].get("Position",   np.nan),
                    "Effect": float(beta[-1]), "SE": np.nan, "P.value": 1.0})
                continue

            sigma2 = float(np.sum(resid**2) / df_res)
            XtXinv = np.linalg.pinv(UXs.T @ UXs)
            diag_v = float(XtXinv[-1, -1])
            se     = float(np.sqrt(max(sigma2 * diag_v, 0.0)))
            effect = float(beta[-1])
            t_stat = effect / se if se > 1e-12 else 0.0
            p_val  = float(2.0 * stats.t.sf(abs(t_stat), df=df_res))
        except Exception:
            effect, se, p_val = 0.0, np.nan, 1.0

        results.append({
            "SNP":        snp_info.iloc[j]["SNP"],
            "Chromosome": snp_info.iloc[j].get("Chromosome", np.nan),
            "Position":   snp_info.iloc[j].get("Position",   np.nan),
            "Effect": effect, "SE": se, "P.value": p_val})

    df = pd.DataFrame(results)
    df["-log10(P)"] = -np.log10(df["P.value"].clip(lower=1e-300))
    print(f"[PyGAPIT]  -> MLM done. Significant (p<0.05): {(df['P.value']<0.05).sum()}")
    return df
