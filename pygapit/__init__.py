"""
pyGAPIT - Genome Association and Prediction Integrated Tool (Python)
A complete Python reimplementation of the R GAPIT package.
"""

__version__ = "2.0.0"
__author__  = "pyGAPIT contributors (based on GAPIT by Jiabo Wang & Zhiwu Zhang)"
__license__ = "GPL-3.0"

from .gapit import GAPIT, GAPITResult
from .gwas.glm     import glm_gwas,     GLMResult
from .gwas.mlm     import mlm_gwas,     cmlm_gwas,   MLMResult
from .gwas.mlmm    import mlmm_gwas,    MLMMResult
from .gwas.blink   import blink_gwas,   BLINKResult
from .gwas.farmcpu import farmcpu_gwas, FarmCPUResult
from .gs.blup      import gblup, cblup, sblup, predict_new, GBLUPResult
from .stats.kinship import vanraden_kinship, zhang_kinship
from .stats.pca     import compute_pca, build_covariate_matrix
from .stats.emma    import emma_remle, emmax_p3d
from .stats.testing import (
    bonferroni_threshold, benjamini_hochberg,
    genomic_inflation_factor, get_significant_snps,
)
from .io.formats import (
    read_hapmap, read_numeric, read_phenotype,
    align_taxa, maf_filter, GenotypeData, PhenotypeData,
)
from .visualization.plots import (
    manhattan_plot, qq_plot, kinship_heatmap,
    pca_plot_2d, pca_plot_3d_interactive,
    manhattan_interactive, gs_scatter, phenotype_distribution,
)

__all__ = [
    "GAPIT","GAPITResult",
    "glm_gwas","GLMResult","mlm_gwas","cmlm_gwas","MLMResult",
    "mlmm_gwas","MLMMResult","blink_gwas","BLINKResult",
    "farmcpu_gwas","FarmCPUResult",
    "gblup","cblup","sblup","predict_new","GBLUPResult",
    "vanraden_kinship","zhang_kinship",
    "compute_pca","build_covariate_matrix",
    "emma_remle","emmax_p3d",
    "bonferroni_threshold","benjamini_hochberg",
    "genomic_inflation_factor","get_significant_snps",
    "read_hapmap","read_numeric","read_phenotype",
    "align_taxa","maf_filter","GenotypeData","PhenotypeData",
    "manhattan_plot","qq_plot","kinship_heatmap",
    "pca_plot_2d","pca_plot_3d_interactive",
    "manhattan_interactive","gs_scatter","phenotype_distribution",
]
