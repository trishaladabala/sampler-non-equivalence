"""
frontier_audit: Standalone diagnostic package for iso-frontier sensitivity and TOST gatekeeping.

Zero external dependencies beyond numpy and scipy.
"""

from .core import (
    IsoFrontierAudit,
    IsoCurveResult,
    JacobianTestResult,
)
from .tost import (
    TOSTResult,
    tost_paired,
    tost_welch,
    joint_tost,
    holm_bonferroni,
    hierarchical_gatekeeping,
    format_multiplicity_p,
)

__version__ = "0.1.0"
__all__ = [
    "IsoFrontierAudit",
    "IsoCurveResult",
    "JacobianTestResult",
    "TOSTResult",
    "tost_paired",
    "tost_welch",
    "joint_tost",
    "holm_bonferroni",
    "hierarchical_gatekeeping",
    "format_multiplicity_p",
]
