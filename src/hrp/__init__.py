"""
Hierarchical Risk Parity (HRP) Portfolio Optimization Module

This module implements HRP portfolio optimization using the Riskfolio-lib library.
HRP is a modern portfolio allocation method that uses hierarchical clustering
to build diversified portfolios without requiring matrix inversion.
"""

from .optimization_engines import (
    riskfolio_hrp,
    riskfolio_hrp_with_variants,
    equal_weights_baseline,
    portfolio_stats
)

__all__ = [
    'riskfolio_hrp',
    'riskfolio_hrp_with_variants',
    'equal_weights_baseline',
    'portfolio_stats'
]
