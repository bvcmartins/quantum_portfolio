import numpy as np
import pandas as pd
import riskfolio as rp
import logging
from sklearn.covariance import LedoitWolf

seed = 12
np.random.seed(seed)
logger = logging.getLogger("hrp_logger")


def portfolio_stats(weights, data):
    """Calculate portfolio statistics (return, volatility)"""
    weights = np.array(weights)
    returns = np.log(data) - np.log(data.shift(1))  # log return to minimize fp error
    port_return = np.sum(returns.mean() * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    return port_return, port_vol


def equal_weights_baseline(data):
    """
    Equal weights baseline portfolio.

    Args:
        data: Stock price DataFrame

    Returns:
        weights: Equal weight allocation
    """
    n_assets = len(data.columns)
    return np.ones(n_assets) / n_assets


def riskfolio_hrp(data, codependence='pearson', linkage='ward', max_k=10, leaf_order=True):
    """
    Hierarchical Risk Parity (HRP) portfolio optimization using Riskfolio-lib.

    HRP is a modern portfolio allocation method that uses hierarchical clustering
    to build diversified portfolios without requiring the inversion of the
    covariance matrix. This makes it more robust than traditional mean-variance
    optimization, especially for large numbers of assets.

    The HRP algorithm works in three steps:
    1. Tree Clustering: Reorganize the covariance matrix based on hierarchical clustering
    2. Quasi-Diagonalization: Reorder the covariance matrix to group similar assets
    3. Recursive Bisection: Allocate weights by recursively splitting the dendrogram

    Args:
        data: Stock price DataFrame
        codependence: Method to calculate codependence matrix
            - 'pearson': Pearson correlation (default)
            - 'spearman': Spearman correlation
            - 'kendall': Kendall tau correlation
            - 'gerber1': Gerber statistic 1
            - 'gerber2': Gerber statistic 2
        linkage: Linkage method for hierarchical clustering
            - 'ward': Ward variance minimization (default)
            - 'single': Single linkage (minimum distance)
            - 'complete': Complete linkage (maximum distance)
            - 'average': Average linkage (UPGMA)
            - 'weighted': Weighted average linkage (WPGMA)
            - 'centroid': Centroid linkage (UPGMC)
            - 'median': Median linkage (WPGMC)
        max_k: Maximum number of clusters for gap statistic (default 10)
        leaf_order: If True, optimize dendrogram leaf order for better visualization (default True)

    Returns:
        weights: HRP optimized portfolio weights as numpy array

    Raises:
        Exception: If Riskfolio HRP optimization fails
    """
    try:
        # Calculate returns using log returns
        returns = np.log(data) - np.log(data.shift(1))
        returns = returns.dropna()

        # Ensure returns is a proper DataFrame with float64 dtype
        returns = returns.astype(np.float64)

        logger.debug(f"Returns shape: {returns.shape}, data shape: {data.shape}, returns dtype: {returns.dtypes.unique()}")

        # Create HCPortfolio object for hierarchical clustering methods (HRP, HERC, NCO)
        # HCPortfolio is the correct class for HRP optimization
        port = rp.HCPortfolio(returns=returns)

        # Run HRP optimization with codependence and linkage parameters
        # HCPortfolio.optimization() accepts these parameters directly
        logger.debug(f"Running HRP optimization with codependence={codependence}, linkage={linkage}...")
        weights_hrp = port.optimization(
            model='HRP',           # Hierarchical Risk Parity
            codependence=codependence,
            linkage=linkage,
            max_k=max_k,
            leaf_order=leaf_order
        )

        if weights_hrp is None or len(weights_hrp) == 0:
            raise ValueError("Riskfolio HRP optimization returned None or empty weights")

        # Debug: Check what riskfolio returned
        logger.debug(f"Riskfolio HRP returned type: {type(weights_hrp)}")
        logger.debug(f"Riskfolio HRP returned shape: {weights_hrp.shape}")

        # Extract values from DataFrame and flatten to 1D array
        # Ensure we get a proper 1D numpy array of floats
        if isinstance(weights_hrp, pd.DataFrame):
            # If it's a DataFrame, extract the first column's values
            weights = weights_hrp.iloc[:, 0].values.astype(np.float64)
        elif isinstance(weights_hrp, pd.Series):
            # If it's a Series, extract values directly
            weights = weights_hrp.values.astype(np.float64)
        else:
            # Otherwise try to convert to array
            weights = np.asarray(weights_hrp, dtype=np.float64).flatten()

        logger.debug(f"Extracted weights shape: {weights.shape}, dtype: {weights.dtype}")

        # Verify weights sum to 1 (HRP should already ensure this)
        weights_sum = weights.sum()
        if not np.isclose(weights_sum, 1.0):
            logger.warning(f"Weights sum to {weights_sum:.6f}, normalizing to 1.0")
            weights = weights / weights_sum

        # Calculate portfolio statistics
        port_return, port_vol = portfolio_stats(weights, data)
        logger.info(f'Riskfolio HRP optimization completed successfully.')
        logger.info(f'Portfolio return: {port_return:.6f}, volatility: {port_vol:.6f}')
        logger.debug(f'Weight statistics: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}')
        logger.debug(f'Number of assets with >1% weight: {(weights > 0.01).sum()}/{len(weights)}')

        return weights

    except Exception as e:
        logger.error(f"Riskfolio HRP optimization error: {e}", exc_info=True)
        raise


def riskfolio_hrp_with_variants(data):
    """
    Run HRP optimization with multiple parameter variants and return all results.

    This function tests different combinations of codependence measures and
    linkage methods to explore the sensitivity of HRP to these parameters.

    Args:
        data: Stock price DataFrame

    Returns:
        results: Dictionary with variant names as keys and weights as values
    """
    results = {}

    # Test different codependence and linkage combinations
    variants = [
        ('pearson', 'ward', 'HRP_Pearson_Ward'),
        ('pearson', 'single', 'HRP_Pearson_Single'),
        ('spearman', 'ward', 'HRP_Spearman_Ward'),
        ('spearman', 'single', 'HRP_Spearman_Single'),
    ]

    for codependence, linkage, name in variants:
        try:
            logger.info(f"Running {name}...")
            weights = riskfolio_hrp(data, codependence=codependence, linkage=linkage)
            results[name] = weights
            logger.info(f"{name} completed successfully")
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            results[name] = None

    return results
