# Hierarchical Risk Parity (HRP) Portfolio Optimization

This folder contains the implementation of Hierarchical Risk Parity (HRP) portfolio optimization using the Riskfolio-lib library.

## Overview

Hierarchical Risk Parity (HRP) is a modern portfolio allocation method introduced by Marcos López de Prado. It uses hierarchical clustering to build diversified portfolios without requiring the inversion of the covariance matrix, making it more robust than traditional mean-variance optimization.

### Key Advantages of HRP

1. **No Matrix Inversion**: HRP doesn't require inverting the covariance matrix, making it more stable and robust to estimation errors
2. **Better Diversification**: Uses hierarchical clustering to group similar assets and allocate weights based on risk
3. **Robust to Noise**: Less sensitive to estimation errors in the covariance matrix
4. **No Expected Returns Required**: Only requires covariance/correlation estimates, not expected returns

### How HRP Works

The HRP algorithm consists of three main steps:

1. **Tree Clustering**: Reorganize the covariance matrix based on hierarchical clustering of assets
2. **Quasi-Diagonalization**: Reorder the covariance matrix to group similar assets together
3. **Recursive Bisection**: Allocate weights by recursively splitting the dendrogram and allocating inversely proportional to cluster variance

## Files

- `optimization_engines.py`: Core HRP implementation using Riskfolio-lib
  - `riskfolio_hrp()`: Main HRP optimization function with configurable parameters
  - `riskfolio_hrp_with_variants()`: Run multiple HRP variants with different parameters
  - `equal_weights_baseline()`: Equal weights baseline for comparison
  - `portfolio_stats()`: Calculate portfolio statistics

- `portfolio_optimization.ipynb`: Main notebook for running HRP experiments
  - Data loading and preprocessing
  - Backtesting framework
  - Experiment execution
  - Results visualization

- `hrp_results_analysis.ipynb`: Analysis notebook for HRP results
  - Performance metrics calculation
  - Comparison with baseline methods
  - Weight distribution analysis
  - Visualization of results

## Usage

### Basic HRP Optimization

```python
from optimization_engines import riskfolio_hrp
import pandas as pd

# Load your price data (DataFrame with assets as columns)
data = pd.read_pickle('path/to/data.pkl')

# Run HRP optimization
weights = riskfolio_hrp(
    data,
    codependence='pearson',  # Correlation method
    linkage='ward',          # Clustering linkage method
    max_k=10,                # Max clusters for gap statistic
    leaf_order=True          # Optimize dendrogram order
)

print(weights)
```

### Running Experiments

Open `portfolio_optimization.ipynb` and run the cells to:

1. Load stock price data
2. Configure experiment parameters
3. Run HRP optimization on multiple time periods
4. Compare with equal weights baseline
5. Visualize results

### Analyzing Results

Open `hrp_results_analysis.ipynb` to:

1. Load saved experiment results
2. Calculate performance metrics
3. Compare HRP vs baseline methods
4. Visualize weight distributions
5. Generate summary statistics

## Parameters

### Codependence Methods

- `'pearson'`: Pearson correlation (default, linear relationships)
- `'spearman'`: Spearman correlation (monotonic relationships)
- `'kendall'`: Kendall tau correlation (rank correlation)
- `'gerber1'`: Gerber statistic 1 (downside correlation)
- `'gerber2'`: Gerber statistic 2 (downside correlation)

### Linkage Methods

- `'ward'`: Ward variance minimization (default, creates balanced clusters)
- `'single'`: Single linkage / minimum distance (creates chain-like clusters)
- `'complete'`: Complete linkage / maximum distance (creates compact clusters)
- `'average'`: Average linkage (UPGMA, balanced approach)
- `'weighted'`: Weighted average linkage (WPGMA)

## Results

Results are saved in `../../results/hrp/` as pickle files with the naming convention:
- `results_hrp_<variant>_{period}.pkl`: HRP results for each variant and period
- `results_ew_{period}.pkl`: Equal weights baseline results

Each result file contains:
- `weights_history`: Time series of portfolio weights
- `portfolio_cumulative_returns`: Cumulative returns over the period
- `benchmark_cumulative_returns`: S&P 500 benchmark returns
- `total_run_time`: Execution time in seconds
- `start_date`, `end_date`: Period dates

## Dependencies

- `riskfolio-lib`: Main library for HRP implementation
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Covariance estimation (Ledoit-Wolf shrinkage)

## References

1. López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. The Journal of Portfolio Management, 42(4), 59-69.
2. Riskfolio-Lib Documentation: https://riskfolio-lib.readthedocs.io/
3. Hierarchical Risk Parity: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
