# Pareto Optimization for QUBO Portfolio Optimization

This module implements Pareto frontier analysis for portfolio optimization using QUBO (Quadratic Unconstrained Binary Optimization) formulations.

## Overview

The Pareto optimization approach explores the **risk-return tradeoff** by varying the risk aversion parameter (λ) in the QUBO objective function:

```
Objective = Return - λ × Variance
```

By sweeping through different values of λ, we generate multiple portfolio solutions that form the **Pareto efficient frontier** - the set of portfolios where you cannot improve return without increasing risk, or reduce risk without sacrificing return.

## Files

### `optimization_engines.py`
Contains optimization algorithms adapted for Pareto frontier exploration:
- **`genetic_algorithm_qubo()`** - GA with parameterized risk level
- **`scipy_slsqp_qubo()`** - SciPy SLSQP with parameterized risk level
- **`riskfolio_qubo()`** - Riskfolio-lib with parameterized risk level
- **`dwave_cqm_qubo()`** - D-Wave quantum solver with parameterized risk level
- **`equal_weights_baseline()`** - Equal weights baseline (risk-agnostic)

All functions accept a `risk_level` parameter to control the risk-return tradeoff.

### `analysis_utils.py`
Utilities for Pareto frontier analysis and visualization:
- **`extract_pareto_frontier()`** - Filter Pareto-optimal solutions from results
- **`is_dominated()`** - Check if a solution is dominated by others
- **`plot_pareto_frontier()`** - Visualize Pareto frontier with risk-return scatter
- **`compare_pareto_frontiers()`** - Compare multiple optimization methods
- **`plot_weight_evolution()`** - Show how weights change along frontier
- **`write_pickle_dict()` / `read_pickle_dict()`** - Save/load results

### `pareto_optimization.ipynb`
Main notebook implementing Pareto frontier analysis:
- Data loading (stocks and ETFs)
- Pareto sweep across multiple risk levels
- Comparison of optimization methods
- Weight evolution analysis
- Visualization and insights

## Usage

### Basic Pareto Sweep

```python
from optimization_engines import scipy_slsqp_qubo, portfolio_stats
from analysis_utils import plot_pareto_frontier

# Define risk levels to explore
risk_levels = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

# Optimize for each risk level
results = {}
for risk_level in risk_levels:
    weights = scipy_slsqp_qubo(data, risk_level=risk_level)
    port_return, port_vol = portfolio_stats(weights, data)

    results[risk_level] = {
        'weights': weights,
        'portfolio_return': port_return,
        'portfolio_risk': port_vol,
        'sharpe_ratio': port_return / port_vol
    }

# Visualize Pareto frontier
plot_pareto_frontier(results, title="Pareto Frontier Analysis")
```

### Using the Helper Function

```python
from pareto_optimization import pareto_sweep

# Automatic Pareto sweep
results = pareto_sweep(
    data=stock_data,
    optimization_function=genetic_algorithm_qubo,
    risk_levels=[0.1, 0.5, 1.0, 2.0],
    population_size=100,
    num_generations=100
)
```

## Key Concepts

### Pareto Dominance
A solution A **dominates** solution B if:
- A has **higher return** AND **lower risk** than B

Pareto-optimal solutions are those that are **not dominated** by any other solution.

### Risk Aversion Parameter (λ)
- **λ → 0**: Risk-seeking (maximize return, ignore risk)
- **λ = 0.5**: Balanced approach
- **λ → ∞**: Risk-averse (minimize risk, ignore return)

Typical range: [0.05, 3.0]

### Efficient Frontier
The curve connecting Pareto-optimal solutions in risk-return space. Points on this frontier represent the best possible risk-return tradeoffs.

## Experiments

The notebook runs two scale experiments:

1. **Full Dataset (439 stocks)**: Large-scale real-world S&P 500 stocks
2. **Wide Dataset (680 assets)**: Stocks + ETFs combined portfolio

For each scale, we compare:
- Genetic Algorithm (GA)
- SciPy SLSQP (gradient-based)
- Riskfolio-lib (specialized portfolio optimizer)

## Visualization

The module provides rich visualizations:

1. **Pareto Frontier Plot**: Risk vs Return scatter with frontier highlighted
2. **Method Comparison**: Overlay frontiers from different optimizers
3. **Weight Evolution**: How asset allocations change along frontier
4. **Concentration Metrics**: Herfindahl index and effective number of assets
5. **Diversification Analysis**: Portfolio concentration vs risk level

## Results

Results are saved to `../../results/pareto_optimization/` in pickle format:
- `pareto_full_ga.pkl` - GA results for full dataset (439 stocks)
- `pareto_full_slsqp.pkl` - SLSQP results for full dataset
- `pareto_full_riskfolio.pkl` - Riskfolio results for full dataset
- `pareto_wide_ga.pkl` - GA results for wide dataset (680 assets)
- `pareto_wide_slsqp.pkl` - SLSQP results for wide dataset
- `pareto_wide_riskfolio.pkl` - Riskfolio results for wide dataset

Each result contains:
```python
{
    risk_level: {
        'weights': np.array(...),
        'portfolio_return': float,
        'portfolio_risk': float,
        'portfolio_variance': float,
        'sharpe_ratio': float,
        'execution_time': float
    }
}
```

## Compatibility

This implementation is **fully compatible** with the existing QUBO framework in:
- `src/qubo_unrestricted/`
- `src/qubo_diversified/`

The main difference is that optimization functions now accept a `risk_level` parameter, allowing systematic exploration of the Pareto frontier.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- riskfolio-lib
- dwave-ocean-sdk (for quantum optimization)

## Example Output

```
Pareto Frontier Summary:
============================================================
Total solutions evaluated: 10
Pareto-optimal solutions: 7
Efficiency: 70.0%

Pareto Solutions:
Risk Level   Return       Risk         Sharpe
------------------------------------------------------------
0.050        0.002145     0.015234     0.1408
0.100        0.002312     0.012456     0.1856
0.200        0.002489     0.010123     0.2459
0.500        0.002534     0.008891     0.2850
1.000        0.002398     0.007234     0.3315
2.000        0.001989     0.005678     0.3502
```

## Future Enhancements

- Multi-period Pareto analysis (tracking frontier over time)
- Robust Pareto optimization (uncertainty quantification)
- Constrained Pareto frontier (sector limits, ESG constraints)
- Interactive Pareto frontier exploration
- Quantum-enhanced Pareto optimization with D-Wave

## References

1. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
2. Pareto, V. (1896). *Cours d'économie politique*.
3. Sharpe, W. F. (1964). Capital Asset Prices. *Journal of Finance*.
