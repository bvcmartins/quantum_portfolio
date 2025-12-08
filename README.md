# Quantum Portfolio Optimization

A comprehensive comparison of classical and quantum optimization methods for portfolio allocation problems. This project benchmarks multiple optimization algorithms on financial market data across different optimization objectives.

## Overview

This project compares classical and quantum methods for portfolio optimization using a rich dataset of 680 securities (stocks and ETFs) spanning 2011-2024. The objective is **not** to create a production investing tool, but rather to provide a rigorous comparison of optimization methods across different problem formulations.

## Dataset

- **Securities**: 680 stocks and ETFs
- **Time Period**: 2011-2024
- **Test Periods**: 10 randomly selected 30-day intervals for walk-forward testing
- **Data Sources**: Historical market data and synthetic data generation
- **Approach**: Historical backtesting (no forecasting) to focus purely on optimization method comparison

## Project Structure

```
quantum_portfolio/
├── src/
│   ├── sharpe_optimization/     # Sharpe ratio maximization
│   ├── pareto_optimization/     # Multi-objective risk-return optimization
│   ├── qubo/                    # Legacy QUBO formulations
│   ├── util/                    # Shared utilities and plotting
│   └── synthetic_data/          # Synthetic data generation capabilities
├── results/                     # Optimization results (pickle files)
└── data/                        # Market data (680 securities)
```

## Optimization Methods

### 1. Sharpe Ratio Optimization

**Objective**: Maximize risk-adjusted returns (Sharpe ratio)

**Solvers**:
- **Equal Weights** - Baseline (1/n allocation)
- **Genetic Algorithm** - Evolutionary optimization
- **Scipy SLSQP** - Sequential Least Squares Programming
- **Riskfolio QUBO** - QUBO-based portfolio optimization
- **D-Wave Nonlinear** - Quantum nonlinear programming (limited testing)

**Analysis**:
- `sharpe_results_analysis.ipynb` - Cumulative returns traces, performance metrics, weight distribution analysis

### 2. Pareto Optimization

**Objective**: Explore the efficient frontier (risk-return tradeoffs)

**Solvers**:
- **Genetic Algorithm** - Multi-objective evolutionary algorithm
- **Scipy SLSQP** - Constrained optimization with varying lambda (risk aversion)
- **Riskfolio** - Modern portfolio theory methods
- **D-Wave CQM** - Constrained Quadratic Model on quantum hardware

**Lambda Values**: 11 risk-aversion levels (0.0 to 1.0) per solver

**Analysis**:
- `pareto_results_analysis.ipynb` - Individual and combined efficient frontiers
- Averaging across periods for fair comparison
- Pareto-optimal point identification

### 3. QUBO Optimization (Legacy)

**Objective**: Quadratic Unconstrained Binary Optimization formulations

**Note**: Earlier experimental approach, now superseded by Sharpe and Pareto methods

## Key Results

### Sharpe Optimization
- **10 periods** tested across all classical solvers
- **3 periods** tested on D-Wave (due to quantum resource constraints)
- Weight distribution analysis across 680 securities showing portfolio concentration metrics
- Runtime vs. performance tradeoff analysis

### Pareto Optimization
- **Classical solvers**: 10 periods × 11 lambda values = 110 data points each
- **D-Wave CQM**: 1 period × 11 lambda values = 11 data points
- Efficient frontiers averaged across periods for robust comparison
- Direct visualization of risk-return tradeoffs across the 680-security universe

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- (Optional) D-Wave Ocean SDK for quantum computing access

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bvcmartins/quantum_portfolio.git
   cd quantum_portfolio
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure D-Wave credentials:
   ```bash
   dwave config create
   ```

## Usage

### Running Optimizations

Navigate to the relevant optimization directory:

```bash
# Sharpe optimization
cd src/sharpe_optimization
jupyter notebook portfolio_optimization.ipynb

# Pareto optimization
cd src/pareto_optimization
jupyter notebook pareto_optimization.ipynb
```

### Analyzing Results

Each optimization approach has a dedicated analysis notebook:

```bash
# Sharpe results
jupyter notebook src/sharpe_optimization/sharpe_results_analysis.ipynb

# Pareto results
jupyter notebook src/pareto_optimization/pareto_results_analysis.ipynb
```

## Analysis Features

### Sharpe Results Analysis
- **Cumulative returns traces** for all periods and solvers
- **Average performance metrics** (returns, runtime, consistency)
- **Weight distribution histograms** with concentration analysis
- **Statistical measures**: Herfindahl-Hirschman Index, Gini coefficient, effective positions

### Pareto Results Analysis
- **Individual efficient frontiers** for each solver (averaged across periods)
- **Combined comparison** of all solvers on one plot
- **Lambda-labeled points** showing risk preference tradeoffs
- **Pareto-optimal identification** for each solver

## Synthetic Data Generation

The project includes capabilities for generating synthetic market data to complement historical backtesting. This allows for:
- Testing optimization methods on realistic but controlled scenarios
- Evaluating robustness across different market regimes
- Stress-testing portfolio allocation strategies

## Limitations & Future Work

### Current Limitations
- **No forecasting**: Backtesting on historical data only
- **Limited quantum testing**: D-Wave runs limited to reduce costs
- **No transaction costs**: Simplified portfolio rebalancing model

### Planned Improvements
- **Extended quantum testing**: More comprehensive D-Wave Advantage 2 benchmarks
- **Additional solvers**: Simulated annealing, particle swarm optimization
- **Enhanced risk metrics**: Beyond volatility - CVaR, drawdown, tail risk
- **Real-time optimization**: Adaptation for streaming market data

## Technical Notes

### Data Format
Results are stored as pickle files containing:
- Portfolio weights (680-dimensional vectors)
- Cumulative returns (daily time series)
- Risk metrics (volatility, Sharpe ratio, portfolio variance)
- Execution time
- Solver-specific metadata

### Reproducibility
- Fixed random seeds for stochastic algorithms
- Deterministic test period selection
- All hyperparameters documented in optimization notebooks
- Version-controlled dependencies

## Contributing

We welcome contributions! Areas of interest:

1. **New solvers**: Additional classical or quantum optimization methods
2. **Analysis tools**: Enhanced visualization or statistical analysis
3. **Documentation**: Improved explanations or tutorials
4. **Synthetic data**: Enhanced market simulation models
5. **Performance**: Optimization speed improvements or parallel implementations

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-solver`)
3. Commit your changes (`git commit -m 'Add amazing solver'`)
4. Push to the branch (`git push origin feature/amazing-solver`)
5. Open a Pull Request with clear description

For major changes, please open an issue first to discuss the proposed modifications.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{quantum_portfolio_2024,
  title = {Quantum Portfolio Optimization: Benchmarking Classical and Quantum Methods},
  author = {Martins, B.V.C.},
  year = {2024},
  url = {https://github.com/bvcmartins/quantum_portfolio}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **D-Wave Systems**: Quantum computing platform and Ocean SDK
- **Riskfolio-Lib**: Portfolio optimization library
- **Market Data Providers**: Historical securities data (680 stocks and ETFs)

## Contact

For questions, issues, or collaboration inquiries, please open an issue on GitHub.

---

**Disclaimer**: This project is for research and educational purposes only. It should not be used as financial advice or for actual investment decisions.
