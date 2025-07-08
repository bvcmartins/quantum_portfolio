# Quantum Portfolio Optimization

Portfolio Optimization using Classical and Quantum methods. The objective of this project
is to compare classical and quantum methods applied to portfolio optimization problems. Our objective is
not to generate a real-world tool for investing problems but to compare different optimization methods
using a rich data set.

## Description

For this project we collected open data on 493 S&P 500 stocks for the period between 2011 and 2024. 

Given the data we randomly chose 10 30-day intervals. The portfolio was accessed without forecasting since the main 
objective of this project was only to compare optimization methods. 

We tested the following methods:
- Genetic Algorithms 
- Matrix Diagonalization using Scipy
- Hierarchical Risk Parity (HRP)

We are well-aware that backtesting on historical data is not the ideal way of testing allocation methods. We are developing another project aimed at generating mock data using a Variational Auto Encoder (VAE) and we intend to use its results in future iterations of the portfolio optimization problem. 

The project is ready to soon incorporate results obtained d-wave's Advantage 2. 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bvcmartins/quantum_portfolio.git
   ```

2. Install dependencies:
   ```bash
   conda install -y --file requirements.txt
   ```

## Contributing

1. Fork the project
2. Create a new branch for your feature/fix
3. Submit a pull request with clear descriptions

For major changes, please open an issue first to discuss what you'd like to implement.

## License

MIT License - see [LICENSE](LICENSE) file for details.
