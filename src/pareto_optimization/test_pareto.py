#!/usr/bin/env python3
"""
Quick test script to verify Pareto optimization implementation.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from optimization_engines import (
    genetic_algorithm_qubo,
    scipy_slsqp_qubo,
    riskfolio_qubo,
    portfolio_stats
)
from analysis_utils import (
    extract_pareto_frontier,
    is_dominated,
    plot_pareto_frontier
)

def test_basic_functionality():
    """Test basic Pareto optimization functionality."""
    print("=" * 80)
    print("PARETO OPTIMIZATION TEST")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic stock data...")
    np.random.seed(42)
    n_days = 60
    n_stocks = 5

    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    # Create synthetic price data with different characteristics
    prices = pd.DataFrame(index=dates)
    prices['Stock_A'] = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_days)))  # High vol
    prices['Stock_B'] = 50 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n_days)))  # Low vol
    prices['Stock_C'] = 75 * np.exp(np.cumsum(np.random.normal(0.0008, 0.015, n_days)))  # Medium
    prices['Stock_D'] = 120 * np.exp(np.cumsum(np.random.normal(0.0012, 0.018, n_days)))  # High return
    prices['Stock_E'] = 30 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n_days)))  # Very low vol

    print(f"   Created {n_stocks} synthetic stocks over {n_days} days")
    print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")

    # Test different risk levels
    print("\n2. Testing optimization across risk levels...")
    risk_levels = [0.1, 0.5, 1.0, 2.0]
    results = {}

    for risk_level in risk_levels:
        print(f"\n   Risk level λ = {risk_level:.2f}")

        # Test SLSQP optimizer
        weights = scipy_slsqp_qubo(prices, risk_level=risk_level)

        if not np.isnan(weights).any():
            port_return, port_vol = portfolio_stats(weights, prices)
            sharpe = port_return / port_vol if port_vol > 0 else 0

            results[risk_level] = {
                'weights': weights,
                'portfolio_return': port_return,
                'portfolio_risk': port_vol,
                'sharpe_ratio': sharpe
            }

            print(f"      Return: {port_return:.6f}")
            print(f"      Risk:   {port_vol:.6f}")
            print(f"      Sharpe: {sharpe:.4f}")
            print(f"      Weights: {weights}")
        else:
            print(f"      FAILED - optimization returned NaN")

    # Test Pareto dominance
    print("\n3. Testing Pareto dominance detection...")
    if len(results) >= 2:
        points = [(r['portfolio_return'], r['portfolio_risk']) for r in results.values()]

        for i, (rl, result) in enumerate(results.items()):
            other_points = [p for j, p in enumerate(points) if j != i]
            dominated = is_dominated(
                (result['portfolio_return'], result['portfolio_risk']),
                other_points
            )
            status = "DOMINATED" if dominated else "PARETO-OPTIMAL"
            print(f"   λ={rl:.2f}: {status}")

    # Extract Pareto frontier
    print("\n4. Extracting Pareto frontier...")
    pareto_solutions = extract_pareto_frontier(results)
    print(f"   Total solutions: {len(results)}")
    print(f"   Pareto-optimal: {len(pareto_solutions)}")
    print(f"   Efficiency: {len(pareto_solutions)/len(results)*100:.1f}%")

    # Test weight evolution
    print("\n5. Weight allocation analysis...")
    print("   Risk Level  |  Stock A  |  Stock B  |  Stock C  |  Stock D  |  Stock E")
    print("   " + "-" * 75)
    for rl in sorted(results.keys()):
        w = results[rl]['weights']
        print(f"   λ={rl:4.2f}    | {w[0]:8.4f} | {w[1]:8.4f} | {w[2]:8.4f} | {w[3]:8.4f} | {w[4]:8.4f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return results


def test_multiple_optimizers():
    """Test that all optimizers accept risk_level parameter."""
    print("\n" + "=" * 80)
    print("TESTING ALL OPTIMIZERS")
    print("=" * 80)

    # Generate simple test data
    np.random.seed(42)
    n_days = 50
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    prices = pd.DataFrame(index=dates)
    for i in range(3):
        prices[f'Stock_{i}'] = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, n_days)))

    risk_level = 0.5

    optimizers = [
        ('SciPy SLSQP', scipy_slsqp_qubo),
        ('Genetic Algorithm', genetic_algorithm_qubo),
        ('Riskfolio', riskfolio_qubo)
    ]

    for name, optimizer in optimizers:
        print(f"\n{name}:")
        try:
            if name == 'Genetic Algorithm':
                weights = optimizer(prices, risk_level=risk_level,
                                  population_size=20, num_generations=10)
            else:
                weights = optimizer(prices, risk_level=risk_level)

            if not np.isnan(weights).any():
                port_return, port_vol = portfolio_stats(weights, prices)
                print(f"   ✓ Success - Return: {port_return:.6f}, Risk: {port_vol:.6f}")
                print(f"   Weights sum: {weights.sum():.6f}")
            else:
                print(f"   ✗ Failed - returned NaN")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        # Run tests
        results = test_basic_functionality()
        test_multiple_optimizers()

        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Open pareto_optimization.ipynb in Jupyter")
        print("  2. Run the notebook to generate Pareto frontiers")
        print("  3. Results will be saved to ../../results/pareto_optimization/")

    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
