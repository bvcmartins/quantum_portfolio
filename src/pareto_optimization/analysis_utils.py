import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging

logger = logging.getLogger("inspect_results_logger")


def write_pickle_dict(data, file_path):
    """Pickles a dictionary and saves it to a file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dictionary pickled and saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while pickling: {e}")


def read_pickle_dict(file_path):
    """Loads a pickled dictionary from a file."""
    try:
        with open(file_path, 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def is_dominated(point, other_points):
    """
    Check if a point is dominated by any other point in the set.

    For portfolio optimization:
    - point A dominates point B if A has higher return AND lower risk

    Args:
        point: (return, risk) tuple
        other_points: list of (return, risk) tuples

    Returns:
        True if point is dominated, False otherwise
    """
    ret, risk = point
    for other_ret, other_risk in other_points:
        # Other point dominates if it has higher return AND lower risk
        if other_ret > ret and other_risk < risk:
            return True
    return False


def extract_pareto_frontier(results_dict):
    """
    Extract Pareto-optimal solutions from a set of optimization results.

    Args:
        results_dict: Dictionary mapping risk_level -> {return, risk, weights, ...}

    Returns:
        pareto_dict: Dictionary of Pareto-optimal solutions
    """
    # Extract return-risk pairs
    points = []
    for risk_level, result in results_dict.items():
        if result is not None and 'portfolio_return' in result and 'portfolio_risk' in result:
            ret = result['portfolio_return']
            risk = result['portfolio_risk']
            if not np.isnan(ret) and not np.isnan(risk):
                points.append((risk_level, ret, risk))

    if not points:
        return {}

    # Find Pareto-optimal points
    pareto_solutions = {}
    for i, (risk_level, ret, risk) in enumerate(points):
        other_points = [(r, rsk) for j, (_, r, rsk) in enumerate(points) if j != i]
        if not is_dominated((ret, risk), other_points):
            pareto_solutions[risk_level] = results_dict[risk_level]

    return pareto_solutions


def plot_pareto_frontier(results_dict, title="Pareto Frontier", show_dominated=True, benchmark_point=None):
    """
    Plot the Pareto frontier for a set of optimization results.

    Args:
        results_dict: Dictionary mapping risk_level -> {portfolio_return, portfolio_risk, ...}
        title: Plot title
        show_dominated: Whether to show dominated points
        benchmark_point: Optional (return, risk) tuple for benchmark comparison
    """
    # Extract return-risk pairs
    risk_levels = []
    returns = []
    risks = []

    for risk_level, result in sorted(results_dict.items()):
        if result is not None and 'portfolio_return' in result and 'portfolio_risk' in result:
            ret = result['portfolio_return']
            risk = result['portfolio_risk']
            if not np.isnan(ret) and not np.isnan(risk):
                risk_levels.append(risk_level)
                returns.append(ret)
                risks.append(risk)

    if not returns:
        print("No valid data points to plot")
        return

    # Identify Pareto-optimal points
    pareto_solutions = extract_pareto_frontier(results_dict)
    pareto_risk_levels = list(pareto_solutions.keys())
    pareto_returns = [pareto_solutions[rl]['portfolio_return'] for rl in pareto_risk_levels]
    pareto_risks = [pareto_solutions[rl]['portfolio_risk'] for rl in pareto_risk_levels]

    # Sort Pareto points by risk for plotting
    pareto_sorted = sorted(zip(pareto_risks, pareto_returns, pareto_risk_levels))
    pareto_risks_sorted = [x[0] for x in pareto_sorted]
    pareto_returns_sorted = [x[1] for x in pareto_sorted]
    pareto_risk_levels_sorted = [x[2] for x in pareto_sorted]

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot all points with lambda labels
    if show_dominated:
        # Plot dominated points
        dominated_mask = [rl not in pareto_risk_levels for rl in risk_levels]
        dominated_risks = [risks[i] for i in range(len(risks)) if dominated_mask[i]]
        dominated_returns = [returns[i] for i in range(len(returns)) if dominated_mask[i]]
        dominated_lambdas = [risk_levels[i] for i in range(len(risk_levels)) if dominated_mask[i]]

        plt.scatter(dominated_risks, dominated_returns, color='lightgray',
                   s=100, alpha=0.4, label='Dominated Solutions', edgecolors='gray')

        # Annotate dominated points with lambda
        for risk, ret, rl in zip(dominated_risks, dominated_returns, dominated_lambdas):
            plt.text(risk, ret, f'λ={rl:.1f}',
                    ha='center', va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.6, edgecolor='gray'))

    # Plot Pareto frontier
    plt.scatter(pareto_risks_sorted, pareto_returns_sorted,
               color='steelblue',
               s=200, alpha=0.9, edgecolors='black', linewidths=2,
               label='Pareto Frontier', marker='D')

    # Connect Pareto points with line
    plt.plot(pareto_risks_sorted, pareto_returns_sorted,
            'k--', alpha=0.5, linewidth=1.5)

    # Annotate Pareto points with risk_level (lambda) on top of each point
    for risk, ret, rl in zip(pareto_risks_sorted, pareto_returns_sorted, pareto_risk_levels_sorted):
        plt.text(risk, ret, f'λ={rl:.1f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))

    # Plot benchmark if provided
    if benchmark_point is not None:
        bench_ret, bench_risk = benchmark_point
        plt.scatter([bench_risk], [bench_ret],
                   s=300, marker='*', color='red',
                   edgecolors='black', linewidths=2,
                   label='Benchmark', zorder=10)

    plt.xlabel('Portfolio Risk (Volatility)', fontsize=12)
    plt.ylabel('Portfolio Return', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nPareto Frontier Summary:")
    print(f"{'='*60}")
    print(f"Total solutions evaluated: {len(returns)}")
    print(f"Pareto-optimal solutions: {len(pareto_returns)}")
    print(f"Efficiency: {len(pareto_returns)/len(returns)*100:.1f}%")
    print(f"\nPareto Solutions:")
    print(f"{'Risk Level':<12} {'Return':<12} {'Risk':<12} {'Sharpe':<12}")
    print(f"{'-'*60}")
    for rl, ret, risk in zip(pareto_risk_levels_sorted, pareto_returns_sorted, pareto_risks_sorted):
        sharpe = ret / risk if risk > 0 else 0
        print(f"{rl:<12.3f} {ret:<12.6f} {risk:<12.6f} {sharpe:<12.4f}")


def compare_pareto_frontiers(frontiers_dict, title="Pareto Frontier Comparison", annotate_lambda=False):
    """
    Compare Pareto frontiers from multiple optimization methods.

    Args:
        frontiers_dict: Dictionary mapping method_name -> results_dict
        title: Plot title
        annotate_lambda: If True, annotate points with lambda values (only recommended for single method)
    """
    plt.figure(figsize=(14, 9))

    colors = plt.cm.Set1(np.linspace(0, 1, len(frontiers_dict)))

    for idx, (method_name, results_dict) in enumerate(frontiers_dict.items()):
        # Extract Pareto frontier
        pareto_solutions = extract_pareto_frontier(results_dict)

        if not pareto_solutions:
            continue

        pareto_risk_levels = list(pareto_solutions.keys())
        pareto_returns = [pareto_solutions[rl]['portfolio_return'] for rl in pareto_risk_levels]
        pareto_risks = [pareto_solutions[rl]['portfolio_risk'] for rl in pareto_risk_levels]

        # Sort by risk
        pareto_sorted = sorted(zip(pareto_risks, pareto_returns, pareto_risk_levels))
        pareto_risks_sorted = [x[0] for x in pareto_sorted]
        pareto_returns_sorted = [x[1] for x in pareto_sorted]
        pareto_risk_levels_sorted = [x[2] for x in pareto_sorted]

        # Plot
        plt.scatter(pareto_risks_sorted, pareto_returns_sorted,
                   s=150, alpha=0.8, color=colors[idx],
                   edgecolors='black', linewidths=1.5,
                   label=method_name, marker='o')

        plt.plot(pareto_risks_sorted, pareto_returns_sorted,
                alpha=0.6, linewidth=2, color=colors[idx], linestyle='--')

        # Annotate with lambda values if requested
        if annotate_lambda:
            for risk, ret, rl in zip(pareto_risks_sorted, pareto_returns_sorted, pareto_risk_levels_sorted):
                plt.text(risk, ret, f'λ={rl:.2f}',
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=colors[idx]))

    plt.xlabel('Portfolio Risk (Volatility)', fontsize=12)
    plt.ylabel('Portfolio Return', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weight_evolution(pareto_solutions, data_columns, top_n=10):
    """
    Plot how portfolio weights evolve along the Pareto frontier.

    Args:
        pareto_solutions: Dictionary of Pareto-optimal solutions
        data_columns: List of asset names
        top_n: Number of top assets to show
    """
    risk_levels = sorted(pareto_solutions.keys())

    # Collect weights for each risk level
    weights_matrix = []
    for rl in risk_levels:
        weights = pareto_solutions[rl].get('weights', None)
        if weights is not None:
            weights_matrix.append(weights)

    if not weights_matrix:
        print("No weight data available")
        return

    weights_matrix = np.array(weights_matrix)

    # Find top N assets by average weight
    avg_weights = weights_matrix.mean(axis=0)
    top_indices = np.argsort(avg_weights)[-top_n:]

    # Plot weight evolution for top assets
    plt.figure(figsize=(14, 8))

    for idx in top_indices:
        asset_weights = weights_matrix[:, idx]
        plt.plot(risk_levels, asset_weights, marker='o', linewidth=2,
                label=data_columns[idx], alpha=0.7)

    plt.xlabel('Risk Aversion Parameter (λ)', fontsize=12)
    plt.ylabel('Portfolio Weight', fontsize=12)
    plt.title(f'Weight Evolution Along Pareto Frontier (Top {top_n} Assets)',
             fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot weight concentration
    plt.figure(figsize=(12, 6))

    # Calculate concentration metrics
    herfindahl_indices = [(w**2).sum() for w in weights_matrix]
    effective_n = [1/h for h in herfindahl_indices]

    plt.subplot(1, 2, 1)
    plt.plot(risk_levels, herfindahl_indices, marker='o', linewidth=2, color='steelblue')
    plt.xlabel('Risk Aversion Parameter (λ)', fontsize=11)
    plt.ylabel('Herfindahl Index', fontsize=11)
    plt.title('Portfolio Concentration', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(risk_levels, effective_n, marker='o', linewidth=2, color='coral')
    plt.xlabel('Risk Aversion Parameter (λ)', fontsize=11)
    plt.ylabel('Effective Number of Assets', fontsize=11)
    plt.title('Portfolio Diversification', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
