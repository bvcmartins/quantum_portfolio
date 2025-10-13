"""
Plotting utilities for portfolio optimization results analysis.
Provides consistent styling and reusable functions for all analysis notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import os
from typing import List, Dict, Optional, Any

# ============================================================================
# CONSISTENT STYLE CONFIGURATION
# ============================================================================

# Define consistent color palette
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'warning': '#D62246',      # Red
    'neutral': '#6C757D',      # Gray
    'benchmark': '#2D3142',    # Dark gray
}

# Solver-specific colors for consistency across plots
SOLVER_COLORS = {
    'Equal Weights': '#2D3142',
    'Genetic Algorithm': '#06A77D',
    'Scipy SLSQP': '#2E86AB',
    'Riskfolio QUBO': '#A23B72',
    'D-Wave CQM': '#F18F01',
    'D-Wave Hybrid': '#D62246',
    'Benchmark': '#6C757D',
}

# Font sizes
FONT_SIZES = {
    'title': 16,
    'subtitle': 14,
    'label': 12,
    'tick': 10,
    'legend': 10,
}

# Figure sizes (width, height)
FIGURE_SIZES = {
    'small': (10, 6),
    'medium': (12, 7),
    'large': (14, 8),
    'wide': (16, 6),
}


def apply_style():
    """Apply consistent matplotlib style settings."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'font.size': FONT_SIZES['label'],
        'axes.labelsize': FONT_SIZES['label'],
        'axes.titlesize': FONT_SIZES['subtitle'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'figure.titlesize': FONT_SIZES['title'],
    })


def get_solver_color(solver_name: str) -> str:
    """Get consistent color for a solver."""
    return SOLVER_COLORS.get(solver_name, COLORS['neutral'])


# ============================================================================
# CUMULATIVE RETURNS PLOTTING
# ============================================================================

def plot_cumulative_returns(
    returns_data: Dict[str, List[float]],
    benchmark_returns: Optional[List[float]] = None,
    title: str = "Portfolio Cumulative Returns",
    period_label: str = "",
    start_date: str = "",
    figsize: tuple = None
) -> plt.Figure:
    """
    Plot cumulative returns for multiple portfolios with benchmark comparison.

    Args:
        returns_data: Dictionary mapping solver names to cumulative returns lists
        benchmark_returns: Optional benchmark cumulative returns
        title: Plot title
        period_label: Label for the time period
        start_date: Start date for the period
        figsize: Figure size tuple (width, height)

    Returns:
        matplotlib Figure object
    """
    apply_style()

    if figsize is None:
        figsize = FIGURE_SIZES['large']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each solver's returns
    for solver_name, returns in returns_data.items():
        if returns is not None and len(returns) > 0:
            color = get_solver_color(solver_name)
            ax.plot(returns, label=solver_name, linewidth=2, color=color, alpha=0.9)

    # Plot benchmark if provided
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        ax.plot(benchmark_returns, label='Benchmark (Equal Weight)',
                linewidth=2, linestyle='--', color=COLORS['benchmark'], alpha=0.7)

    # Formatting
    full_title = title
    if period_label:
        full_title += f" - {period_label}"
    if start_date:
        full_title += f" (Starting {start_date})"

    ax.set_title(full_title, fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_xlabel('Trading Days', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_multi_period_comparison(
    periods_data: List[Dict[str, Any]],
    solver_names: List[str],
    title_prefix: str = "Portfolio Optimization",
    cols: int = 3,
    figsize: tuple = None
) -> plt.Figure:
    """
    Create multi-period comparison plot in a grid layout.

    Args:
        periods_data: List of dictionaries with period data
        solver_names: List of solver names to plot
        title_prefix: Prefix for the overall title
        cols: Number of columns in the grid
        figsize: Figure size tuple (width, height)

    Returns:
        matplotlib Figure object
    """
    apply_style()

    n_periods = len(periods_data)
    rows = (n_periods + cols - 1) // cols

    if figsize is None:
        figsize = (6 * cols, 5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_periods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, period_data in enumerate(periods_data):
        ax = axes[idx]

        # Plot each solver
        for solver_name in solver_names:
            if solver_name in period_data:
                returns = period_data[solver_name].get('returns', [])
                if returns is not None and len(returns) > 0:
                    color = get_solver_color(solver_name)
                    ax.plot(returns, label=solver_name, linewidth=2, color=color, alpha=0.9)

        # Plot benchmark if available
        if 'benchmark' in period_data:
            benchmark = period_data['benchmark']
            if benchmark is not None and len(benchmark) > 0:
                ax.plot(benchmark, label='Benchmark', linewidth=2,
                       linestyle='--', color=COLORS['benchmark'], alpha=0.7)

        # Formatting
        period_label = period_data.get('label', f'Period {idx}')
        start_date = period_data.get('start_date', '')
        ax.set_title(f"{period_label}\n{start_date}", fontsize=FONT_SIZES['subtitle'])
        ax.set_xlabel('Days', fontsize=FONT_SIZES['tick'])
        ax.set_ylabel('Cumulative Return', fontsize=FONT_SIZES['tick'])
        ax.legend(loc='best', fontsize=FONT_SIZES['tick'] - 1)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_periods, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title_prefix, fontsize=FONT_SIZES['title'], fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


# ============================================================================
# PERFORMANCE COMPARISON PLOTTING
# ============================================================================

def plot_performance_comparison(
    df_stats: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Performance Comparison",
    figsize: tuple = None
) -> plt.Figure:
    """
    Create bar plots comparing performance metrics across solvers.

    Args:
        df_stats: DataFrame with performance statistics
        metrics: List of metric column names to plot
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    apply_style()

    if metrics is None:
        metrics = ['Avg_Return', 'Std_Return', 'Avg_Time']

    n_metrics = len(metrics)
    if figsize is None:
        figsize = (5 * n_metrics, 6)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric not in df_stats.columns:
            ax.text(0.5, 0.5, f'Metric "{metric}" not found',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Create bar plot
        solvers = df_stats['Solver']
        values = df_stats[metric]
        colors = [get_solver_color(s) for s in solvers]

        bars = ax.bar(range(len(solvers)), values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=FONT_SIZES['tick'])

        # Formatting
        ax.set_xticks(range(len(solvers)))
        ax.set_xticklabels(solvers, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' '), fontsize=FONT_SIZES['label'], fontweight='bold')
        ax.set_title(metric.replace('_', ' '), fontsize=FONT_SIZES['subtitle'])
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=FONT_SIZES['title'], fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_performance_summary_table(
    df_stats: pd.DataFrame,
    title: str = "Performance Summary",
    figsize: tuple = None
) -> plt.Figure:
    """
    Create a formatted table visualization of performance statistics.

    Args:
        df_stats: DataFrame with performance statistics
        title: Table title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    apply_style()

    if figsize is None:
        figsize = FIGURE_SIZES['wide']

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df_stats.values,
                    colLabels=df_stats.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZES['tick'])
    table.scale(1, 2)

    # Color header
    for i in range(len(df_stats.columns)):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df_stats) + 1):
        for j in range(len(df_stats.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# ============================================================================
# WEIGHT DISTRIBUTION PLOTTING
# ============================================================================

def plot_weight_distribution_histograms(
    weights_by_solver: Dict[str, List[np.ndarray]],
    dataset_name: str,
    bins: int = 30,
    figsize: tuple = None
) -> plt.Figure:
    """
    Plot histograms of portfolio weight distributions for multiple solvers.

    Args:
        weights_by_solver: Dictionary mapping solver names to lists of weight arrays
        dataset_name: Name of the dataset for the title
        bins: Number of histogram bins
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    apply_style()

    n_solvers = len(weights_by_solver)
    if n_solvers == 0:
        return None

    if figsize is None:
        figsize = (14, 3 * n_solvers)

    fig, axes = plt.subplots(n_solvers, 1, figsize=figsize)
    if n_solvers == 1:
        axes = [axes]

    for idx, (solver_name, weights_list) in enumerate(weights_by_solver.items()):
        ax = axes[idx]

        if not weights_list:
            ax.text(0.5, 0.5, f'No weight data for {solver_name}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Flatten all weights
        all_weights = np.concatenate([w.flatten() for w in weights_list])

        # Create histogram
        color = get_solver_color(solver_name)
        ax.hist(all_weights, bins=bins, color=color, alpha=0.7, edgecolor='black')

        # Add statistics
        mean_weight = np.mean(all_weights)
        median_weight = np.median(all_weights)
        ax.axvline(mean_weight, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_weight:.4f}')
        ax.axvline(median_weight, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_weight:.4f}')

        # Formatting
        ax.set_title(f'{solver_name} - Weight Distribution',
                    fontsize=FONT_SIZES['subtitle'], fontweight='bold')
        ax.set_xlabel('Portfolio Weight', fontsize=FONT_SIZES['label'])
        ax.set_ylabel('Frequency', fontsize=FONT_SIZES['label'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Portfolio Weight Distributions - {dataset_name}',
                fontsize=FONT_SIZES['title'], fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_weight_heatmap(
    weights: np.ndarray,
    stock_names: List[str] = None,
    title: str = "Portfolio Weights Heatmap",
    figsize: tuple = None
) -> plt.Figure:
    """
    Create a heatmap visualization of portfolio weights across periods.

    Args:
        weights: 2D array of weights (periods x assets)
        stock_names: Optional list of stock names
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    apply_style()

    if figsize is None:
        figsize = (12, max(6, weights.shape[1] * 0.3))

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(weights.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', fontsize=FONT_SIZES['label'], fontweight='bold')

    # Labels
    ax.set_xlabel('Period', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_ylabel('Asset', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)

    # Set ticks
    if stock_names is not None and len(stock_names) == weights.shape[1]:
        ax.set_yticks(range(len(stock_names)))
        ax.set_yticklabels(stock_names, fontsize=FONT_SIZES['tick'] - 2)

    plt.tight_layout()
    return fig


# ============================================================================
# EXECUTION TIME PLOTTING
# ============================================================================

def plot_execution_times(
    df_stats: pd.DataFrame,
    title: str = "Execution Time Comparison",
    log_scale: bool = False,
    figsize: tuple = None
) -> plt.Figure:
    """
    Create bar plot of execution times with error bars.

    Args:
        df_stats: DataFrame with 'Solver', 'Avg_Time', and optionally 'Std_Time'
        title: Plot title
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    apply_style()

    if figsize is None:
        figsize = FIGURE_SIZES['medium']

    fig, ax = plt.subplots(figsize=figsize)

    solvers = df_stats['Solver']
    times = df_stats['Avg_Time']
    colors = [get_solver_color(s) for s in solvers]

    # Get error bars if available
    yerr = df_stats['Std_Time'] if 'Std_Time' in df_stats.columns else None

    # Create bar plot
    bars = ax.bar(range(len(solvers)), times, yerr=yerr,
                  color=colors, alpha=0.8, edgecolor='black', capsize=5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}s', ha='center', va='bottom', fontsize=FONT_SIZES['tick'])

    # Formatting
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels(solvers, rotation=45, ha='right')
    ax.set_ylabel('Execution Time (seconds)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()
    return fig


# ============================================================================
# CORRELATION AND RELATIONSHIP PLOTTING
# ============================================================================

def plot_risk_return_scatter(
    df_stats: pd.DataFrame,
    title: str = "Risk-Return Profile",
    figsize: tuple = None
) -> plt.Figure:
    """
    Create scatter plot of risk vs return for different solvers.

    Args:
        df_stats: DataFrame with 'Solver', 'Avg_Return', and 'Std_Return'
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    apply_style()

    if figsize is None:
        figsize = FIGURE_SIZES['medium']

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df_stats.iterrows():
        solver = row['Solver']
        risk = row['Std_Return']
        ret = row['Avg_Return']
        color = get_solver_color(solver)

        ax.scatter(risk, ret, s=200, color=color, alpha=0.7,
                  edgecolor='black', linewidth=2, label=solver)

    # Formatting
    ax.set_xlabel('Risk (Std. Deviation)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_ylabel('Return (Average)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300):
    """
    Save figure with consistent settings.

    Args:
        fig: matplotlib Figure object
        filename: Output filename
        dpi: Resolution in dots per inch
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {filename}")


def close_all_figures():
    """Close all open matplotlib figures to free memory."""
    plt.close('all')


# ============================================================================
# DATA LOADING AND ANALYSIS FUNCTIONS
# ============================================================================

def load_pickle_result(file_path):
    """Load a single pickle result file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def analyze_performance(df_subset, dataset_name):
    """Analyze performance statistics for a dataset subset."""
    print(f"\nAnalyzing performance for {dataset_name}...")

    if len(df_subset) == 0:
        print(f"No data available for {dataset_name}")
        return None

    performance_stats = []

    for solver_name in df_subset['solver_name'].unique():
        solver_data = df_subset[df_subset['solver_name'] == solver_name]

        # Filter for valid returns
        valid_data = solver_data[solver_data['final_return'].notna()]

        if len(valid_data) > 0:
            avg_return = valid_data['final_return'].mean()
            std_return = valid_data['final_return'].std()
            min_return = valid_data['final_return'].min()
            max_return = valid_data['final_return'].max()
            avg_time = valid_data['execution_time'].mean()
            std_time = valid_data['execution_time'].std()
            success_rate = len(valid_data) / len(solver_data) * 100

            performance_stats.append({
                'Solver': solver_name,
                'Periods': len(valid_data),
                'Avg_Return': avg_return,
                'Std_Return': std_return,
                'Min_Return': min_return,
                'Max_Return': max_return,
                'Avg_Time': avg_time,
                'Std_Time': std_time,
                'Success_Rate': success_rate
            })

    if performance_stats:
        return pd.DataFrame(performance_stats)
    return None


def extract_weights_from_results(df_subset):
    """Extract portfolio weights from all results in a dataset subset."""
    weights_by_solver = {}

    for _, result in df_subset.iterrows():
        solver_name = result['solver_name']
        data = result['data']

        if data is not None:
            weights = None

            # Check for weights_history first
            if 'weights_history' in data and data['weights_history'] is not None:
                weights_hist = data['weights_history']

                if isinstance(weights_hist, (list, tuple)) and len(weights_hist) > 0:
                    weights = np.array(weights_hist[-1])
                elif hasattr(weights_hist, 'shape') or hasattr(weights_hist, '__len__'):
                    weights = np.array(weights_hist)

            # Fallback to other possible keys
            if weights is None:
                possible_weight_keys = [
                    'portfolio_weights',
                    'weights',
                    'optimal_weights',
                    'final_weights',
                    'solution_weights'
                ]

                for key in possible_weight_keys:
                    if key in data and data[key] is not None:
                        weights = np.array(data[key])
                        break

            # Store weights if found
            if weights is not None and len(weights) > 0:
                if solver_name not in weights_by_solver:
                    weights_by_solver[solver_name] = []
                weights_by_solver[solver_name].append(weights)

    return weights_by_solver


def multi_comparison(solver_configs, n_periods, title_prefix="Portfolio Optimization", show_benchmark=True):
    """
    Compare multiple portfolio optimization solvers across periods.

    Args:
        solver_configs (list): List of dictionaries with 'path_template' and 'name' keys
        n_periods (int): Number of periods to compare
        title_prefix (str): Prefix for plot titles
        show_benchmark (bool): Whether to show benchmark comparison
    """

    for period in range(n_periods):
        returns_data = {}
        benchmark_returns = None
        period_label = f"Period {period}"
        start_date = ""

        # Load data for all solvers for this period
        for config in solver_configs:
            path_template = config['path_template']
            solver_name = config['name']
            file_path = path_template.format(period)

            try:
                result_data = load_pickle_result(file_path)
                if result_data is not None:
                    portfolio_returns = result_data.get('portfolio_cumulative_returns', [])
                    benchmark_ret = result_data.get('benchmark_cumulative_returns', [])
                    start_date = result_data.get('start_date', 'Unknown')

                    # Store portfolio returns
                    if portfolio_returns is not None and len(portfolio_returns) > 0:
                        returns_data[solver_name] = portfolio_returns

                    # Store benchmark (same for all solvers)
                    if benchmark_returns is None and benchmark_ret is not None and len(benchmark_ret) > 0:
                        benchmark_returns = benchmark_ret
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Plot if we have data
        if returns_data:
            fig = plot_cumulative_returns(
                returns_data,
                benchmark_returns=benchmark_returns if show_benchmark else None,
                title=title_prefix,
                period_label=period_label,
                start_date=start_date,
                figsize=FIGURE_SIZES['large']
            )
            plt.show()


def analyze_dataset(df, dataset_key, dataset_name, solver_configs):
    """
    Standardized analysis for a dataset.

    Args:
        df: Full results DataFrame
        dataset_key: Dataset key ('10', '100', 'full', 'wide')
        dataset_name: Display name for the dataset
        solver_configs: List of solver configurations for multi-period comparison

    Returns:
        Tuple of (df_subset, stats) for the dataset
    """
    print("\n" + "=" * 100)
    print(f"{dataset_name.upper()} - COMPREHENSIVE ANALYSIS")
    print("=" * 100)

    # Filter dataset
    df_subset = df[df['dataset'] == dataset_key].copy()

    if len(df_subset) == 0:
        print(f"No results found for {dataset_name}")
        return None, None

    print(f"\nFound {len(df_subset)} results for {dataset_name}")
    print(f"Solvers: {sorted(df_subset['solver_name'].unique())}")
    print(f"Periods: {sorted(df_subset['period'].unique())}")

    # 1. Performance Statistics
    print(f"\n{'='*80}")
    print(f"{dataset_name} - PERFORMANCE STATISTICS")
    print(f"{'='*80}")

    stats = analyze_performance(df_subset, dataset_name)
    if stats is not None:
        stats_sorted = stats.sort_values('Avg_Return', ascending=False)
        print("\n" + stats_sorted.to_string(index=False))

        # Plot 1: Performance Comparison
        fig = plot_performance_comparison(
            stats_sorted,
            metrics=['Avg_Return', 'Std_Return', 'Avg_Time'],
            title=f"{dataset_name} - Performance Comparison"
        )
        plt.show()

        # Plot 2: Risk-Return Profile
        fig = plot_risk_return_scatter(
            stats_sorted,
            title=f"{dataset_name} - Risk-Return Profile"
        )
        plt.show()

    # 2. Weight Distribution Analysis
    print(f"\n{'='*80}")
    print(f"{dataset_name} - WEIGHT DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")

    weights_by_solver = extract_weights_from_results(df_subset)

    if weights_by_solver:
        # Plot 3: Weight Distributions
        fig = plot_weight_distribution_histograms(
            weights_by_solver,
            dataset_name,
            bins=50 if dataset_key in ['full', 'wide'] else 30
        )
        if fig:
            plt.show()

    return df_subset, stats
