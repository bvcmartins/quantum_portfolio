"""
Utility functions for QUBO portfolio optimization analysis.

This module contains reusable functions for analyzing and visualizing
portfolio optimization results across different solvers and datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def load_pickle_result(file_path):
    """
    Load results from a pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        dict or None: Loaded data or None if file doesn't exist/can't be loaded
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, Exception):
        return None


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
        period_data = []
        benchmark_data = None

        # Load data for all solvers for this period
        for config in solver_configs:
            path_template = config['path_template']
            solver_name = config['name']
            file_path = path_template.format(period)

            try:
                result_data = load_pickle_result(file_path)
                if result_data is not None:
                    portfolio_returns = result_data.get('portfolio_cumulative_returns', [])
                    benchmark_returns = result_data.get('benchmark_cumulative_returns', [])
                    start_date = result_data.get('start_date', 'Unknown')
                    execution_time = result_data.get('total_run_time', 0)

                    # Check if portfolio_returns has data (handle NumPy arrays properly)
                    has_portfolio_data = False
                    if portfolio_returns is not None:
                        try:
                            has_portfolio_data = len(portfolio_returns) > 0
                        except TypeError:
                            # Handle case where portfolio_returns is not a sequence
                            has_portfolio_data = portfolio_returns is not None

                    if has_portfolio_data:
                        period_data.append({
                            'name': solver_name,
                            'portfolio_returns': portfolio_returns,
                            'benchmark_returns': benchmark_returns,
                            'start_date': start_date,
                            'execution_time': execution_time,
                            'file_path': file_path
                        })

                        # Store benchmark from first successful load
                        if benchmark_data is None and benchmark_returns is not None:
                            try:
                                if len(benchmark_returns) > 0:
                                    benchmark_data = benchmark_returns
                            except (TypeError, AttributeError):
                                pass

                        print(f'Period {period}: Loaded {solver_name} (exec_time: {execution_time:.2f}s)')
                    else:
                        print(f'Period {period}: {solver_name} has no portfolio returns data')
                else:
                    print(f'Period {period}: Could not load {solver_name} from {file_path}')

            except Exception as e:
                print(f'Period {period}: Error loading {solver_name} from {file_path}: {e}')

        # Plot results if we have any data
        if period_data:
            plt.figure(figsize=(15, 8))

            # Color palette for better distinction
            colors = plt.cm.Set1(np.linspace(0, 1, len(period_data)))

            # Plot portfolio returns for each solver
            for i, data in enumerate(period_data):
                plt.plot(data['portfolio_returns'],
                        label=f"{data['name']} (t={data['execution_time']:.1f}s)",
                        color=colors[i],
                        linewidth=2,
                        alpha=0.8)

            # Plot benchmark if available and requested
            if show_benchmark and benchmark_data is not None:
                plt.plot(benchmark_data,
                        label='S&P 500 Benchmark',
                        linestyle='--',
                        color='black',
                        linewidth=2,
                        alpha=0.7)

            plt.title(f'{title_prefix} Comparison - Period {period}\n'
                     f'Start Date: {period_data[0]["start_date"]} | Solvers: {len(period_data)}')
            plt.xlabel('Trading Days')
            plt.ylabel('Cumulative Returns')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # Print summary statistics
            print(f"\nPeriod {period} Summary:")
            print("-" * 50)
            for data in period_data:
                try:
                    final_return = data['portfolio_returns'][-1] if len(data['portfolio_returns']) > 0 else 0
                except (IndexError, TypeError):
                    final_return = 0
                print(f"{data['name']:<25}: {final_return:>8.4f} ({data['execution_time']:>6.2f}s)")
            if benchmark_data is not None:
                try:
                    benchmark_final = benchmark_data[-1] if len(benchmark_data) > 0 else 0
                except (IndexError, TypeError):
                    benchmark_final = 0
                print(f"{'Benchmark':<25}: {benchmark_final:>8.4f}")
            print()

        else:
            print(f'No data available for any solver in period {period}')


def multicomparison_barplots(stats_data, dataset_name):
    """
    Create side-by-side bar charts showing average returns and execution time for solvers.

    Args:
        stats_data (DataFrame): Performance statistics with columns 'Solver', 'Avg_Return', 'Avg_Time'
        dataset_name (str): Name of the dataset for the plot title
    """
    if stats_data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'{dataset_name} - Average Returns & Execution Time Summary', fontsize=16, fontweight='bold')

        # Define consistent color mapping for solvers using blue and orange hues
        solver_colors = {
            'Equal Weights': '#1E40AF',      # Deep blue
            'Genetic Algorithm': '#3B82F6',  # Medium blue
            'Scipy SLSQP': '#60A5FA',        # Light blue
            'Riskfolio QUBO': '#EA580C',     # Deep orange
            'D-Wave CQM': '#FB923C'          # Light orange
        }

        # Get sorted data
        stats_sorted = stats_data.sort_values('Avg_Return', ascending=False)
        stats_time = stats_data.sort_values('Avg_Time', ascending=True)

        # Get colors for each solver in sorted order
        colors_returns = [solver_colors.get(solver, '#2E8B57') for solver in stats_sorted['Solver']]
        colors_time = [solver_colors.get(solver, '#2E8B57') for solver in stats_time['Solver']]

        # Average Returns (left plot)
        ax1.bar(stats_sorted['Solver'], stats_sorted['Avg_Return'],
               color=colors_returns, alpha=0.8)
        ax1.set_xlabel('Solver', fontweight='bold')
        ax1.set_ylabel('Average Return', fontweight='bold')
        ax1.set_title('Average Returns by Solver', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (solver, value) in enumerate(zip(stats_sorted['Solver'], stats_sorted['Avg_Return'])):
            ax1.text(i, value + 0.003, f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Execution Time (right plot)
        ax2.bar(stats_time['Solver'], stats_time['Avg_Time'],
               color=colors_time, alpha=0.8)
        ax2.set_xlabel('Solver', fontweight='bold')
        ax2.set_ylabel('Average Execution Time (seconds)', fontweight='bold')
        ax2.set_title('Execution Time by Solver', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')

        # Add value labels
        for i, (solver, value) in enumerate(zip(stats_time['Solver'], stats_time['Avg_Time'])):
            ax2.text(i, value * 1.15, f'{value:.0f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.show()
    else:
        print(f"No valid data available for {dataset_name} multicomparison analysis")


def performance_summary_extended(solver_configs, n_periods):
    """Generate performance summary across all periods for multiple solvers."""
    summary_data = []

    for config in solver_configs:
        solver_name = config['name']
        path_template = config['path_template']

        returns_data = []
        execution_times = []
        successful_periods = 0

        for period in range(n_periods):
            file_path = path_template.format(period)
            try:
                result_data = load_pickle_result(file_path)
                if result_data is not None:
                    portfolio_returns = result_data.get('portfolio_cumulative_returns', [])
                    execution_time = result_data.get('total_run_time', 0)

                    # Safe check for portfolio returns data
                    try:
                        if portfolio_returns is not None and len(portfolio_returns) > 0:
                            final_return = portfolio_returns[-1]
                            returns_data.append(final_return)
                            execution_times.append(execution_time)
                            successful_periods += 1
                    except (TypeError, IndexError):
                        pass
            except Exception as e:
                print(f'Could not process {file_path}: {e}')

        if returns_data:
            summary_data.append({
                'Solver': solver_name,
                'Periods': successful_periods,
                'Avg_Return': np.mean(returns_data),
                'Std_Return': np.std(returns_data),
                'Min_Return': np.min(returns_data),
                'Max_Return': np.max(returns_data),
                'Avg_Time': np.mean(execution_times),
                'Std_Time': np.std(execution_times),
                'Success_Rate': successful_periods / n_periods * 100
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        df = df.round(4)
        return df
    else:
        return pd.DataFrame()