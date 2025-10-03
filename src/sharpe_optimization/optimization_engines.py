import numpy as np
import scipy.optimize as optimize
import riskfolio as rp
import logging
from functools import partial
import gc
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, quicksum
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler, LeapHybridSampler
from dimod import ExactSolver
from dwave.optimization import Model
from dwave.system.samplers import LeapHybridNLSampler
from itertools import product
from dwave.optimization.mathematical import (
    maximum, minimum, sqrt, safe_divide, multiply, add,
    exp, log, absolute, where, logical_and, logical_or
)

seed = 12
np.random.seed(seed)
logger = logging.getLogger("inspect_results_logger")


def portfolio_stats(weights, data):
    """Calculate portfolio statistics (Sharpe ratio, return, volatility)"""
    weights = np.array(weights)
    returns = np.log(data) - np.log(data.shift(1))  # log return to minimize fp error
    port_return = np.sum(returns.mean() * weights) 
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    try:
        sharpe_ratio = port_return / port_vol
    except Exception as e:
        sharpe_ratio = 0
    return sharpe_ratio, port_return, port_vol


def sharpe_fitness_function(weights, data):
    """
    Risk-neutral Sharpe ratio fitness function.
    
    Standard Sharpe ratio maximization (risk-free rate = 0 for simplicity).
    All optimizers will maximize this same objective.
    
    Args:
        weights: Portfolio weights
        data: Stock price data
    
    Returns:
        Sharpe ratio (for maximization)
    """
    sharpe_ratio, _, _ = portfolio_stats(weights, data)
    return sharpe_ratio


def ensure_valid_weights(weights, min_weight_per_asset=0.001):
    """
    Ensure weights satisfy constraints:
    1. Sum to 1 (budget constraint)
    2. Every asset has positive allocation (no zero weights)
    
    Args:
        weights: Raw portfolio weights
        min_weight_per_asset: Minimum weight per asset to ensure allocation
    
    Returns:
        Adjusted weights satisfying all constraints
    """
    weights = np.array(weights)
    #n_assets = len(weights)
    
    # Ensure no negative weights
    weights = np.maximum(weights, 0)
    
    # # If all weights are zero, start with equal weights
    # if weights.sum() == 0:
    #     weights = np.ones(n_assets) / n_assets
    #     return weights
    
    # Ensure minimum allocation per asset
    weights = np.maximum(weights, min_weight_per_asset)
    
    # Normalize to sum to 1 (budget constraint)
    weights = weights / weights.sum()
    weights = np.array(weights)
    
    return weights


def equal_weights_baseline(data):
    """
    Equal weights baseline model for portfolio optimization.
    
    Constraints satisfied:
    - Budget: weights sum to 1
    - Every asset allocated: 1/N weight each
    
    Args:
        data: Stock price data (DataFrame)
    
    Returns:
        weights: Equal weights portfolio as numpy array
    """
    n_assets = len(data.columns)
    weights = np.ones(n_assets) / n_assets
    
    logger.debug(f"Equal weights baseline: {n_assets} assets, weight={1/n_assets:.6f} each")
    return weights


def genetic_algorithm_sharpe(data, population_size=500, num_generations=1000, mutation_rate=0.05, elitism=0.1):
    """
    Genetic algorithm optimizing risk-neutral Sharpe ratio.
    
    Constraints satisfied:
    - Budget: weights normalized to sum to 1 each generation
    - Every asset allocated: minimum weight enforcement
    
    Args:
        data: Stock price data (DataFrame)
        population_size: Size of population
        num_generations: Number of generations
        mutation_rate: Mutation rate
        elitism: Elite preservation rate
    
    Returns:
        weights: Optimized portfolio weights
    """
    n_assets = len(data.columns)
    
    # Initialize population with valid weights
    population = np.random.rand(population_size, n_assets)
    for i in range(population_size):
        population[i] = ensure_valid_weights(population[i])
    
    # Evaluate fitness
    fitness = np.array([sharpe_fitness_function(individual, data) for individual in population])
    
    for generation in range(num_generations):
        # Selection: sort by fitness (descending for maximization)
        sorted_idx = np.argsort(fitness)[::-1]
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Elitism: keep best individuals
        num_elites = int(elitism * population_size)
        new_population = population[:num_elites].copy()
        
        # Generate offspring
        for i in range(num_elites, population_size):
            # Tournament selection
            parent1_idx = np.random.randint(0, min(num_elites * 2, population_size))
            parent2_idx = np.random.randint(0, min(num_elites * 2, population_size))
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            crossover_point = np.random.randint(1, n_assets)
            offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Mutation
            if np.random.rand() < mutation_rate:
                mutation_idx = np.random.randint(0, n_assets)
                offspring[mutation_idx] += np.random.normal(0, 0.01)
            
            # Ensure constraints
            offspring = ensure_valid_weights(offspring)
            new_population = np.vstack([new_population, offspring])
        
        population = new_population
        fitness = np.array([sharpe_fitness_function(individual, data) for individual in population])
    
    # Return best individual
    best_idx = np.argmax(fitness)
    best_weights = population[best_idx]
    
    logger.debug(f'GA Sharpe optimization completed. Best Sharpe: {fitness[best_idx]:.4f}')
    return ensure_valid_weights(best_weights)


def scipy_minimize_sharpe(data):
    """
    SciPy SLSQP optimization for risk-neutral Sharpe ratio.
    
    Constraints satisfied:
    - Budget: equality constraint sum(weights) = 1
    - Every asset allocated: lower bound > 0 for all assets
    
    Args:
        data: Stock price data (DataFrame)
    
    Returns:
        weights: Optimized portfolio weights
    """
    n_assets = len(data.columns)
    min_weight = 0.001  # Minimum allocation per asset
    
    # Objective: minimize negative Sharpe (to maximize Sharpe)
    def objective(weights):
        return -sharpe_fitness_function(weights, data)
    
    # Constraints
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bounds: minimum allocation per asset, max 50% per asset for diversification
    bounds = tuple((min_weight, 0.5) for _ in range(n_assets))
    
    # Initial guess: equal weights
    x0 = np.ones(n_assets) / n_assets
    
    # Optimization
    try:
        result = optimize.minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            weights = ensure_valid_weights(weights)
            logger.debug(f'SciPy SLSQP completed successfully. Sharpe: {sharpe_fitness_function(weights, data):.4f}')
            return weights
        else:
            logger.warning(f'SciPy SLSQP failed: {result.message}. Using equal weights.')
            return equal_weights_baseline(data)
            
    except Exception as e:
        logger.error(f'SciPy SLSQP error: {e}. Using equal weights.')
        return equal_weights_baseline(data)


def riskfolio_sharpe_optimized(data):
    """
    Modified Riskfolio approach optimized for Sharpe ratio.
    
    Uses mean-variance optimization from riskfolio-lib instead of HRP
    to optimize Sharpe ratio while satisfying constraints.
    
    Constraints satisfied:
    - Budget: built into riskfolio optimization
    - Every asset allocated: post-processing to ensure minimum weights
    
    Args:
        data: Stock price data (DataFrame)
    
    Returns:
        weights: Optimized portfolio weights
    """
    try:
        # Calculate returns
        returns = np.log(data) - np.log(data.shift(1))
        returns = returns.dropna()
        
        # Create portfolio object
        port = rp.Portfolio(returns=returns)
        
        # Estimate expected returns and covariance
        port.assets_stats(method_mu='hist', method_cov='hist')
        
        # Optimize for maximum Sharpe ratio (mean-variance optimization)
        weights_rf = port.optimization(
            model='Classic',  # Classic mean-variance
            rm='MV',         # Mean-Variance risk measure
            obj='Sharpe',    # Maximize Sharpe ratio
            rf=0.0,          # Risk-free rate
            l=0,              # No regularization,
            solver='MOSEK'
        )
        
        weights = np.array(weights_rf).flatten()
        weights = ensure_valid_weights(weights)
        
        logger.debug(f'Riskfolio Sharpe optimization completed. Sharpe: {sharpe_fitness_function(weights, data):.4f}')
        return weights
        
    except Exception as e:
        logger.warning(f'Riskfolio optimization failed: {e}. Using equal weights.')
        return equal_weights_baseline(data)


def dwave_nl_sharpe(data, budget, min_investment):
    print("="*80)
    print("D-Wave NL Sharpe Optimization Started")
    print("="*80)

    # Input validation and logging
    print(f"Input Parameters:")
    print(f"  - Budget: {budget}")
    print(f"  - Min Investment: {min_investment}")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Assets: {list(data.columns)}")

    # Calculate returns and statistics
    print("Calculating returns and portfolio statistics...")
    returns = np.log(data) - np.log(data.shift(1))
    expected_annual_returns = returns.mean().values
    annual_covariance = returns.cov().values

    print(f"Expected Annual Returns:")
    for i, (col, ret) in enumerate(zip(data.columns, expected_annual_returns)):
        print(f"  - {col}: {ret:.6f}")

    print(f"Covariance Matrix shape: {annual_covariance.shape}")
    logger.debug(f"Covariance Matrix:\n{annual_covariance}")

    # Model initialization
    print("Initializing D-Wave Nonlinear Model...")
    model = Model()

    n_cols = data.shape[1]
    print(f"Number of assets: {n_cols}")

    # Create continuous decision variables for investments
    # Scale up for integer representation (D-Wave optimization uses integers)
    # Use scale factor that ensures min_investment_scaled >= 1
    # Adjust scale factor based on budget to avoid exceeding D-Wave integer limits
    # D-Wave integer variables have a practical limit around 10^9
    max_safe_value = 1e9  # Safe upper bound for D-Wave integers
    scale_factor = min(1000, int(max_safe_value / budget))
    scale_factor = max(100, scale_factor)  # Ensure minimum scale factor of 100
    max_scaled_investment = int(budget * scale_factor)

    print(f"Scaling Parameters:")
    print(f"  - Scale factor: {scale_factor}")
    print(f"  - Max scaled investment: {max_scaled_investment}")
    print(f"  - Budget scaled: {int(budget * scale_factor)}")
    print(f"  - Min investment scaled: {max(1, int(min_investment * scale_factor))}")

    # Integer investment amounts for each asset
    print("Creating decision variables...")
    investments = [model.integer(lower_bound=0, upper_bound=max_scaled_investment)
                   for _ in range(n_cols)]
    print(f"  - Created {n_cols} integer investment variables")

    # Constants
    budget_scaled = model.constant(int(budget * scale_factor))
    min_investment_scaled = model.constant(max(1, int(min_investment * scale_factor)))
    max_investment_scaled = model.constant(max_scaled_investment)
    one_const = model.constant(1)
    zero_const = model.constant(0)
    min_std_const = model.constant(0.0001)
    min_var_const = model.constant(1e-10)
    neg_one_const = model.constant(-1)
    print("  - Created model constants")

    # Budget constraint: 0.95 * budget <= sum of investments <= budget
    print("Adding constraints...")
    total_investment = add(*investments)
    budget_lower_bound = model.constant(int(budget * scale_factor * 0.95))
    model.add_constraint(total_investment >= budget_lower_bound)
    model.add_constraint(total_investment <= budget_scaled)

    # Minimum investment constraints - ensure all weights > 0
    constraint_count = 0
    for i in range(n_cols):
        # Each investment must be at least min_investment
        model.add_constraint(investments[i] >= min_investment_scaled)
        constraint_count += 1

    print(f"  - Added {constraint_count} minimum investment constraints (all assets > {min_investment})")

    # Calculate weights from investments
    print("Constructing objective function...")
    weights = [safe_divide(investments[i], total_investment) for i in range(n_cols)]
    print("  - Calculated weight variables from investments")

    # Portfolio expected return
    return_terms = []
    for i in range(n_cols):
        expected_return_const = model.constant(expected_annual_returns[i])
        return_term = multiply(expected_return_const, weights[i])
        return_terms.append(return_term)
    portfolio_return = add(*return_terms)
    print("  - Constructed portfolio return expression")

    # Portfolio variance
    variance_terms = []
    for i in range(n_cols):
        for j in range(n_cols):
            cov_const = model.constant(annual_covariance[i, j])
            cov_term = multiply(cov_const, weights[i])
            variance_term = multiply(cov_term, weights[j])
            variance_terms.append(variance_term)
    portfolio_variance = add(*variance_terms)
    print(f"  - Constructed portfolio variance expression ({len(variance_terms)} terms)")

    # Ensure positive variance
    portfolio_variance = maximum(portfolio_variance, min_var_const)
    print(f"  - Applied minimum variance bound: {min_var_const}")

    # Portfolio standard deviation
    portfolio_std = sqrt(portfolio_variance)
    print("  - Calculated portfolio standard deviation")

    # Ensure minimum std for numerical stability
    portfolio_std_bounded = maximum(portfolio_std, min_std_const)
    model.add_constraint(portfolio_std >= min_std_const)
    print(f"  - Applied minimum std constraint: {min_std_const}")

    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = safe_divide(portfolio_return, portfolio_std_bounded)
    print("  - Constructed Sharpe ratio objective")

    # Minimize negative Sharpe (maximize Sharpe)
    negative_sharpe = multiply(neg_one_const, sharpe_ratio)
    model.minimize(negative_sharpe)
    print("  - Set objective: minimize(-Sharpe) to maximize Sharpe")

    # Model summary
    print(f"\nModel Summary:")
    print(f"  - Decision variables: {n_cols} integer investments")
    print(f"  - Total constraints: {1 + constraint_count} (1 budget + {constraint_count} min investment)")
    print(f"  - Objective: Maximize Sharpe Ratio")

    print("\nSubmitting to D-Wave LeapHybridNLSampler...")
    sampler = LeapHybridNLSampler()
    print("  - Sampler initialized")

    sampleset = sampler.sample(model, label="Sharpe_Portfolio_Optimization").result()
    print("  - Sampling complete")

    # Extract best solution from the model's state
    # The solution is stored in the model after sampling
    print("\nExtracting solution...")
    optimized_investments = np.array([investments[i].state() for i in range(n_cols)])
    print(f"Optimized Investments (scaled):")
    for i, (col, inv) in enumerate(zip(data.columns, optimized_investments)):
        print(f"  - {col}: {inv} (unscaled: {inv/scale_factor:.4f})")

    optimized_weights = optimized_investments / optimized_investments.sum()
    print(f"\nOptimized Weights:")
    print(optimized_weights)
    # for i, (col, w) in enumerate(zip(data.columns, optimized_weights)):
    #     print(f"  - {col}: {w:.6f} ({w*100:.2f}%)")

    # # Calculate and log final portfolio statistics
    # final_sharpe, final_return, final_vol = portfolio_stats(optimized_weights, data)
    # print(f"\nFinal Portfolio Statistics:")
    # print(f"  - Sharpe Ratio: {final_sharpe:.6f}")
    # print(f"  - Expected Return: {final_return:.6f}")
    # print(f"  - Volatility (Std Dev): {final_vol:.6f}")

    # # Validation checks
    # print(f"\nValidation Checks:")
    # print(f"  - Sum of weights: {optimized_weights.sum():.6f} (should be ~1.0)")
    # print(f"  - Min weight: {optimized_weights.min():.6f} (should be >= {min_investment:.6f})")
    # print(f"  - Max weight: {optimized_weights.max():.6f}")
    # print(f"  - All weights positive: {np.all(optimized_weights > 0)}")

    print("="*80)
    print("D-Wave NL Sharpe Optimization Completed Successfully")
    print("="*80)


    return optimized_weights

       