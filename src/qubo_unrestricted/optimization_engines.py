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
from dimod import Integer
from itertools import product

seed = 12
np.random.seed(seed)
logger = logging.getLogger("inspect_results_logger")


def portfolio_stats(weights, data):
    """Calculate portfolio statistics (return, volatility)"""
    weights = np.array(weights)
    returns = np.log(data) - np.log(data.shift(1))  # log return to minimize fp error
    port_return = np.sum(returns.mean() * weights) 
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    return port_return, port_vol


def qubo_fitness_function(weights, data, risk_level=0.5):
    """
    QUBO fitness function with fixed risk level.
    
    Maximizes: return - risk_level * variance
    Fixed risk level of 0.5 for consistent comparison across QUBO optimizers.
    
    Args:
        weights: Portfolio weights
        data: Stock price data
        risk_level: Risk aversion parameter (fixed at 0.5)
    
    Returns:
        Objective value (return - risk * variance) for maximization
    """
    port_return, port_vol = portfolio_stats(weights, data)
    portfolio_variance = port_vol ** 2
    objective = port_return - risk_level * portfolio_variance
    return objective


def equal_weights_baseline(data):
    """
    Equal weights baseline portfolio.
    
    Args:
        data: Stock price DataFrame
        
    Returns:
        weights: Equal weight allocation
    """
    n_assets = len(data.columns)
    return np.full(n_assets, 1.0 / n_assets)


def genetic_algorithm_qubo(data, population_size=500, num_generations=100, mutation_rate=0.1, elitism=0.1):
    """
    Genetic Algorithm for QUBO portfolio optimization.
    
    Uses fixed risk level of 0.5 in the objective function.
    
    Args:
        data: Stock price DataFrame
        population_size: Number of individuals in each generation
        num_generations: Number of generations to evolve
        mutation_rate: Probability of mutation for each gene
        elitism: Fraction of best individuals to preserve
        
    Returns:
        weights: Optimized portfolio weights
    """
    n_assets = len(data.columns)
    n_elite = int(population_size * elitism)
    
    # Initialize population with normalized random weights
    population = []
    for _ in range(population_size):
        weights = np.random.rand(n_assets)
        weights = weights / weights.sum()  # Normalize to sum to 1
        population.append(weights)
    
    def fitness_function(weights):
        return qubo_fitness_function(weights, data, risk_level=0.5)
    
    for generation in range(num_generations):
        # Calculate fitness for all individuals
        fitness_scores = [fitness_function(individual) for individual in population]
        
        # Sort population by fitness (descending - higher is better)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Elite selection
        new_population = population[:n_elite].copy()
        
        # Fill rest of population through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.choice(min(50, len(population)))
            parent2_idx = np.random.choice(min(50, len(population)))
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Uniform crossover
            child = np.where(np.random.rand(n_assets) < 0.5, parent1, parent2)
            
            # Mutation
            if np.random.rand() < mutation_rate:
                mutation_strength = 0.1
                noise = np.random.normal(0, mutation_strength, n_assets)
                child = child + noise
                child = np.abs(child)  # Ensure non-negative
            
            # Normalize weights to sum to 1
            child = child / child.sum() if child.sum() > 0 else np.full(n_assets, 1.0/n_assets)
            new_population.append(child)
        
        population = new_population
    
    # Return best individual
    final_fitness = [fitness_function(individual) for individual in population]
    best_idx = np.argmax(final_fitness)
    return population[best_idx]


def scipy_slsqp_qubo(data):
    """
    SciPy SLSQP optimizer for QUBO portfolio optimization.
    
    Uses fixed risk level of 0.5 in the objective function.
    
    Args:
        data: Stock price DataFrame
        
    Returns:
        weights: Optimized portfolio weights
    """
    n_assets = len(data.columns)
    
    def objective_function(weights):
        # Minimize negative objective (maximize objective)
        return -qubo_fitness_function(weights, data, risk_level=0.5)
    
    # Constraints: weights sum to 1, all weights >= 0
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Budget constraint
    ]
    
    bounds = [(0.001, 1.0) for _ in range(n_assets)]  # Min investment constraint
    
    # Initial guess (equal weights)
    x0 = np.full(n_assets, 1.0 / n_assets)
    
    try:
        result = optimize.minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning(f"SciPy optimization failed: {result.message}")
            return np.full(n_assets, np.nan)

    except Exception as e:
        logger.error(f"SciPy SLSQP optimization error: {e}")
        return np.full(n_assets, np.nan)


def riskfolio_qubo(data):
    """
    Riskfolio-lib optimizer for QUBO portfolio optimization.
    
    Uses mean-variance optimization with fixed risk aversion.
    
    Args:
        data: Stock price DataFrame
        
    Returns:
        weights: Optimized portfolio weights
    """
    try:
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Create Portfolio object
        port = rp.Portfolio(returns=returns)
        
        # Calculate risk and return parameters
        port.assets_stats(method_mu='hist', method_cov='hist')
        
        # Fixed risk aversion parameter (equivalent to risk_level=0.5)
        # In Riskfolio, risk aversion (rm) controls the risk-return tradeoff
        risk_aversion = 1.0  # Corresponds roughly to risk_level=0.5
        
        # Mean-variance optimization
        weights = port.optimization(
            model='Classic',
            rm='MV',  # Mean Variance
            obj='Utility',  # Utility maximization (return - risk_aversion * risk)
            rf=0.0,  # Risk-free rate
            l=risk_aversion,  # Risk aversion parameter
            hist=True,
            solver='CLARABEL'  # Free open-source solver
        )
        
        if weights is not None and len(weights) > 0:
            return weights.values.flatten()
        else:
            logger.warning("Riskfolio optimization returned empty result")
            n_assets = len(data.columns)
            return np.full(n_assets, np.nan)

    except Exception as e:
        logger.error(f"Riskfolio optimization error: {e}")
        n_assets = len(data.columns)
        return np.full(n_assets, np.nan)


def dwave_cqm_qubo(data, budget=1000000.0):
    """
    D-Wave CQM (Constrained Quadratic Model) for QUBO portfolio optimization.
    
    Uses fixed risk level of 0.5 in the objective function.
    
    Args:
        data: Stock price DataFrame
        budget: Total budget (default 1.0 for normalized weights)
        
    Returns:
        weights: Optimized portfolio weights
    """
    alpha = 0.5
    returns = np.log(data) - np.log(data.shift(1))
    expected_returns = returns.mean()
    covariance_matrix = returns.cov()
    cqm = ConstrainedQuadraticModel()
    stocks = data.columns.tolist()
    price = data.iloc[-1, :]
    #print(f'price: {price}')
    #print(f'budget: {budget}')
    max_num_shares = (budget / price).astype(int)
    #print('max_num_shares')
    #print(max_num_shares)
    x = {s: Integer("%s" %s, lower_bound=1, upper_bound=max_num_shares[s]) for s in stocks}

    returns = 0
    for s in stocks:
        returns = returns + price[s] * expected_returns[s] * x[s]

    risk = 0
    for s1, s2 in product(stocks, stocks):
        coeff = covariance_matrix[s1][s2] * price[s1] * price[s2]
        risk = risk + coeff * x[s1] * x[s2]

    # Budget constraints: 0.95 * budget <= total investment <= budget
    total_investment = quicksum([x[s] * price[s] for s in stocks])
    cqm.add_constraint(total_investment <= budget, label='upper_budget')
    cqm.add_constraint(total_investment >= 0.95 * budget, label='lower_budget')

    cqm.set_objective(alpha * risk - returns)
    #cqm.substitute_self_loops()

    sampler = LeapHybridCQMSampler()
    results = sampler.sample_cqm(cqm, time_limit=20, label="QUBO_Portfolio_Optimization")
    n_samples = len(results.record)
    logger.info(f'n samples: {n_samples}')
    feasible_samples = results.filter(lambda d: d.is_feasible)

    # Check if any feasible samples were found
    if len(feasible_samples) == 0:
        logger.warning("No feasible samples found by D-Wave CQM solver")
        n_assets = len(stocks)
        return np.full(n_assets, np.nan)

    best_sample = feasible_samples.first
    amounts = []
    for s in stocks:
        amounts.append(best_sample.sample[s])
    total = sum(amounts)
    weights = amounts / total
    return weights  