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
    n_assets = len(weights)
    
    # Ensure no negative weights
    weights = np.maximum(weights, 0)
    
    # If all weights are zero, start with equal weights
    if weights.sum() == 0:
        weights = np.ones(n_assets) / n_assets
        return weights
    
    # Ensure minimum allocation per asset
    weights = np.maximum(weights, min_weight_per_asset)
    
    # Normalize to sum to 1 (budget constraint)
    weights = weights / weights.sum()
    
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


def dwave_quantum_sharpe_minimize(data, budget=1.0, min_investment=0.001, risk_free_rate=0.0):
    """
    Pure D-Wave Quantum Non-Linear Programming optimization for risk-neutral Sharpe ratio.
    
    Uses ONLY D-Wave's quantum optimization solver with binary and continuous variables 
    for minimum investment constraints. NO classical fallbacks allowed.
    
    Constraints satisfied:
    - Budget: sum(investments) <= budget
    - Every asset allocated: minimum investment enforced via binary variables
    - Practical trading: if investing, must invest at least min_investment
    
    Args:
        data: Stock price data (DataFrame)
        budget: Total budget (default 1.0 for normalized weights)
        min_investment: Minimum investment per asset (practical trading constraint)
        risk_free_rate: Risk-free rate for Sharpe calculation
    
    Returns:
        weights: Optimized portfolio weights from quantum solver ONLY
        
    Raises:
        RuntimeError: If D-Wave quantum solver is not available or fails
    """
    # Prepare data
    returns = np.log(data) - np.log(data.shift(1))
    expected_annual_returns = returns.mean().values
    annual_covariance = returns.cov().values
    n_assets = len(expected_annual_returns)
    
    logger.info("Creating D-Wave quantum optimization model...")
    
    # Create Non-Linear Programming model
    model = Model()
    
    # Binary: whether to invest in each asset
    invest_binary = model.binary(n_assets)
    
    # Decision: scaled integer amounts to invest in each asset (scaled by 10000 for precision)
    # Since D-Wave optimization only supports integer variables, we scale up
    scale_factor = 10000
    max_scaled_investment = int(budget * scale_factor)
    
    # Use bounds enforcement with maximum() and minimum() functions
    investments = []
    zero_const = model.constant(0)
    max_const = model.constant(max_scaled_investment)
    
    for i in range(n_assets):
        # Create integer variable with bounds
        inv = model.integer(lower_bound=0, upper_bound=max_scaled_investment)
        # Apply bounds using maximum() and minimum() functions with model constants
        bounded_inv = maximum(inv, zero_const)  # Ensure non-negative
        bounded_inv = minimum(bounded_inv, max_const)  # Ensure upper bound
        investments.append(bounded_inv)
    
    # Budget constraint using add() function and model constants
    total_scaled_investment = add(*investments)
    budget_limit = model.constant(int(budget * scale_factor))
    model.add_constraint(total_scaled_investment <= budget_limit)
    
    # Minimum investment constraints using mathematical functions:
    min_scaled_investment = int(min_investment * scale_factor)
    max_scaled_investment_per_asset = int(budget * scale_factor)
    
    # Create model constants for constraint values
    min_scaled_const = model.constant(min_scaled_investment)
    max_scaled_const = model.constant(max_scaled_investment_per_asset)
    
    for i in range(n_assets):
        # If investing, must invest at least min_investment (transaction costs, etc.)
        # Use multiply() for constraint calculations
        min_constraint = multiply(min_scaled_const, invest_binary[i])
        model.add_constraint(investments[i] >= min_constraint)
        
        # If not investing, investment is 0
        max_constraint = multiply(max_scaled_const, invest_binary[i])
        model.add_constraint(investments[i] <= max_constraint)
    
    # Force investment in ALL assets using model constants
    one_constant = model.constant(1)
    for i in range(n_assets):
        model.add_constraint(invest_binary[i] == one_constant)
    
    # Calculate weights using D-Wave mathematical functions
    total_investment = add(*investments)  # Use add() for summing investments
    logger.debug(f"Total investment: {total_investment}")
    
    # Use safe_divide() for weight calculations to handle division by zero
    weights = [safe_divide(investments[i], total_investment) for i in range(n_assets)]
   
    # Portfolio return calculation using mathematical functions
    # Convert numpy arrays to model constants for proper type handling
    return_terms = []
    for i in range(n_assets):
        # Convert expected return to model constant
        expected_return_const = model.constant(expected_annual_returns[i])
        # Use multiply() for weight and return multiplication
        return_term = multiply(expected_return_const, weights[i])
        return_terms.append(return_term)
    
    portfolio_return = add(*return_terms)
    logger.debug(f'Portfolio return: {portfolio_return}')
    
    # Portfolio variance calculation using mathematical functions with enhanced stability
    variance_terms = []
    for i in range(n_assets):
        for j in range(n_assets):
            # Convert covariance matrix element to model constant
            cov_const = model.constant(annual_covariance[i][j])
            # Use multiply() for weight products and covariance multiplication
            cov_term = multiply(cov_const, weights[i])
            variance_term = multiply(cov_term, weights[j])
            variance_terms.append(variance_term)
    
    portfolio_variance = add(*variance_terms)
    
    # Apply maximum() to ensure positive variance with model constant
    min_variance_const = model.constant(1e-10)
    portfolio_variance = maximum(portfolio_variance, min_variance_const)
    
    # Use sqrt() from dwave.optimization.mathematical for portfolio standard deviation
    portfolio_std = sqrt(portfolio_variance)
    logger.debug(f'Portfolio std: {portfolio_std}')
    
    # Add constraint to ensure portfolio standard deviation is positive using maximum()
    min_std_const = model.constant(0.0001)  # Minimum allowed standard deviation
    portfolio_std_bounded = maximum(portfolio_std, min_std_const)
    model.add_constraint(portfolio_std >= min_std_const)
    
    # Calculate excess return using addition of negative risk-free rate (no subtract function available)
    # Convert risk_free_rate to model constant if it's a regular number
    if isinstance(risk_free_rate, (int, float)):
        risk_free_constant = model.constant(-risk_free_rate)  # Make negative for addition
    else:
        risk_free_constant = multiply(model.constant(-1), risk_free_rate)  # Make negative
        
    excess_return = add(portfolio_return, risk_free_constant)  # portfolio_return + (-risk_free_rate)
    sharpe_ratio = safe_divide(excess_return, portfolio_std_bounded)
    
    logger.info(f'D-Wave quantum optimization model created successfully')
    logger.info(f'Assets: {n_assets}, Scale factor: {scale_factor}')
    
    # Minimize negative Sharpe ratio (maximize Sharpe ratio) using multiplication by -1
    # Create negative Sharpe ratio by multiplying by -1
    neg_one_constant = model.constant(-1)
    negative_sharpe = multiply(neg_one_constant, sharpe_ratio)
    model.minimize(negative_sharpe)
    
    # Solve using D-Wave quantum solver - NO classical fallbacks
    logger.info("Solving with D-Wave quantum solver...")
    
    # Use D-Wave Cloud solver - this requires D-Wave access
    from dwave.cloud import Client
    from dwave.system import LeapHybridNonlinearProgramSampler
    
    # Connect to D-Wave Leap cloud service
    client = Client.from_config()
    logger.info("Connected to D-Wave quantum cloud service")
    
    # Use D-Wave's nonlinear program solver for the optimization model
    sampler = LeapHybridNonlinearProgramSampler()
    logger.info("Created D-Wave nonlinear program sampler")
    
    # Solve the optimization model using D-Wave's nonlinear solver
    logger.info("Submitting optimization model to D-Wave quantum cloud...")
    sampleset = sampler.sample(model, label="Sharpe_Ratio_Portfolio_Optimization")
    logger.info(f"Received {len(sampleset)} samples from D-Wave")
    
    # Get the best solution
    best_sample = sampleset.first
    logger.info(f"Best solution energy: {best_sample.energy}")
    logger.info(f"Best solution feasible: {best_sample.is_feasible}")
    
    # Extract the optimized weights from the solution
    optimized_weights = np.zeros(n_assets)
    for i in range(n_assets):
        # Get the scaled weight value and convert back to original scale
        scaled_weight = best_sample.sample[f'w_{i}']
        optimized_weights[i] = scaled_weight / scale_factor
        logger.debug(f"Asset {i}: scaled_weight={scaled_weight}, actual_weight={optimized_weights[i]}")
    
    # Normalize weights to ensure they sum to 1
    total_weight = optimized_weights.sum()
    final_weights = optimized_weights / total_weight
    logger.info(f"Optimized weights sum: {final_weights.sum():.6f}")
    logger.info(f"Weight distribution: min={final_weights.min():.4f}, max={final_weights.max():.4f}")
    
    return final_weights
        


def dwave_classical_sharpe(data, n_levels=5, sampler_type='simulated_annealing'):
    """
    Classical D-Wave QUBO optimization for risk-neutral Sharpe ratio.
    
    Uses BinaryQuadraticModel with discrete weight levels to approximate
    continuous Sharpe ratio optimization.
    
    Constraints satisfied:
    - Budget: weights sum to 1 through normalization
    - Every asset allocated: each asset gets one discrete level
    
    Args:
        data: Stock price data (DataFrame)
        n_levels: Number of discrete investment levels per asset
        sampler_type: Classical sampler type
    
    Returns:
        weights: Optimized portfolio weights
    """
    returns = np.log(data) - np.log(data.shift(1))
    avg_returns = returns.mean().fillna(0)
    cov_matrix = returns.cov().fillna(0)
    
    stocks = data.columns.tolist()
    n_stocks = len(stocks)
    
    # Select classical sampler
    samplers = {
        'simulated_annealing': SimulatedAnnealingSampler(),
        'tabu': TabuSampler(),
        'steepest_descent': SteepestDescentSampler(),
        'exact': ExactSolver()
    }
    
    sampler = samplers.get(sampler_type, SimulatedAnnealingSampler())
    
    # Create BQM for Sharpe ratio optimization
    bqm = BinaryQuadraticModel('BINARY')
    
    # Variables: each asset gets n_levels binary variables
    for i, stock in enumerate(stocks):
        for level in range(n_levels):
            var_name = f"{stock}_{level}"
            # Weight for this level
            weight = (level + 1) / (n_levels * n_stocks)
            # Expected return contribution (positive for maximization)
            return_contrib = avg_returns.iloc[i] * weight
            bqm.add_variable(var_name, return_contrib)  # Maximize returns
    
    # Risk terms (quadratic - minimize variance)
    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks):
            for level1 in range(n_levels):
                for level2 in range(n_levels):
                    var1 = f"{stock1}_{level1}"
                    var2 = f"{stock2}_{level2}"
                    
                    weight1 = (level1 + 1) / (n_levels * n_stocks)
                    weight2 = (level2 + 1) / (n_levels * n_stocks)
                    
                    # Risk penalty (negative to minimize risk)
                    risk_coeff = -0.5 * cov_matrix.iloc[i, j] * weight1 * weight2
                    
                    if i == j and level1 == level2:
                        bqm.add_variable(var1, risk_coeff)
                    else:
                        bqm.add_interaction(var1, var2, risk_coeff)
    
    # Constraint: each asset must have exactly one level selected
    penalty_strength = 100
    for stock in stocks:
        stock_vars = [f"{stock}_{level}" for level in range(n_levels)]
        
        # Penalty for not selecting exactly one level per asset
        for var in stock_vars:
            bqm.add_variable(var, penalty_strength * (-2))
        
        for var1 in stock_vars:
            for var2 in stock_vars:
                if var1 != var2:
                    bqm.add_interaction(var1, var2, penalty_strength * 2)
    
    # Solve
    if sampler_type == 'exact' and len(bqm.variables) > 20:
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=1000)
    else:
        num_reads = 1000 if sampler_type == 'simulated_annealing' else 100
        sampleset = sampler.sample(bqm, num_reads=num_reads)
    
    # Extract solution
    best_sample = sampleset.first.sample
    weights = np.zeros(n_stocks)
    
    for i, stock in enumerate(stocks):
        for level in range(n_levels):
            var_name = f"{stock}_{level}"
            if best_sample.get(var_name, 0) == 1:
                weights[i] = (level + 1) / (n_levels * n_stocks)
                break
    
    weights = ensure_valid_weights(weights)
    
    logger.debug(f'D-Wave Classical QUBO completed. Sharpe: {sharpe_fitness_function(weights, data):.4f}')
    return weights
       