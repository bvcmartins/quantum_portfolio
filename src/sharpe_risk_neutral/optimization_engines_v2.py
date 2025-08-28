import numpy as np
import scipy.optimize as optimize
import riskfolio as rp
import logging
from functools import partial
import gc
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, quicksum
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler
from dimod import ExactSolver
from itertools import product

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
            l=0              # No regularization
        )
        
        weights = np.array(weights_rf).flatten()
        weights = ensure_valid_weights(weights)
        
        logger.debug(f'Riskfolio Sharpe optimization completed. Sharpe: {sharpe_fitness_function(weights, data):.4f}')
        return weights
        
    except Exception as e:
        logger.warning(f'Riskfolio optimization failed: {e}. Using equal weights.')
        return equal_weights_baseline(data)


def dwave_cqm_sharpe_minimize(data, budget=1.0, min_investment=0.001):
    """
    D-Wave CQM optimization for risk-neutral Sharpe ratio.
    
    Uses ConstrainedQuadraticModel to maximize Sharpe ratio with explicit constraints.
    
    Constraints satisfied:
    - Budget: sum(investments) <= budget
    - Every asset allocated: min_investment constraint per asset
    
    Args:
        data: Stock price data (DataFrame)
        budget: Total budget (default 1.0 for normalized weights)
        min_investment: Minimum investment per asset
    
    Returns:
        weights: Optimized portfolio weights
    """
    
    try:
        n_assets = len(data.columns)
        returns = np.log(data) - np.log(data.shift(1))
        expected_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Create CQM model
        model = ConstrainedQuadraticModel()
        
        # Decision variables: investment amounts
        investments = [model.continuous(lower_bound=min_investment, upper_bound=budget) 
                      for _ in range(n_assets)]
        
        # Budget constraint
        model.add_constraint(sum(investments) <= budget)
        model.add_constraint(sum(investments) >= budget * 0.99)  # Ensure nearly full investment
        
        # Calculate portfolio return and variance
        total_investment = sum(investments)
        weights = [investments[i] / total_investment for i in range(n_assets)]
        
        portfolio_return = sum(expected_returns[i] * weights[i] for i in range(n_assets))
        portfolio_variance = sum(sum(cov_matrix[i][j] * weights[i] * weights[j] 
                                    for j in range(n_assets)) 
                               for i in range(n_assets))
        
        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        # Approximate sqrt for quadratic model: use variance instead of std dev
        # This optimizes return/variance which is monotonic with Sharpe for positive returns
        model.minimize(-portfolio_return + 0.001 * portfolio_variance)
        
        # Try to solve with D-Wave
        try:
            sampler = LeapHybridCQMSampler()
            sampleset = sampler.sample_cqm(model, label="Sharpe_CQM_Portfolio")
            
            if len(sampleset) > 0:
                best_sample = sampleset.first.sample
                investment_values = [best_sample[f'investments_{i}'] for i in range(n_assets)]
                weights = np.array(investment_values)
                weights = ensure_valid_weights(weights)
                
                logger.debug(f'D-Wave CQM completed. Sharpe: {sharpe_fitness_function(weights, data):.4f}')
                return weights
            else:
                logger.warning("D-Wave CQM returned no solutions. Using equal weights.")
                return equal_weights_baseline(data)
                
        except Exception as e:
            logger.warning(f"D-Wave CQM execution failed: {e}. Using equal weights.")
            return equal_weights_baseline(data)
            
    except Exception as e:
        logger.error(f'D-Wave CQM setup error: {e}. Using equal weights.')
        return equal_weights_baseline(data)


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
    try:
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
        
    except Exception as e:
        logger.error(f'D-Wave Classical QUBO error: {e}. Using equal weights.')
        return equal_weights_baseline(data)


def dwave_quantum_sharpe(data, n_levels=5):
    """
    Quantum D-Wave QUBO optimization for risk-neutral Sharpe ratio.
    
    Same formulation as classical version but uses quantum annealer.
    
    Constraints satisfied:
    - Budget: weights sum to 1 through normalization  
    - Every asset allocated: each asset gets one discrete level
    
    Args:
        data: Stock price data (DataFrame)
        n_levels: Number of discrete investment levels per asset
    
    Returns:
        weights: Optimized portfolio weights
    """
    try:
        # Same setup as classical version
        returns = np.log(data) - np.log(data.shift(1))
        avg_returns = returns.mean().fillna(0)
        cov_matrix = returns.cov().fillna(0)
        
        stocks = data.columns.tolist()
        n_stocks = len(stocks)
        
        # Try quantum sampler
        try:
            sampler = LeapHybridBQMSampler()
            logger.debug("Connected to D-Wave Leap quantum service")
        except Exception as e:
            logger.warning(f"D-Wave quantum connection failed: {e}. Using classical.")
            return dwave_classical_sharpe(data, n_levels, 'simulated_annealing')
        
        # Create same BQM as classical version
        bqm = BinaryQuadraticModel('BINARY')
        
        # Same variable and constraint setup as classical version
        for i, stock in enumerate(stocks):
            for level in range(n_levels):
                var_name = f"{stock}_{level}"
                weight = (level + 1) / (n_levels * n_stocks)
                return_contrib = avg_returns.iloc[i] * weight
                bqm.add_variable(var_name, return_contrib)
        
        # Risk terms
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                for level1 in range(n_levels):
                    for level2 in range(n_levels):
                        var1 = f"{stock1}_{level1}"
                        var2 = f"{stock2}_{level2}"
                        
                        weight1 = (level1 + 1) / (n_levels * n_stocks)
                        weight2 = (level2 + 1) / (n_levels * n_stocks)
                        risk_coeff = -0.5 * cov_matrix.iloc[i, j] * weight1 * weight2
                        
                        if i == j and level1 == level2:
                            bqm.add_variable(var1, risk_coeff)
                        else:
                            bqm.add_interaction(var1, var2, risk_coeff)
        
        # Constraints
        penalty_strength = 100
        for stock in stocks:
            stock_vars = [f"{stock}_{level}" for level in range(n_levels)]
            
            for var in stock_vars:
                bqm.add_variable(var, penalty_strength * (-2))
            
            for var1 in stock_vars:
                for var2 in stock_vars:
                    if var1 != var2:
                        bqm.add_interaction(var1, var2, penalty_strength * 2)
        
        # Quantum solve
        sampleset = sampler.sample(bqm, label=f"Quantum_Sharpe_Portfolio_{n_stocks}assets")
        
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
        
        logger.debug(f'D-Wave Quantum completed. Sharpe: {sharpe_fitness_function(weights, data):.4f}')
        return weights
        
    except Exception as e:
        logger.error(f'D-Wave Quantum error: {e}. Using classical QUBO.')
        return dwave_classical_sharpe(data, n_levels, 'simulated_annealing')
