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
            return x0
            
    except Exception as e:
        logger.error(f"SciPy SLSQP optimization error: {e}")
        return x0


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
            hist=True
        )
        
        if weights is not None and len(weights) > 0:
            return weights.values.flatten()
        else:
            logger.warning("Riskfolio optimization returned empty result")
            n_assets = len(data.columns)
            return np.full(n_assets, 1.0 / n_assets)
            
    except Exception as e:
        logger.error(f"Riskfolio optimization error: {e}")
        n_assets = len(data.columns)
        return np.full(n_assets, 1.0 / n_assets)


def dwave_bqm_qubo(data, budget=1.0, min_investment=0.001):
    """
    D-Wave BQM (Binary Quadratic Model) for QUBO portfolio optimization.
    
    Uses fixed risk level of 0.5 in the QUBO formulation.
    
    Args:
        data: Stock price DataFrame
        budget: Total budget (default 1.0 for normalized weights)
        min_investment: Minimum investment per asset
        
    Returns:
        weights: Optimized portfolio weights
    """
    try:
        # Prepare data
        returns = np.log(data) - np.log(data.shift(1))
        expected_returns = returns.mean().values
        covariance_matrix = returns.cov().values
        n_assets = len(expected_returns)
        
        # QUBO formulation parameters
        risk_level = 0.5
        precision = 100  # Scale factor for integer weights
        
        # Create BQM
        bqm = BinaryQuadraticModel({}, {}, 0.0, 'BINARY')
        
        # Create binary variables for each possible investment level per asset
        # Each asset can have investment levels: 0, min_investment, 2*min_investment, ..., budget
        max_investment_units = int(budget / min_investment)
        
        # Variables: x[asset][level] = 1 if asset has investment level
        variables = {}
        for i in range(n_assets):
            for level in range(max_investment_units + 1):
                var_name = f'x_{i}_{level}'
                variables[(i, level)] = var_name
                bqm.add_variable(var_name, 0.0)
        
        # Constraint: exactly one investment level per asset
        for i in range(n_assets):
            asset_vars = [variables[(i, level)] for level in range(max_investment_units + 1)]
            # Add penalty for not selecting exactly one level
            penalty = 1000.0
            for j, var1 in enumerate(asset_vars):
                for k, var2 in enumerate(asset_vars):
                    if j == k:
                        bqm.add_variable(var1, -penalty)  # Reward selecting one
                    else:
                        bqm.add_interaction(var1, var2, penalty)  # Penalize selecting multiple
        
        # Budget constraint
        budget_penalty = 1000.0
        total_investment = 0
        for i in range(n_assets):
            for level in range(1, max_investment_units + 1):  # Skip level 0 (no investment)
                investment_amount = level * min_investment
                var_name = variables[(i, level)]
                total_investment += investment_amount
                # Penalty for exceeding budget
                if total_investment > budget:
                    bqm.add_variable(var_name, budget_penalty * (total_investment - budget))
        
        # Objective: maximize return - risk_level * variance
        for i in range(n_assets):
            for level in range(1, max_investment_units + 1):
                weight = (level * min_investment) / budget
                var_name = variables[(i, level)]
                # Add return component (negative because BQM minimizes)
                bqm.add_variable(var_name, -expected_returns[i] * weight * precision)
                
                # Add risk component (variance)
                for j in range(n_assets):
                    for level2 in range(1, max_investment_units + 1):
                        weight2 = (level2 * min_investment) / budget
                        var_name2 = variables[(j, level2)]
                        risk_term = risk_level * covariance_matrix[i][j] * weight * weight2 * precision
                        if i == j:
                            bqm.add_variable(var_name, risk_term)
                        else:
                            bqm.add_interaction(var_name, var_name2, risk_term / 2)
        
        # Solve with D-Wave
        try:
            sampler = LeapHybridBQMSampler()
            sampleset = sampler.sample(bqm, label="QUBO_Portfolio")
            
            if len(sampleset) > 0:
                best_sample = sampleset.first.sample
                
                # Extract weights from solution
                weights = np.zeros(n_assets)
                for i in range(n_assets):
                    for level in range(max_investment_units + 1):
                        var_name = variables[(i, level)]
                        if best_sample.get(var_name, 0) == 1:
                            weights[i] = (level * min_investment) / budget
                            break
                
                # Normalize weights
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    return weights
        
        except Exception as e:
            logger.warning(f"D-Wave BQM solver error: {e}")
        
        # Fallback to equal weights
        logger.warning("D-Wave BQM optimization failed. Using equal weights.")
        return np.full(n_assets, 1.0 / n_assets)
        
    except Exception as e:
        logger.error(f"D-Wave BQM setup error: {e}")
        n_assets = len(data.columns)
        return np.full(n_assets, 1.0 / n_assets)


def dwave_cqm_qubo(data, budget=1.0, min_investment=0.001):
    """
    D-Wave CQM (Constrained Quadratic Model) for QUBO portfolio optimization.
    
    Uses fixed risk level of 0.5 in the objective function.
    
    Args:
        data: Stock price DataFrame
        budget: Total budget (default 1.0 for normalized weights)
        min_investment: Minimum investment per asset
        
    Returns:
        weights: Optimized portfolio weights
    """
    try:
        # Prepare data
        returns = np.log(data) - np.log(data.shift(1))
        expected_returns = returns.mean().values
        covariance_matrix = returns.cov().values
        n_assets = len(expected_returns)
        
        # Create CQM
        cqm = ConstrainedQuadraticModel()
        
        # Binary variables: whether to invest in each asset
        invest_binary = [Binary(f'invest_{i}') for i in range(n_assets)]
        
        # Integer variables: investment amounts (scaled by 1000 for precision)
        scale_factor = 1000
        max_investment = int(budget * scale_factor)
        investments = []
        for i in range(n_assets):
            var_name = f'investment_{i}'
            investments.append(var_name)
            cqm.add_variable('INTEGER', var_name, lower_bound=0, upper_bound=max_investment)
        
        # Budget constraint
        budget_constraint = quicksum(investments) <= int(budget * scale_factor)
        cqm.add_constraint(budget_constraint, label='budget')
        
        # Minimum investment constraints
        min_scaled_investment = int(min_investment * scale_factor)
        for i in range(n_assets):
            # If investing, must invest at least min_investment
            min_constraint = investments[i] >= min_scaled_investment * invest_binary[i]
            cqm.add_constraint(min_constraint, label=f'min_investment_{i}')
            
            # If not investing, investment is 0
            max_constraint = investments[i] <= max_investment * invest_binary[i]
            cqm.add_constraint(max_constraint, label=f'max_investment_{i}')
        
        # Force investment in all assets (every asset gets allocated)
        for i in range(n_assets):
            cqm.add_constraint(invest_binary[i] == 1, label=f'force_invest_{i}')
        
        # QUBO Objective: maximize return - risk_level * variance
        risk_level = 0.5
        objective = 0
        
        # Return component
        total_investment = quicksum(investments)
        for i in range(n_assets):
            weight = investments[i] / total_investment
            objective += expected_returns[i] * weight
        
        # Risk component (variance)
        for i in range(n_assets):
            for j in range(n_assets):
                weight_i = investments[i] / total_investment
                weight_j = investments[j] / total_investment
                objective -= risk_level * covariance_matrix[i][j] * weight_i * weight_j
        
        # Set objective (CQM minimizes, so negate for maximization)
        cqm.set_objective(-objective)
        
        # Solve with D-Wave
        try:
            sampler = LeapHybridCQMSampler()
            sampleset = sampler.sample_cqm(cqm, label="QUBO_CQM_Portfolio")
            
            if len(sampleset) > 0:
                best_sample = sampleset.first.sample
                
                # Extract investment amounts
                investment_amounts = np.zeros(n_assets)
                for i in range(n_assets):
                    investment_amounts[i] = best_sample.get(f'investment_{i}', 0) / scale_factor
                
                # Convert to weights
                total = investment_amounts.sum()
                if total > 0:
                    weights = investment_amounts / total
                    return weights
        
        except Exception as e:
            logger.warning(f"D-Wave CQM solver error: {e}")
        
        # Fallback to equal weights
        logger.warning("D-Wave CQM optimization failed. Using equal weights.")
        return np.full(n_assets, 1.0 / n_assets)
        
    except Exception as e:
        logger.error(f"D-Wave CQM setup error: {e}")
        n_assets = len(data.columns)
        return np.full(n_assets, 1.0 / n_assets)