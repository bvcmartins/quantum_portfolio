import numpy as np
import scipy.optimize as optimize
import riskfolio as rp
import logging
from functools import partial

# D-Wave imports for QUBO optimization
try:
    import gc
    from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, quicksum
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
    from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler
    from dimod import ExactSolver
    from itertools import product
    DWAVE_AVAILABLE = True
    DWAVE_QUANTUM_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    DWAVE_QUANTUM_AVAILABLE = False
    print("D-Wave Ocean SDK not available. Install with: pip install dwave-ocean-sdk")

seed = 12
np.random.seed(seed)
logger = logging.getLogger("inspect_results_logger")


def dwave_cqm_minimize(expected_annual_returns, annual_covariance, 
                                  risk_free_rate, budget, min_investment=100.0):
    """
    When you DO need minimum investment constraints (practical trading)
    """
    n_assets = len(expected_annual_returns)
    model = ConstrainedQuadraticModel()
    
    # Binary: whether to invest in each asset
    invest_binary = [model.binary() for _ in range(n_assets)]
    
    # Decision: dollar amount to invest in each asset  
    investments = [model.continuous(lower_bound=0.0) for _ in range(n_assets)]
    
    # Budget constraint
    model.add_constraint(sum(investments) <= budget)
    
    # Minimum investment constraints (PRACTICAL REASONS):
    for i in range(n_assets):
        # If investing, must invest at least min_investment (transaction costs, etc.)
        model.add_constraint(investments[i] >= min_investment * invest_binary[i])
        # If not investing, investment is 0
        model.add_constraint(investments[i] <= budget * invest_binary[i])
    
    # Force investment in ALL assets (to ensure weights > 0)
    for i in range(n_assets):
        model.add_constraint(invest_binary[i] == 1)
    
    total_investment = sum(investments)
    weights = [investments[i] / total_investment for i in range(n_assets)]
    
    # Sharpe ratio calculation
    portfolio_return = sum(expected_annual_returns[i] * weights[i] 
                          for i in range(n_assets))
    portfolio_variance = sum(sum(annual_covariance[i][j] * weights[i] * weights[j] 
                                for j in range(n_assets)) 
                            for i in range(n_assets))
    
    sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_variance ** 0.5)
    model.minimize(-sharpe_ratio)
    
    return model, investments, invest_binary, weights

def portfolio_stats(weights, data):
    """Calculate portfolio statistics (Sharpe ratio, return, volatility)"""
    weights = np.array(weights)
    returns = np.log(data) - np.log(data.shift(1)) # log return to minimize fp error
    port_return = np.sum(returns.mean() * weights) 
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() , weights)))
    try:
        sharpe_ratio = port_return/port_vol
    except Exception as e:
        sharpe_ratio = 0
    return sharpe_ratio, port_return, port_vol


def fitness_function(weights, data, risk_aversion=0.5):
    """
    QUBO-compatible fitness function for fair comparison.
    
    Uses the same objective as QUBO: minimize λ*risk - (1-λ)*return
    Converted to maximization for compatibility with optimization routines.
    
    Args:
        weights: Portfolio weights
        data: Stock price data
        risk_aversion: Risk-return tradeoff parameter (λ)
    
    Returns:
        Negative of QUBO objective (for maximization)
    """
    _, portfolio_return, portfolio_risk = portfolio_stats(weights, data)
    
    # QUBO objective: minimize λ*risk - (1-λ)*return
    qubo_objective = risk_aversion * portfolio_risk - (1 - risk_aversion) * portfolio_return
    
    # Return negative for maximization (since optimizers typically maximize)
    return -qubo_objective


# Equal Weights Baseline Model
def equal_weights_baseline(data):
    """
    Equal weights baseline model for portfolio optimization.
    
    This serves as a naive baseline where each asset receives equal weight (1/N).
    It's a common benchmark in portfolio optimization research.
    
    Args:
        data: Stock price data (DataFrame)
    
    Returns:
        weights: Equal weights portfolio as numpy array
    """
    n_assets = len(data.columns)
    weights = np.ones(n_assets) / n_assets
    
    logger.debug(f"Equal weights baseline: {n_assets} assets, weight={1/n_assets:.6f} each")
    return weights


# Genetic Algorithm 
def genetic_algorithm(data, fitness_function, risk_aversion, population_size=500, num_generations=1000, mutation_rate=0.05, elitism=0.1):
    population = np.random.rand(population_size, len(data.columns))
    population = population / np.sum(population, axis=1)[:, np.newaxis]
    fitness = np.array([fitness_function(weights=individual, data=data, risk_aversion=risk_aversion) for individual in population])
    for generation in range(num_generations):
        sorted_idx = np.argsort(fitness)[::-1]
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        num_elites = int(elitism * population_size)
        offspring = population[:num_elites]
        parent1_idx = np.random.randint(num_elites, population_size, size=population_size-num_elites)
        parent2_idx = np.random.randint(num_elites, population_size, size=population_size-num_elites)
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        crossover_prob = np.random.rand(population_size-num_elites, len(data.columns))
        crossover_mask = crossover_prob <= 0.5
        offspring_crossover = np.where(crossover_mask, parent1, parent2)
        mutation_prob = np.random.rand(population_size-num_elites, len(data.columns))
        mutation_mask = mutation_prob <= 0.5
        mutation_values = np.random.rand(population_size-num_elites, len(data.columns))
        mutation_direction = np.random.choice([-1, 1], size=(population_size - num_elites, len(data.columns)))
        offspring_mutation = np.where(mutation_mask, offspring_crossover + mutation_direction * mutation_values, offspring_crossover)
        population = np.vstack((population[:num_elites], offspring_mutation))
        fitness = np.array([fitness_function(individual, data) for individual in population])
    selected = []
    for f in fitness:
        if np.all(f > 0):
            selected.append(f)
    best_idx = np.argmax(selected)
    best_individual = population[best_idx]
    logger.debug('### Best Individual ###')
    logger.debug(best_individual)
    return best_individual


def scipy_minimize(data, fitness_function):
    num_assets = data.shape[1]
    constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
    bounds = tuple((0.01, 0.2) for x in range(num_assets))
    initializer = num_assets * [1./num_assets,]
    port_return = partial(fitness_function, data=data)

    def minimize_sharpe(weights):
        return -1 * port_return(weights)
    
    weights = np.random.dirichlet(np.ones(num_assets),size=1)
    optimal_sharpe=optimize.minimize(minimize_sharpe,
                                    initializer,
                                    method = 'SLSQP',
                                    bounds = bounds,
                                    constraints = constraints)
    
    optimal_sharpe_weights=optimal_sharpe['x'].round(4)
    return np.array(optimal_sharpe_weights)


def riskfolio_minimize(data):
    y = np.log(data) - np.log(data.shift(1))
    port = rp.HCPortfolio(returns=y[1:])

    # Estimate optimal portfolio:
    model='HERC' # Could be HRP or HERC
    codependence = 'pearson' # Correlation matrix used to group assets in clusters
    rm = 'MV' # Risk measure used, this time will be variance
    rf = 0 # Risk free rate
    linkage = 'single' # Linkage method used to build clusters
    max_k = 10 # Max number of clusters used in two difference gap statistic, only for HERC model
    leaf_order = True # Consider optimal order of leafs in dendrogram

    w = port.optimization(model=model,
                        codependence=codependence,
                        rm=rm,
                        rf=rf,
                        linkage=linkage,
                        max_k=max_k,
                        leaf_order=leaf_order)

    return np.array(w).flatten()

def dwave_classical_minimize(data, risk_aversion=0.5, n_levels=5, sampler_type='simulated_annealing'):
    
    logger.debug(f"Classical QUBO portfolio optimization with {len(data.columns)} assets")
    logger.debug(f"Using {sampler_type} sampler with risk_aversion={risk_aversion}")
    
    # Select classical sampler
    samplers = {
        'simulated_annealing': SimulatedAnnealingSampler(),
        'tabu': TabuSampler(),
        'steepest_descent': SteepestDescentSampler(),
        'exact': ExactSolver()
    }
    
    if sampler_type not in samplers:
        logger.warning(f"Unknown sampler type: {sampler_type}. Using simulated_annealing.")
        sampler_type = 'simulated_annealing'
    
    sampler = samplers[sampler_type]
    
    # Data preparation - same as classical methods
    returns = np.log(data) - np.log(data.shift(1))
    avg_returns = returns.mean().fillna(0)
    cov_matrix = returns.cov().fillna(0)
    
    stocks = data.columns.tolist()
    n_stocks = len(stocks)
    
    # Create binary variables for discrete weight levels
    # Each asset gets n_levels binary variables representing investment levels
    bqm = BinaryQuadraticModel('BINARY')
    
    # Variable naming: asset_level (e.g., 'AAPL_0', 'AAPL_1', ..., 'AAPL_4')
    variables = {}
    for i, stock in enumerate(stocks):
        for level in range(n_levels):
            var_name = f"{stock}_{level}"
            variables[var_name] = (i, level)
    
    # QUBO formulation: minimize λ*risk - (1-λ)*return
    # This is equivalent to the mean-variance optimization framework
    
    # Linear terms (expected returns) - negative because we want to maximize
    for i, stock in enumerate(stocks):
        for level in range(n_levels):
            var_name = f"{stock}_{level}"
            # Weight represented by this binary variable
            weight = (level + 1) / (n_levels * n_stocks)  # Normalized discrete levels
            # Expected return contribution
            return_contrib = avg_returns.iloc[i] * weight
            # Coefficient: -return_contrib (negative to maximize) scaled by (1-λ)
            bqm.add_variable(var_name, -(1 - risk_aversion) * return_contrib)
    
    # Quadratic terms (risk/covariance) - positive because we want to minimize
    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks):
            for level1 in range(n_levels):
                for level2 in range(n_levels):
                    var1 = f"{stock1}_{level1}"
                    var2 = f"{stock2}_{level2}"
                    
                    # Weights represented by these binary variables
                    weight1 = (level1 + 1) / (n_levels * n_stocks)
                    weight2 = (level2 + 1) / (n_levels * n_stocks)
                    
                    # Risk coefficient from covariance matrix
                    risk_coeff = cov_matrix.iloc[i, j] * weight1 * weight2 * risk_aversion
                    
                    if i == j and level1 == level2:
                        # Diagonal terms (self-interaction)
                        bqm.add_variable(var1, risk_coeff)
                    else:
                        # Off-diagonal terms (cross-interactions)
                        bqm.add_interaction(var1, var2, risk_coeff)
    
    # Add constraint penalties
    penalty_strength = 1000  # Strong penalty for constraint violations
    
    # Constraint: Each asset must have exactly one level selected
    for i, stock in enumerate(stocks):
        stock_vars = [f"{stock}_{level}" for level in range(n_levels)]
        
        # Penalty for (sum_levels - 1)^2 = 0
        # Expanded: sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2*sum_i x_i + 1
        # Since x_i^2 = x_i for binary variables: sum_i x_i + 2*sum_{i<j} x_i*x_j - 2*sum_i x_i + 1
        # Simplified: 2*sum_{i<j} x_i*x_j - sum_i x_i + 1
        
        # Linear penalty terms
        for var in stock_vars:
            bqm.add_variable(var, penalty_strength * (-2 + 2 * n_levels))
        
        # Quadratic penalty terms
        for var1 in stock_vars:
            for var2 in stock_vars:
                if var1 != var2:
                    bqm.add_interaction(var1, var2, penalty_strength * 2)
    
    logger.debug(f"BQM created with {len(bqm.variables)} variables")
    
    # Solve using selected classical sampler
    try:
        if sampler_type == 'exact':
            if len(bqm.variables) > 20:
                logger.warning(f"Too many variables ({len(bqm.variables)}) for exact solver. Using simulated annealing.")
                sampler = SimulatedAnnealingSampler()
                sampleset = sampler.sample(bqm, num_reads=1000)
            else:
                sampleset = sampler.sample(bqm)
        else:
            # Use more reads for better solutions with heuristic samplers
            num_reads = 1000 if sampler_type == 'simulated_annealing' else 100
            sampleset = sampler.sample(bqm, num_reads=num_reads)
        
        logger.debug(f"Sampling completed. Best energy: {sampleset.first.energy}")
        
        # Extract solution
        best_sample = sampleset.first.sample
        
        # Convert binary solution to portfolio weights
        weights = np.zeros(n_stocks)
        
        for i, stock in enumerate(stocks):
            for level in range(n_levels):
                var_name = f"{stock}_{level}"
                if best_sample.get(var_name, 0) == 1:
                    # This level is selected for this stock
                    weights[i] = (level + 1) / (n_levels * n_stocks)
                    break
        
        # Normalize weights to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Fallback to equal weights
            weights = np.ones(n_stocks) / n_stocks
        
        # Ensure minimum weights (avoid zero weights)
        min_weight = 0.01 / n_stocks
        weights = np.maximum(weights, min_weight)
        weights = weights / weights.sum()  # Renormalize
        
        logger.debug(f"Portfolio weights sum: {weights.sum():.6f}")
        logger.debug(f"All weights > 0: {all(w > 0 for w in weights)}")
        
        # Calculate resulting portfolio statistics
        sharpe, ret, vol = portfolio_stats(weights, data)
        logger.debug(f"Resulting Sharpe ratio: {sharpe:.4f}, Return: {ret:.4f}, Vol: {vol:.4f}")
        
        return weights
        
    except Exception as e:
        logger.error(f"Classical QUBO solver error: {e}")
        # Return equal weights as fallback
        return np.ones(n_stocks) / n_stocks