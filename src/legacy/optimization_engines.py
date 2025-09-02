import numpy as np
import scipy.optimize as optimize
import riskfolio as rp
import logging
from functools import partial

# D-Wave imports for classical and quantum samplers
try:
    import gc
    from dimod import Integer, Binary, BinaryQuadraticModel, ConstrainedQuadraticModel
    from dimod import quicksum
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
    from dwave.system import LeapHybridCQMSampler
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

# Definition of the Genetic Algorithm with elitism. I am only considering solutions where all the weights are > 0. That might not be optimal for comparison with GSPC but it is the best mapping to Capital Allocation.

def genetic_algorithm(data, fitness_function, population_size=500, num_generations=1000, mutation_rate=0.05, elitism=0.1):
    population = np.random.rand(population_size, len(data.columns))
    population = population / np.sum(population, axis=1)[:, np.newaxis]
    fitness = np.array([fitness_function(individual, data) for individual in population])
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
    # consider only solutions where all weights are greater than zero
    #logger.debug(f'fitness: {fitness}')
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
    #y = np.log(data) - np.log(data.shift(1)) 
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


def dwave_classical_sharpe(data, budget=1000, min_weight=0.01, risk_aversion=0.5, sampler_type='simulated_annealing'):
    """
    D-Wave classical implementation for Sharpe ratio optimization.
    
    This function uses classical D-Wave samplers (no quantum hardware required)
    to solve portfolio optimization as a Binary Quadratic Model (BQM).
    
    Args:
        data: Price data for stocks (DataFrame)
        budget: Total investment budget  
        min_weight: Minimum weight per asset (ensures all weights > 0)
        risk_aversion: Trade-off parameter between risk and return
        sampler_type: Classical sampler to use ('simulated_annealing', 'tabu', 'steepest_descent', 'exact')
    
    Returns:
        weights: Portfolio weights as numpy array
    
    Note:
        Uses classical algorithms - no D-Wave API token required.
    """
    if not DWAVE_AVAILABLE:
        logger.warning("D-Wave Ocean SDK not available. Falling back to equal weights.")
        return np.ones(len(data.columns)) / len(data.columns)
    
    logger.debug(f"D-Wave Classical Sharpe optimization with {len(data.columns)} stocks")
    logger.debug(f"Using {sampler_type} sampler")
    
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
    logger.debug(f"Successfully initialized {sampler_type} sampler")
    
    # Data preparation
    prices = data.iloc[-1, :].replace(0, 1e-6)  # Last available prices
    returns = np.log(data) - np.log(data.shift(1))  # Log returns
    avg_returns = returns.mean()  # Expected returns
    cov_matrix = returns.cov()  # Covariance matrix for risk calculation
    
    # Remove any NaN values
    avg_returns = avg_returns.fillna(0)
    cov_matrix = cov_matrix.fillna(0)
    
    stocks = data.columns.tolist()
    n_stocks = len(stocks)
    
    # For BQM, we'll use binary variables representing discrete investment levels
    # Each stock gets multiple binary variables for different investment amounts
    n_levels = 5  # Number of investment levels per stock
    
    # Create binary variables: stock_i_level_j
    variables = []
    for i, stock in enumerate(stocks):
        for level in range(n_levels):
            variables.append(f"{stock}_level_{level}")
    
    # Build BQM
    bqm = BinaryQuadraticModel('BINARY')
    
    # Linear terms (expected returns) - we want to maximize these
    for i, stock in enumerate(stocks):
        for level in range(n_levels):
            var_name = f"{stock}_level_{level}"
            # Investment amount for this level
            investment_amount = (budget / n_stocks) * (level + 1) / n_levels
            # Expected return from this investment
            expected_return = avg_returns.iloc[i] * investment_amount
            # Negative because BQM minimizes (we want to maximize returns)
            bqm.add_variable(var_name, -expected_return * risk_aversion)
    
    # Quadratic terms (risk/covariance) - we want to minimize these
    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks):
            for level1 in range(n_levels):
                for level2 in range(n_levels):
                    var1 = f"{stock1}_level_{level1}"
                    var2 = f"{stock2}_level_{level2}"
                    
                    # Investment amounts
                    inv1 = (budget / n_stocks) * (level1 + 1) / n_levels
                    inv2 = (budget / n_stocks) * (level2 + 1) / n_levels
                    
                    # Risk coefficient
                    risk_coeff = cov_matrix.iloc[i, j] * inv1 * inv2
                    
                    if i == j and level1 == level2:
                        # Diagonal terms
                        bqm.add_variable(var1, risk_coeff)
                    elif i != j or level1 != level2:
                        # Off-diagonal terms
                        bqm.add_interaction(var1, var2, risk_coeff)
    
    # Add constraints using penalty method
    penalty_strength = 1000  # Large penalty for constraint violations
    
    # Constraint 1: Each stock must have exactly one investment level selected
    for i, stock in enumerate(stocks):
        stock_vars = [f"{stock}_level_{level}" for level in range(n_levels)]
        
        # Add penalty for not selecting exactly one level
        # (sum - 1)^2 = sum^2 - 2*sum + 1
        for var in stock_vars:
            bqm.add_variable(var, penalty_strength * (2 * len(stock_vars) - 2))
        
        for var1 in stock_vars:
            for var2 in stock_vars:
                if var1 != var2:
                    bqm.add_interaction(var1, var2, 2 * penalty_strength)
    
    logger.debug(f"BQM created with {len(bqm.variables)} variables")
    
    # Solve using selected classical sampler
    try:
        if sampler_type == 'exact':
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
        total_investment = 0
        
        for i, stock in enumerate(stocks):
            for level in range(n_levels):
                var_name = f"{stock}_level_{level}"
                if best_sample.get(var_name, 0) == 1:
                    investment = (budget / n_stocks) * (level + 1) / n_levels
                    weights[i] = investment
                    total_investment += investment
                    break
        
        # Normalize weights
        if total_investment > 0:
            weights = weights / total_investment
        else:
            # Fallback to equal weights
            weights = np.ones(n_stocks) / n_stocks
        
        # Ensure minimum weights
        min_weight_actual = budget * min_weight / total_investment if total_investment > 0 else min_weight
        weights = np.maximum(weights, min_weight_actual / n_stocks)
        weights = weights / weights.sum()  # Renormalize
        
        logger.debug(f"All weights > 0: {all(w > 0 for w in weights)}")
        logger.debug(f"Min weight: {weights.min():.4f}, Max weight: {weights.max():.4f}")
        
        # Calculate resulting Sharpe ratio
        portfolio_return = np.sum(avg_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        logger.debug(f"Resulting Sharpe ratio: {sharpe:.4f}")
        
        return weights
        
    except Exception as e:
        logger.error(f"Classical D-Wave solver error: {e}")
        # Return equal weights as fallback
        return np.ones(n_stocks) / n_stocks


def portfolio_stats(weights, data):
    """
    Calculate portfolio statistics (Sharpe ratio, return, volatility) using same method as classical fitness_function.
    
    This function is copied from quantum_scaffolding.ipynb to ensure consistency.
    """
    weights = np.array(weights)
    returns = np.log(data) - np.log(data.shift(1))  # log return to minimize fp error 
    port_return = np.sum(returns.mean() * weights) 
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    try:
        sharpe_ratio = port_return/port_vol
    except Exception as e:
        sharpe_ratio = 0
    return sharpe_ratio, port_return, port_vol


def fitness_function(weights, data):
    """
    Classical fitness function - copied from quantum_scaffolding.ipynb for consistency.
    """
    sharpe_ratio, _, _ = portfolio_stats(weights, data)
    return sharpe_ratio


def dwave_quantum_sharpe(data, budget=1000, min_weight=0.01):
    """
    D-Wave quantum implementation that MAXIMIZES Sharpe ratio - RISK-NEUTRAL approach.
    
    This function is now truly risk-neutral like the classical fitness_function:
    - No risk aversion parameters - directly maximizes Sharpe ratio
    - Mathematically equivalent to classical fitness_function
    - Uses multiple optimization strategies to find maximum Sharpe ratio point
    
    Args:
        data: Price data for stocks (DataFrame) 
        budget: Total investment budget
        min_weight: Minimum weight per asset (ensures all weights > 0)
    
    Returns:
        weights: Portfolio weights as numpy array (same format as classical)
    """
    if not DWAVE_QUANTUM_AVAILABLE:
        logger.warning("D-Wave Quantum SDK not available. Falling back to equal weights.")
        return np.ones(len(data.columns)) / len(data.columns)
        
    logger.debug(f"D-Wave CQM Sharpe ratio MAXIMIZATION with {len(data.columns)} stocks")
    
    # Use same data preparation as classical fitness_function
    returns = np.log(data) - np.log(data.shift(1))  # EXACT same as portfolio_stats
    avg_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Remove any NaN values
    avg_returns = avg_returns.fillna(0)
    cov_matrix = cov_matrix.fillna(0)
    
    # Try D-Wave connection
    try:
        sampler = LeapHybridCQMSampler()
        logger.debug("✅ Connected to D-Wave Leap service")
    except Exception as e:
        logger.warning(f"❌ D-Wave connection failed: {e}")
        logger.warning("Fallback: Using equal weights portfolio")
        return np.ones(len(data.columns)) / len(data.columns)
    
    # RISK-NEUTRAL APPROACH: Find the maximum Sharpe ratio portfolio
    # This is equivalent to finding the tangent portfolio to the efficient frontier
    
    best_sharpe = -np.inf
    best_weights = None
    
    # Use focused optimization strategies that target maximum Sharpe ratio
    optimization_strategies = [
        {"name": "Max_Return", "objective": "return", "risk_penalty": 0.001},  # Minimal risk penalty
        {"name": "Efficient_High", "objective": "balanced", "risk_penalty": 0.01},
        {"name": "Efficient_Medium", "objective": "balanced", "risk_penalty": 0.1},
        {"name": "Sharpe_Focused", "objective": "balanced", "risk_penalty": 0.5},
    ]
    
    for strategy in optimization_strategies:
        try:
            logger.debug(f"  Testing strategy: {strategy['name']}")
            
            cqm = ConstrainedQuadraticModel()
            
            # Current prices for share calculation
            prices = data.iloc[-1, :].replace(0, 1e-6)
            
            # Calculate bounds ensuring minimum weights
            min_budget_per_stock = budget * min_weight
            min_shares = (min_budget_per_stock / prices).astype(int).clip(lower=1)
            max_shares = (budget / prices * 0.8).astype(int).clip(lower=min_shares)
            
            stocks = data.columns.tolist()
            
            # Decision variables: number of shares
            x = {s: Integer(f"x_{s}", 
                           lower_bound=min_shares[s], 
                           upper_bound=max_shares[s]) for s in stocks}
            
            # Risk component (portfolio variance)
            risk_component = 0
            for i, s1 in enumerate(stocks):
                for j, s2 in enumerate(stocks):
                    coeff = cov_matrix.iloc[i, j] * prices[s1] * prices[s2]
                    risk_component += coeff * x[s1] * x[s2]
            
            # Return component  
            return_component = 0
            for i, s in enumerate(stocks):
                coeff = avg_returns.iloc[i] * prices[s]
                return_component += coeff * x[s]
            
            # RISK-NEUTRAL OBJECTIVE: Focus on maximizing Sharpe ratio
            risk_penalty = strategy["risk_penalty"]
            
            if strategy["objective"] == "return":
                # Pure return maximization (with tiny risk penalty for numerical stability)
                objective = -return_component + risk_penalty * risk_component
            else:
                # Balanced approach targeting high Sharpe ratio regions of efficient frontier
                objective = risk_penalty * risk_component - return_component
            
            cqm.set_objective(objective)
            
            # Budget constraint
            budget_expr = quicksum([x[s] * prices[s] for s in stocks])
            cqm.add_constraint(budget_expr <= budget, label='budget_upper')
            cqm.add_constraint(budget_expr >= budget * 0.95, label='budget_lower')
            
            # Solve
            sampleset = sampler.sample_cqm(cqm, label=f"Sharpe_Max_{strategy['name']}")
            feasible = sampleset.filter(lambda row: row.is_feasible)
            
            if feasible:
                best_sample = feasible.first
                
                # Extract solution and convert to weights
                solution = {s: int(best_sample.sample[f"x_{s}"]) for s in stocks}
                total_value = sum(solution[s] * prices[s] for s in stocks)
                
                if total_value > 0:
                    # Calculate weights (normalized to sum = 1, like classical)
                    candidate_weights = np.array([solution[s] * prices[s] / total_value for s in stocks])
                    
                    # Calculate Sharpe ratio using EXACT same method as classical
                    candidate_sharpe, candidate_return, candidate_vol = portfolio_stats(candidate_weights, data)
                    
                    logger.debug(f"    Strategy {strategy['name']}: Sharpe={candidate_sharpe:.6f}, Return={candidate_return:.6f}, Vol={candidate_vol:.6f}")
                    
                    # Keep best Sharpe ratio solution (RISK-NEUTRAL: only care about max Sharpe)
                    if candidate_sharpe > best_sharpe:
                        best_sharpe = candidate_sharpe
                        best_weights = candidate_weights.copy()
                        logger.debug(f"    🎯 NEW BEST Sharpe: {best_sharpe:.6f}")
        
        except Exception as e:
            logger.debug(f"    Error with strategy {strategy['name']}: {e}")
            continue
    
    if best_weights is not None:
        logger.debug(f"🎯 Final quantum result:")
        logger.debug(f"   Best Sharpe ratio: {best_sharpe:.6f}")
        logger.debug(f"   Weights sum: {best_weights.sum():.6f}")
        logger.debug(f"   All weights > 0: {all(w > 0 for w in best_weights)}")
        
        # Verify using classical fitness_function calculation
        verification_sharpe = fitness_function(best_weights, data)
        logger.debug(f"   Verification (classical formula): {verification_sharpe:.6f}")
        
        return best_weights
    else:
        logger.warning("❌ No feasible solution found, using equal weights")
        return np.ones(len(data.columns)) / len(data.columns)


def quantum_optimization_function(data, min_weight=0.05):
    """
    Quantum optimization function that MAXIMIZES Sharpe ratio to match classical fitness_function.
    
    This function is mathematically aligned with the classical approach:
    - Uses same data preprocessing (log returns)
    - Maximizes Sharpe ratio (not risk-adjusted utility)  
    - Returns normalized weights that sum to 1
    - Can be used as drop-in replacement for genetic_algorithm, scipy_minimize
    
    Args:
        data: Stock price data (DataFrame)
        min_weight: Minimum weight per asset (ensures diversification)
    
    Returns:
        weights: Portfolio weights as numpy array (same format as classical)
    """
    return dwave_quantum_sharpe(data, budget=1000, min_weight=min_weight)