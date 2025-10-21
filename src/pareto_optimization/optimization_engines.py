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
    QUBO fitness function with parameterized risk level for Pareto optimization.

    Maximizes: return - risk_level * variance

    Args:
        weights: Portfolio weights
        data: Stock price data
        risk_level: Risk aversion parameter (variable for Pareto frontier)

    Returns:
        Objective value (return - risk * variance) for maximization
    """
    port_return, port_vol = portfolio_stats(weights, data)
    portfolio_variance = port_vol ** 2
    objective = port_return - risk_level * portfolio_variance
    return objective


def equal_weights_baseline(data, risk_level=None):
    """
    Equal weights baseline portfolio.

    Args:
        data: Stock price DataFrame
        risk_level: Unused, kept for API compatibility

    Returns:
        weights: Equal weight allocation
    """
    n_assets = len(data.columns)
    return np.full(n_assets, 1.0 / n_assets)


def genetic_algorithm_qubo(data, risk_level=0.5, population_size=500, num_generations=100, mutation_rate=0.1, elitism=0.1, max_weight=None):
    """
    Genetic Algorithm for QUBO portfolio optimization with parameterized risk level.

    Args:
        data: Stock price DataFrame
        risk_level: Risk aversion parameter for Pareto optimization
        population_size: Number of individuals in each generation
        num_generations: Number of generations to evolve
        mutation_rate: Probability of mutation for each gene
        elitism: Fraction of best individuals to preserve
        max_weight: Maximum weight per asset (default None = adaptive)

    Returns:
        weights: Optimized portfolio weights
    """
    n_assets = len(data.columns)

    # Adaptive max weight: 10x equal weight or 10% cap, whichever is lower
    if max_weight is None:
        equal_weight = 1.0 / n_assets
        max_weight = min(0.10, 10 * equal_weight)

    n_elite = int(population_size * elitism)

    # Initialize population with normalized random weights
    population = []
    for _ in range(population_size):
        weights = np.random.rand(n_assets)
        weights = weights / weights.sum()  # Normalize to sum to 1
        weights = np.clip(weights, 0.001, max_weight)  # Enforce max allocation
        weights = weights / weights.sum()  # Re-normalize
        population.append(weights)

    def fitness_function(weights):
        return qubo_fitness_function(weights, data, risk_level=risk_level)

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

            # Normalize weights to sum to 1 and enforce max allocation
            child = child / child.sum() if child.sum() > 0 else np.full(n_assets, 1.0/n_assets)
            child = np.clip(child, 0.001, max_weight)  # Enforce max allocation
            child = child / child.sum()  # Re-normalize
            new_population.append(child)

        population = new_population

    # Return best individual
    final_fitness = [fitness_function(individual) for individual in population]
    best_idx = np.argmax(final_fitness)
    return population[best_idx]


def scipy_slsqp_qubo(data, risk_level=0.5, max_weight=None, n_retries=3):
    """
    SciPy SLSQP optimizer for QUBO portfolio optimization with parameterized risk level.

    Improved version with:
    - Multiple random initializations
    - Tighter tolerances for better convergence
    - More iterations for large portfolios
    - Retry mechanism with different starting points

    Args:
        data: Stock price DataFrame
        risk_level: Risk aversion parameter for Pareto optimization
        max_weight: Maximum weight per asset (default None = adaptive)
        n_retries: Number of retries with different initializations (default 3)

    Returns:
        weights: Optimized portfolio weights
    """
    n_assets = len(data.columns)

    # Adaptive max weight: 10x equal weight or 10% cap, whichever is lower
    if max_weight is None:
        equal_weight = 1.0 / n_assets
        max_weight = min(0.10, 10 * equal_weight)

    def objective_function(weights):
        # Minimize negative objective (maximize objective)
        return -qubo_fitness_function(weights, data, risk_level=risk_level)

    # Constraints: weights sum to 1, all weights >= 0
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Budget constraint
    ]

    bounds = [(0.001, max_weight) for _ in range(n_assets)]  # Min 0.1%, adaptive max per asset

    best_result = None
    best_objective = np.inf

    # Try multiple initializations
    initializations = []

    # 1. Equal weights (conservative start)
    initializations.append(np.full(n_assets, 1.0 / n_assets))

    # 2. Random weights with bias towards diversification
    for _ in range(n_retries - 1):
        x0_random = np.random.dirichlet(np.ones(n_assets) * 2)  # Dirichlet ensures sum=1
        x0_random = np.clip(x0_random, 0.001, max_weight)  # Enforce bounds
        x0_random = x0_random / x0_random.sum()  # Re-normalize
        initializations.append(x0_random)

    for i, x0 in enumerate(initializations):
        try:
            result = optimize.minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 2000,      # Increased for large portfolios
                    'ftol': 1e-9,         # Tighter function tolerance
                    'eps': 1e-8,          # Smaller step for gradient estimation
                    'disp': False
                }
            )

            # Check if this is the best result so far
            if result.fun < best_objective:
                best_objective = result.fun
                best_result = result

        except Exception as e:
            logger.debug(f"SciPy SLSQP attempt {i+1} failed: {e}")
            continue

    # Return best result from all attempts
    if best_result is not None and best_result.success:
        return best_result.x
    elif best_result is not None:
        # Return best attempt even if not marked as successful
        logger.warning(f"SciPy optimization converged with warning: {best_result.message}")
        return best_result.x
    else:
        logger.error(f"SciPy SLSQP optimization failed after {len(initializations)} attempts")
        return np.full(n_assets, np.nan)


def riskfolio_qubo(data, risk_level=0.5, max_weight=None):
    """
    Riskfolio-lib optimizer for QUBO portfolio optimization with parameterized risk level.

    Uses mean-variance optimization with parameterized risk aversion.
    Applies Ledoit-Wolf shrinkage for robust covariance estimation.

    Args:
        data: Stock price DataFrame
        risk_level: Risk aversion parameter for Pareto optimization
        max_weight: Maximum weight per asset (default None = adaptive)

    Returns:
        weights: Optimized portfolio weights
    """
    try:
        # Calculate returns using log returns (consistent with other methods)
        returns = np.log(data) - np.log(data.shift(1))
        returns = returns.dropna()

        logger.debug(f"Returns shape: {returns.shape}, data shape: {data.shape}")

        # Create Portfolio object
        port = rp.Portfolio(returns=returns)

        # Calculate risk and return parameters with Ledoit-Wolf shrinkage
        # Use method_cov='ledoit' which applies shrinkage internally
        logger.debug("Calculating portfolio statistics with Ledoit-Wolf shrinkage...")
        port.assets_stats(method_mu='hist', method_cov='ledoit')

        logger.debug(f"Covariance matrix shape: {port.cov.shape}")

        # Verify positive definiteness
        eigenvalues = np.linalg.eigvalsh(port.cov.values)
        min_eigenvalue = eigenvalues.min()
        logger.debug(f"Covariance matrix eigenvalues - min: {min_eigenvalue:.8e}, max: {eigenvalues.max():.8e}")

        # Use Riskfolio's built-in cov_fix method if needed
        threshold = 1e-5
        if min_eigenvalue < threshold:
            logger.warning(f"Covariance matrix eigenvalue {min_eigenvalue:.8e} below threshold {threshold:.8e}")
            logger.debug("Applying Riskfolio's cov_fix method...")

            # Apply Riskfolio's built-in fix
            import riskfolio.AuxFunctions as af
            port.cov = af.cov_fix(port.cov, method='clipped', threshold=1e-5)

            # Verify fix worked
            new_eigenvalues = np.linalg.eigvalsh(port.cov.values)
            new_min_eig = new_eigenvalues.min()
            logger.debug(f"After cov_fix, min eigenvalue: {new_min_eig:.8e}")

            # If still not positive definite, apply manual regularization
            if new_min_eig < 1e-6:
                logger.warning(f"Still not positive definite after cov_fix. Applying manual regularization.")
                epsilon = 1e-5
                import pandas as pd
                port.cov = port.cov + epsilon * pd.DataFrame(np.eye(len(port.cov)),
                                                             index=port.cov.index,
                                                             columns=port.cov.columns)
                final_min_eig = np.linalg.eigvalsh(port.cov.values).min()
                logger.debug(f"After manual regularization (epsilon={epsilon:.8e}), min eigenvalue: {final_min_eig:.8e}")

        logger.debug("Covariance matrix is positive definite and ready for optimization")

        # Set upper bound constraints (max weight per asset)
        n_assets = len(data.columns)

        # Adaptive max weight: 10x equal weight or 10% cap, whichever is lower
        if max_weight is None:
            equal_weight = 1.0 / n_assets
            max_weight = min(0.10, 10 * equal_weight)

        port.ainequality = np.eye(n_assets)  # Identity matrix for individual asset constraints
        port.binequality = np.full((n_assets, 1), max_weight)  # Adaptive max per asset (column vector)

        # Map risk_level to Riskfolio's risk aversion parameter
        # In Riskfolio, risk aversion (rm) controls the risk-return tradeoff
        risk_aversion = risk_level * 2.0  # Scale for Riskfolio

        # Mean-variance optimization
        logger.debug("Running Riskfolio optimization...")
        weights = port.optimization(
            model='Classic',
            rm='MV',  # Mean Variance
            obj='Utility',  # Utility maximization (return - risk_aversion * risk)
            rf=0.0,  # Risk-free rate
            l=risk_aversion,  # Risk aversion parameter
            hist=True
        )

        if weights is not None and len(weights) > 0:
            result_weights = weights.values.flatten()
            logger.debug(f"Riskfolio optimization succeeded. Weights sum: {result_weights.sum():.6f}")
            return result_weights
        else:
            logger.warning("Riskfolio optimization returned empty result")
            n_assets = len(data.columns)
            return np.full(n_assets, np.nan)

    except Exception as e:
        logger.error(f"Riskfolio optimization error: {e}", exc_info=True)
        n_assets = len(data.columns)
        return np.full(n_assets, np.nan)


def dwave_cqm_qubo(data, risk_level=0.5, budget=1000000.0, max_weight=None):
    """
    D-Wave CQM (Constrained Quadratic Model) for QUBO portfolio optimization with parameterized risk level.

    Args:
        data: Stock price DataFrame
        risk_level: Risk aversion parameter for Pareto optimization
        budget: Total budget (default 1.0 for normalized weights)
        max_weight: Maximum weight per asset (default None = adaptive)

    Returns:
        weights: Optimized portfolio weights
    """
    n_assets = len(data.columns)

    # Adaptive max weight: 10x equal weight or 10% cap, whichever is lower
    if max_weight is None:
        equal_weight = 1.0 / n_assets
        max_weight = min(0.10, 10 * equal_weight)

    logger.info("="*80)
    logger.info("=== D-WAVE CQM QUBO OPTIMIZATION START ===")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Budget: ${budget:,.2f}")
    logger.info(f"Max weight per asset: {max_weight*100:.2f}% (adaptive)")

    alpha = risk_level  # Use risk_level directly as alpha
    logger.info(f"Risk parameter alpha: {alpha}")

    logger.info("Calculating returns...")
    returns = np.log(data) - np.log(data.shift(1))
    logger.info(f"Returns shape: {returns.shape}")

    logger.info("Calculating expected returns...")
    expected_returns = returns.mean()
    logger.info(f"Expected returns - min: {expected_returns.min():.6f}, max: {expected_returns.max():.6f}, mean: {expected_returns.mean():.6f}")

    logger.info("Calculating covariance matrix...")
    covariance_matrix = returns.cov()
    logger.info(f"Covariance matrix shape: {covariance_matrix.shape}")
    logger.info(f"Covariance matrix - min: {covariance_matrix.min().min():.8f}, max: {covariance_matrix.max().max():.8f}")

    logger.info("Creating ConstrainedQuadraticModel...")
    cqm = ConstrainedQuadraticModel()
    stocks = data.columns.tolist()
    logger.info(f"Number of stocks: {len(stocks)}")

    price = data.iloc[-1, :]
    logger.info(f"Latest prices - min: ${price.min():.2f}, max: ${price.max():.2f}, mean: ${price.mean():.2f}")

    logger.info("Calculating max number of shares per stock...")
    max_num_shares = (budget / price).astype(int)

    # Apply max weight constraint
    max_shares_with_diversification = np.minimum(max_num_shares, (budget * max_weight / price).astype(int))
    logger.info(f"Max shares (with {max_weight*100:.1f}% constraint) - min: {max_shares_with_diversification.min()}, max: {max_shares_with_diversification.max()}, mean: {max_shares_with_diversification.mean():.0f}")
    logger.info(f"Total possible decision variables: {max_shares_with_diversification.sum()}")

    logger.info("Creating integer decision variables...")
    x = {s: Integer("%s" %s, lower_bound=1, upper_bound=max_shares_with_diversification[s]) for s in stocks}
    logger.info(f"Created {len(x)} integer variables")

    logger.info("Building returns objective term...")
    returns_obj = 0
    for idx, s in enumerate(stocks):
        returns_obj = returns_obj + price[s] * expected_returns[s] * x[s]
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(stocks)} stocks for returns...")
    logger.info("Returns objective term completed")

    logger.info("Building risk objective term (quadratic)...")
    risk = 0
    total_pairs = len(stocks) * len(stocks)
    processed = 0
    for s1, s2 in product(stocks, stocks):
        coeff = float(covariance_matrix[s1][s2]) * float(price[s1]) * float(price[s2])
        risk = risk + coeff * x[s1] * x[s2]
        processed += 1
        if processed % 10000 == 0:
            logger.info(f"  Processed {processed}/{total_pairs} stock pairs for risk ({100*processed/total_pairs:.1f}%)...")
    logger.info(f"Risk objective term completed ({total_pairs} pairs)")

    logger.info("Adding budget constraint...")
    cqm.add_constraint(quicksum([x[s] * price[s] for s in stocks]) <= budget, label='upper_budget')
    logger.info("Budget constraint added")

    logger.info("Setting CQM objective function...")
    cqm.set_objective(alpha * risk - returns_obj)
    logger.info("Objective function set")

    logger.info(f"CQM model statistics:")
    logger.info(f"  Number of variables: {cqm.num_variables()}")
    logger.info(f"  Number of constraints: {cqm.num_constraints()}")
    logger.info(f"  Number of quadratic terms: {cqm.num_quadratic_variables()}")

    logger.info("Initializing LeapHybridCQMSampler...")
    sampler = LeapHybridCQMSampler()
    logger.info("Sampler initialized")

    # Ask for permission before calling D-Wave API
    print("\n" + "="*80)
    print("⚠️  READY TO CALL D-WAVE QUANTUM API")
    print("="*80)
    print(f"Risk Level (λ): {risk_level}")
    print(f"Assets: {n_assets}")
    print(f"Variables: {cqm.num_variables()}")
    print(f"Constraints: {cqm.num_constraints()}")
    print(f"Estimated time: ~20-60 seconds")
    print(f"This will consume D-Wave QPU time credits")
    print("="*80)

    # Get user permission
    response = input("Do you want to proceed with D-Wave API call? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("❌ D-Wave API call cancelled by user")
        logger.info("D-Wave API call cancelled by user")
        logger.info("=== D-WAVE CQM QUBO OPTIMIZATION END ===")
        logger.info("="*80)
        return np.full(n_assets, np.nan)

    print("✓ Proceeding with D-Wave API call...")
    logger.info("*** THIS IS WHERE THE QUANTUM SOLVER IS CALLED - MAY TAKE SEVERAL MINUTES ***")
    results = sampler.sample_cqm(cqm, time_limit=20, label="Pareto_QUBO_Portfolio_Optimization")
    logger.info("D-Wave quantum solver completed!")
    print("✓ D-Wave quantum solver completed!")

    n_samples = len(results.record)
    logger.info(f'Number of samples returned: {n_samples}')

    logger.info("Filtering for feasible samples...")
    feasible_samples = results.filter(lambda d: d.is_feasible)
    logger.info(f"Number of feasible samples: {len(feasible_samples)}")

    if len(feasible_samples) == 0:
        logger.warning("No feasible samples found! Returning NaN vector")
        n_assets = len(stocks)
        logger.info("=== D-WAVE CQM QUBO OPTIMIZATION END ===")
        logger.info("="*80)
        return np.full(n_assets, np.nan)

    logger.info("Extracting best sample...")
    best_sample = feasible_samples.first
    logger.info(f"Best sample energy: {best_sample.energy}")

    logger.info("Converting sample to weights...")
    amounts = []
    for s in stocks:
        amounts.append(best_sample.sample[s])
    amounts = np.array(amounts)
    logger.info(f"Amounts - min: {amounts.min()}, max: {amounts.max()}, sum: {amounts.sum()}")

    total = np.sum(amounts)
    weights = amounts / total
    logger.info(f"Weights - min: {weights.min():.6f}, max: {weights.max():.6f}, sum: {weights.sum():.6f}")
    logger.info(f"Number of non-zero weights: {(weights > 0).sum()}/{len(weights)}")
    print('Amounts')
    print(amounts)
    print('Weights')
    print(weights)

    logger.info("=== D-WAVE CQM QUBO OPTIMIZATION END ===")
    logger.info("="*80)
    return weights
