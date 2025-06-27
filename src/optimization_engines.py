import numpy as np
import scipy.optimize as optimize
import riskfolio as rp
import logging
from functools import partial
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