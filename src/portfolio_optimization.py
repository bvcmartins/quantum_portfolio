#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import logging
import os
import warnings
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from optimization_engines import genetic_algorithm, scipy_minimize, riskfolio_minimize

seed = 12

np.random.seed(seed)
msg_level = logging.INFO
# Suppress all RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ## Portfolio Optimization using multiple engines

# Create a logger
logger = logging.getLogger("inspect_results_logger")
logger.setLevel(msg_level)  # Set the level for this logger

# Create a handler (where to send the logs)
handler = logging.StreamHandler()  # Send to the console
handler.setLevel(msg_level)

# Create a formatter (how to format the logs)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# ### Path definition

benchmark_path = '../data/benchmark_gspc.pkl'
source_path = '../data/stocks_adjclose.pkl'

# ### Data loading
benchmark = pd.read_pickle(benchmark_path)

source = pd.read_pickle(source_path)

# ### Correlation Analysis
# 
# * Determine a set of stocks with minimal correlation

df_corr = source.corr()

# #### Rank correlation ascending

# rank by correlation
corr_sum = df_corr.map(lambda x: abs(x)).sum()
corr_rank = corr_sum.sort_values().rank(method='min').astype(int)

# #### Rank returns descending

# rank by returns
return_rank = source.diff().sum(axis=0).sort_values().rank(method='min', ascending=False).astype(int)

# #### Select sets of 10 and 100 stocks with maximal returns and minimal correlation

select_10 = (return_rank + corr_rank).sort_values().reset_index()['Ticker'].values[:11]
select_100 = (return_rank + corr_rank).sort_values().reset_index()['Ticker'].values[:101]

# Defining Fitness Function for the calculation of Sharpe Ratio. I am using the Portfolio Variance as 
# 
# Portfolio variance = w12σ12 + w22σ22 + 2w1w2Cov1,2

# ### Porfolio Stats

def portfolio_stats(weights, data):

    weights = np.array(weights)
    returns =np.log(data) - np.log(data.shift(1)) # log return to minimize fp error
    port_return = np.sum(returns.mean() * weights) 
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() , weights)))
    try:
        sharpe_ratio = port_return/port_vol
    except Exception as e:
        sharpe_ratio = 0
    return sharpe_ratio, port_return, port_vol

# ### Fitness function

def fitness_function(weights, data):
    sharpe_ratio, _, _ = portfolio_stats(weights, data)
    return sharpe_ratio

def curried_fitness(weights, data):
    port_return = partial(fitness_function, data)
    return port_return

def minimize_sharpe(weights):
    return -1 * curried_fitness(weights)

# Data generator to instantiate blocks on demand

def generate_data(df, benchmark, days_to_avg=30, days_to_opt=30):
    df2 = df.reset_index()
    benchmark2 = benchmark.reset_index()
    elements = df2.sample(n=100).index # definint a maximum of 100 different sampled initial dates
    for idx in elements:
        df_sample = df2.iloc[idx-days_to_avg:idx+days_to_opt, :]
        df_sample = df_sample.set_index('ds')
        df_sample_b = benchmark2.iloc[idx-days_to_avg:idx+days_to_opt, :]
        df_sample_b = df_sample_b.set_index('ds').drop(['index'], axis=1)
        yield df_sample, df_sample_b


# Main Backtest function. A similar function is defined in the bottom of this notebook for detailed testing.
# 
# In case we want to implement annualized returns: 
# 
# Annualized Return = (1 + Period Return)^(365/Number of Days in Period) - 1

def backtest(optimization_function, data, benchmark, initial_capital, avg_period, opt_period):
    portfolio_value = initial_capital
    portfolio_returns = []
    benchmark_returns = []
    portfolio_sharpe_ratios = []
    weights_history = pd.DataFrame(index=data.index, columns=data.columns)
    portfolio_value_history = pd.Series(index=data.index, name='Portfolio Value', dtype='float')
    portfolio_value_history.iloc[0] = portfolio_value

    j = 0
    for i in range(avg_period+1, avg_period + opt_period+1):
        df = data.iloc[j:i, :]
        df_pct = df.iloc[j:i, :].pct_change().dropna(axis=0)
        weights = optimization_function(df)
        weights[weights < 0] = 0
        weights /= weights.sum()
        weights_history.loc[df.index[-1]] = weights
        portfolio_change = df_pct.iloc[-1] * weights
        portfolio_return = portfolio_change.sum()
        portfolio_returns.append(portfolio_return)
        benchmark_return = benchmark.iloc[j:i, :].pct_change().iloc[-1].values.tolist()[0]
        benchmark_returns.append(benchmark_return)
        portfolio_cumulative_returns = np.cumprod([k + 1 for k in portfolio_returns])
        benchmark_cumulative_returns = np.cumprod([k + 1 for k in  benchmark_returns])
        portfolio_mean_return = np.mean(portfolio_returns)
        benchmark_mean_return = np.mean(benchmark_returns)
        portfolio_volatility = np.std(portfolio_returns) 
        benchmark_volatility = np.std(benchmark_returns)
        try:
            sharpe_ratio = (portfolio_mean_return) / portfolio_volatility
        except Exception as e:
            sharpe_ratio = 0
        portfolio_sharpe_ratios.append(sharpe_ratio)

         # Portfolio & Benchmark value
        benchmark_value = initial_capital * benchmark_cumulative_returns[-1]
        portfolio_value = initial_capital * portfolio_cumulative_returns[-1]
        j += 1

    portfolio_cumulative_returns = portfolio_cumulative_returns - portfolio_cumulative_returns[0]
    benchmark_cumulative_returns = benchmark_cumulative_returns - benchmark_cumulative_returns[0]

    return weights_history, portfolio_value_history, portfolio_cumulative_returns, benchmark_cumulative_returns



import pickle

def write_pickle_dict(data, file_path):
    """Pickles a dictionary and saves it to a file."""
    try:
        with open(file_path, 'wb') as f:  # Open the file in binary write mode ('wb')
            pickle.dump(data, f)
        print(f"Dictionary pickled and saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while pickling: {e}")

def read_pickle_dict(file_path):
    try:
        with open(file_path, 'rb') as f:
            loaded_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return loaded_dict


# seed = 12
# np.random.seed(seed)
#     selected = []
#     # consider only solutions where all weights are greater than zero
#     #logger.debug(f'fitness: {fitness}')
#     for f in fitness:
#         if np.all(f > 0):
#             selected.append(f)
#     best_idx = np.argmax(selected)
#     best_individual = population[best_idx]
#     logger.debug('### Best Individual ###')
#     logger.debug(best_individual)

#     return best_individual


def run_experiment(results_path_template, data, benchmark, opt_fun, parameters):

    n_periods = parameters['n_periods']
    days_to_avg = parameters['days_to_avg']
    days_to_opt = parameters['days_to_opt']
    population_size = parameters['population_size']
    num_generations = parameters['num_generations']
    initial_capital = parameters['initial_capital']

    logger.info('Run experiment')
    for i in range(n_periods):
        logger.info(f'run number: {i}')
        results_path = results_path_template.format(i)
        logger.info(f'results_path: {results_path}')
        if not os.path.exists(results_path):
            datagen = generate_data(data, benchmark)
            df, df_b = next(datagen)
            logger.info(f'initial date: {df.iloc[days_to_avg+1:, :].index[0]}')
            start_time = datetime.now()
            weights_history, portfolio_value_history, portfolio_cumulative_returns, benchmark_cumulative_returns = backtest(opt_fun, df, df_b, initial_capital=initial_capital, avg_period=days_to_avg,opt_period=days_to_opt)
            logger.debug(f'portfolio cumulative returns: {portfolio_cumulative_returns}')
            end_time = datetime.now()
            dt = abs(end_time - start_time)

            result = {
                "round": i, 
                "start_date": df.index[0],
                "end_date": df.index[-1],
                "population_size": population_size,
                "num_generations": num_generations,
                "days_to_avg": days_to_avg,
                "days_to_opt": days_to_opt,
                "weights_history": weights_history,
                "portfolio_value_history": portfolio_value_history,
                "portfolio_cumulative_returns": portfolio_cumulative_returns,
                "benchmark_cumulative_returns": benchmark_cumulative_returns,
                "total_run_time": dt.total_seconds()
                }
            write_pickle_dict(result, results_path)
        else:
            logger.info(f'results_path {results_path} exists')

    return None


def pair_comparison(results_path_1_template, results_path_2_template, parameters):
    n_periods = parameters['n_periods']

    for i in range(n_periods):
        results_path_1 = results_path_1_template.format(i)
        results_path_2 = results_path_2_template.format(i)
        results_1 = read_pickle_dict(results_path_1)
        results_2 = read_pickle_dict(results_path_2) 
        portfolio_cumulative_returns_ga = results_1['portfolio_cumulative_returns']
        benchmark_cumulative_returns_ga = results_1['benchmark_cumulative_returns']
        portfolio_cumulative_returns_bf = results_2['portfolio_cumulative_returns']
        benchmark_cumulative_returns_bf = results_2['benchmark_cumulative_returns']

# ### Easy Test: 10 stocks

data_10 = source[select_10]

# 
# #### From Hyperparameter Optimization
# 
# * pop size: 100
# * n gen: 100
# * mut: 0.1
# * elit: 0.5



parameters = {
    "population_size": 100,
    "num_generations": 100,
    "mutation_rate": 0.1,
    "elitism": 0.1,
    "n_periods": 10,
    "days_to_avg": 30,
    "days_to_opt": 30,
    "initial_capital": 1000,
}

opt_fun_ga = partial(genetic_algorithm, 
                  fitness_function=fitness_function, 
                  population_size=parameters["population_size"], 
                  num_generations=parameters["num_generations"], 
                  mutation_rate=parameters["mutation_rate"], 
                  elitism=parameters["elitism"])


# # ### Problem 1: Optimization for 10 stocks
# # #### GA 

# results_path = '../results/results_10_{}_ga.pkl'
# _ = run_experiment(results_path, data_10, benchmark, opt_fun_ga, parameters)


# # ### Run optimization for 10 stocks: Brute Force

opt_fun_bf = partial(scipy_minimize, fitness_function=fitness_function)
# results_path = '../results/results_10_{}_bf.pkl'
# _ = run_experiment(results_path, data_10, benchmark, opt_fun_bf, parameters)

# results_path_ga = '../results/results_10_{}_ga.pkl'
# results_path_bf = '../results/results_10_{}_bf.pkl'

# # ### Medium test: 100 stocks

# data_100 = source[select_100]

# # #### GA

# results_path_ga_100 = '../results/results_100_{}_ga.pkl'
# _ = run_experiment(results_path_ga_100, data_100, benchmark, opt_fun_ga, parameters)

# # ### BF

# results_path_bf_100 = '../results/results_100_bf.pkl'
# _ = run_experiment(results_path_bf_100, data_100, benchmark, opt_fun_bf, parameters)


# ### Full dataset

# ### GA

results_path_ga_full = '../results/results_full_{}_ga.pkl'
_ = run_experiment(results_path_ga_full, source, benchmark, opt_fun_ga, parameters)

# ### BF


results_path_bf_full = '../results/results_full_{}_bf.pkl'
_ = run_experiment(results_path_bf_full, source, benchmark, opt_fun_bf, parameters)

opt_fun_hrp = partial(riskfolio_minimize)

results_path_hrp_full = '../results/results_full_{}_hrp.pkl'
_ = run_experiment(results_path_hrp_full, source, benchmark, opt_fun_hrp, parameters)

