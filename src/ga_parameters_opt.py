#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import pickle
import itertools
import random
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

seed = 12
np.random.seed(seed)

def fitness_function(weights, data):
    data_returns = np.log(data) - np.log(data.shift(1)) # current day - previous day
    data_returns = data_returns.dropna()
    portfolio_returns = np.dot(data_returns, weights)
    portfolio_mean = np.mean(portfolio_returns)
    portfolio_std = np.sum(np.sum(weights * np.std(portfolio_returns) * data.corr().values * np.std(portfolio_returns).T * weights.T, axis=1), axis=0)
    sharpe_ratio = (portfolio_mean / portfolio_std) 
    return sharpe_ratio

def genetic_algorithm(data, population_size=500, num_generations=1000, mutation_rate=0.05, elitism=0.1):
    population = np.random.rand(population_size, len(data.columns))
    print(f'---Population---')

    population = population / np.sum(population, axis=1)[:, np.newaxis]
    mean_fitness = []
    max_fitness = []
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
    for f in fitness:
        if np.all(f > 0):
            selected.append(f)
    best_idx = np.argmax(selected)
    best_individual = population[best_idx]
    # 'Best Sharpe Ratio: ', np.max(fitness)
    return best_individual

def generate_data(df, benchmark, days_to_avg=30, days_to_opt=30):

    df2 = df.reset_index()
    benchmark2 = benchmark.reset_index()
    elements = df2.sample(n=100).index # definint a maximum of 100 different sampled initial dates
    for idx in elements:
        df_sample = df2.iloc[idx-days_to_avg:idx+days_to_opt, :]
        df_sample = df_sample.set_index('ds')
        df_sample_b = benchmark2.iloc[idx-days_to_avg:idx+days_to_opt, :]
        df_sample_b = df_sample_b.set_index('ds')
        yield df_sample, df_sample_b

def backtest(optimization_function, data, benchmark, initial_capital, avg_period, opt_period):
    portfolio_value = initial_capital
    portfolio_returns = []
    benchmark_returns = []
    weights_history = pd.DataFrame(index=data.index, columns=data.columns)
    portfolio_value_history = pd.Series(index=data.index, name='Portfolio Value', dtype='float')
    portfolio_value_history.iloc[0] = portfolio_value
    #print(len())
    j = 0
    for i in range(avg_period+1, avg_period + opt_period+1):
        df = data.iloc[j:i, :]
        weights = optimization_function(df)
        weights[weights < 0] = 0
        weights /= weights.sum()
        weights_history.loc[df.index[-1]] = weights
        portfolio_change = df.iloc[-2:, :].pct_change() * weights
        portfolio_return = portfolio_change.sum(axis=1).iloc[-1]
        portfolio_returns.append(portfolio_return)
        # #print(f'portfolio_returns: {portfolio_returns}')
        benchmark_return = benchmark.iloc[j:i, :].pct_change().iloc[-1].values.tolist()[0]
        benchmark_returns.append(benchmark_return)
        portfolio_cumulative_returns = np.cumprod([k + 1 for k in portfolio_returns])

        benchmark_cumulative_returns = np.cumprod([k + 1 for k in  benchmark_returns])
        portfolio_mean_return = np.mean(portfolio_returns)
        benchmark_mean_return = np.mean(benchmark_returns)
        portfolio_volatility = np.std(portfolio_returns) #* np.sqrt(12)
        benchmark_volatility = np.std(benchmark_returns) #* np.sqrt(12)
        try:
            sharpe_ratio = (portfolio_mean_return) / portfolio_volatility
        except Exception as e:
            sharpe_ratio = 0
        try:
            benchmark_sharpe_ratio = (benchmark_mean_return) / benchmark_volatility
        except Exception as e:
            benchmark_sharpe_ratio = 0

         # Portfolio & Benchmark value
        benchmark_value = initial_capital * benchmark_cumulative_returns[-1]
        portfolio_value = initial_capital * portfolio_cumulative_returns[-1]

        j += 1
    return weights_history, portfolio_value_history, portfolio_cumulative_returns, benchmark_cumulative_returns

def pickle_dict(data, file_path):
    """Pickles a dictionary and saves it to a file."""
    try:
        with open(file_path, 'wb') as f:  # Open the file in binary write mode ('wb')
            pickle.dump(data, f)
        print(f"Dictionary pickled and saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while pickling: {e}")


if __name__ == "__main__":
    benchmark_path = '../data/benchmark_gspc.pkl'
    source_path = '../data/stocks_adjclose.pkl'
    benchmark = pd.read_pickle(benchmark_path)
    source = pd.read_pickle(source_path)

    # correlation analysis
    df_corr = source.corr()
    corr_sum = df_corr.map(lambda x: abs(x)).sum()
    corr_rank = corr_sum.sort_values().rank(method='min').astype(int)
    corr_rank

    return_rank = source.diff().sum(axis=0).sort_values().rank(method='min', ascending=False).astype(int)

    # Select test set 
    select_20 = (return_rank + corr_rank).sort_values().reset_index()['Ticker'].values[:21]
    data = source[select_20]

    #Hyperparameter optimization
    population_size=[10, 100, 1000]
    num_generations=[10, 100, 1000]
    mutation_rate=[0.01, 0.1, 0.2]
    elitism=[0.01, 0.5, 0.1]
    n_periods = 10
    days_to_avg = 30
    days_to_opt = 30

    combs = [i for i in itertools.product(population_size, num_generations, mutation_rate, elitism)]
    n = 0
    datagen = generate_data(data, benchmark)
    df, df_b = next(datagen)
    parameters = []
    max_returns = 0
    while n < len(combs):
        print(n)
        p, n, m, e = random.choice(combs)
        opt_fun = partial(genetic_algorithm, population_size=p, num_generations=n, mutation_rate=m, elitism=e)
        weights_history, portfolio_value_history, portfolio_cumulative_returns, benchmark_cumulative_returns = backtest(opt_fun, df, df_b, initial_capital=1000, avg_period=30, opt_period=30)
        result = {
            "start_date": df.index[0],
            "end_date": df.index[-1],
            "population_size": population_size,
            "num_generations": num_generations,
            "days_to_avg": days_to_avg,
            "days_to_opt": days_to_opt,
            "weights_history": weights_history,
            "portfolio_value_history": portfolio_value_history,
            "portfolio_cumulative_returns": portfolio_cumulative_returns,
            "sum_cumulative_returns": sum(portfolio_cumulative_returns)
            }
        
        parameters.append(result)
        if sum(portfolio_cumulative_returns) > max_returns:
            max_returns = sum(portfolio_cumulative_returns)
            print(f'pop size: {p}, n gen: {n}, mut: {m}, elit: {e}, portfolio_cumulative_returns: {portfolio_cumulative_returns}')
        n += 1
    
    pickle_dict(parameters, '../results/hyperparameters_opt.pkl')
    print("----OPTIMIZATION TERMINATED----")
