"""
This file generates charts for experiment 1. 
It  compare the results of your manual strategy and the strategy learner.

Student Name: Apurva Gandhi
GT User ID: agandhi301
GT ID: 903862828
"""

import datetime as dt
from matplotlib import pyplot as plt
import pandas as pd
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as msc
from util import get_data

def author():
    """
    : return: The GT username of the student
    : rtype: str
    """
    return "agandhi301"

def conduct_experiment1():
    start_date_in_sample = dt.datetime(2008, 1, 1)
    end_date_in_sample = dt.datetime(2009, 12, 31)
    start_date_out_sample = dt.datetime(2010, 1, 1)
    end_date_out_sample = dt.datetime(2011, 12, 31)
    start_value = 100000
    symbol = "JPM"

    # Manual Strategy in-sample Testing
    df_manual_strategy_trades_in_sample = ms.testPolicy(symbol, start_date_in_sample, end_date_in_sample, start_value)
    df_manual_strategy_trades_in_sample = convert_to_orders_df_for_marketsim(df_manual_strategy_trades_in_sample)
    manual_strategy_trades_portfolio_value_in_sample = msc.compute_portvals(df_manual_strategy_trades_in_sample, start_value, 9.95, 0.005)
    manual_strategy_trades_portfolio_value_in_sample = manual_strategy_trades_portfolio_value_in_sample / manual_strategy_trades_portfolio_value_in_sample[0]

    # Manual Strategy out-sample Testing
    df_manual_strategy_trades_out_sample = ms.testPolicy(symbol, start_date_out_sample, end_date_out_sample, start_value)
    df_manual_strategy_trades_out_sample = convert_to_orders_df_for_marketsim(df_manual_strategy_trades_out_sample)
    manual_strategy_trades_portfolio_value_out_sample = msc.compute_portvals(df_manual_strategy_trades_out_sample, start_value, 9.95, 0.005)
    manual_strategy_trades_portfolio_value_out_sample = manual_strategy_trades_portfolio_value_out_sample / manual_strategy_trades_portfolio_value_out_sample[0]

    # Benchmark In Smaple Testing
    df_benchmark_in_sample = benchmark(symbol, start_date_in_sample, end_date_in_sample, start_value)
    benchMark_portfolio_value_in_sample = msc.compute_portvals(df_benchmark_in_sample, start_value, 9.95, 0.005)
    benchMark_portfolio_value_in_sample = benchMark_portfolio_value_in_sample / benchMark_portfolio_value_in_sample[0]

    # Benchmark Out Smaple Testing
    df_benchmark_out_sample = benchmark(symbol, start_date_out_sample, end_date_out_sample, start_value)
    benchMark_portfolio_value_out_sample = msc.compute_portvals(df_benchmark_out_sample, start_value, 9.95, 0.005)
    benchMark_portfolio_value_out_sample = benchMark_portfolio_value_out_sample / benchMark_portfolio_value_out_sample[0]

    #Strategy Learner in-sample Testing
    learner_in_sample = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner_in_sample.add_evidence(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample = learner_in_sample.testPolicy(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample = convert_to_orders_df_for_marketsim(df_strategy_learner_trades_in_sample)
    strategy_learner_trades_portfolio_value_in_sample = msc.compute_portvals(df_strategy_learner_trades_in_sample, start_value, 9.95, 0.005)
    strategy_learner_trades_portfolio_value_in_sample = strategy_learner_trades_portfolio_value_in_sample / strategy_learner_trades_portfolio_value_in_sample[0]

    # Strategy Learner out-sample Testing
    learner_out_sample = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner_out_sample.add_evidence(symbol, start_date_out_sample, end_date_out_sample, start_value)
    df_strategy_learner_trades_out_sample = learner_out_sample.testPolicy(symbol, start_date_out_sample, end_date_out_sample, start_value)
    df_strategy_learner_trades_out_sample = convert_to_orders_df_for_marketsim(df_strategy_learner_trades_out_sample)
    strategy_learner_trades_portfolio_value_out_sample = msc.compute_portvals(df_strategy_learner_trades_out_sample, start_value, 9.95, 0.005)
    strategy_learner_trades_portfolio_value_out_sample = strategy_learner_trades_portfolio_value_out_sample / strategy_learner_trades_portfolio_value_out_sample[0]

    # Plotting strategy learner vs. benchmark vs. manual strategy in-sample
    plt.figure(figsize=(10, 5))
    plt.title("Strategy Learner vs. Benchmark vs. ManualStrategy - In Sample")
    plt.xticks(rotation=20)
    plt.xlim(strategy_learner_trades_portfolio_value_in_sample.index.min(), strategy_learner_trades_portfolio_value_in_sample.index.max())
    plt.ylabel("Normalized Values")
    plt.plot(benchMark_portfolio_value_in_sample, label="Benchmark", color="purple", lw=0.8)
    plt.plot(manual_strategy_trades_portfolio_value_in_sample, label="Manual Learner", color="red", lw=0.8)
    plt.plot(strategy_learner_trades_portfolio_value_in_sample, label="Strategy Learner", color="blue", lw=0.8)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("experiment1_in_sample.png")
    plt.clf()

    # Plotting strategy learner vs. benchmark vs. manual strategy out-sample
    plt.figure(figsize=(10, 5))
    plt.title("Strategy Learner vs. Benchmark vs. ManualStrategy - Out Sample")
    plt.xticks(rotation=20)
    plt.xlim(strategy_learner_trades_portfolio_value_out_sample.index.min(), strategy_learner_trades_portfolio_value_out_sample.index.max())
    plt.ylabel("Normalized Values")
    plt.plot(benchMark_portfolio_value_out_sample, label="Benchmark", color="purple", lw=0.8)
    plt.plot(manual_strategy_trades_portfolio_value_out_sample, label="Manual Learner", color="red", lw=0.8)
    plt.plot(strategy_learner_trades_portfolio_value_out_sample, label="Strategy Learner", color="blue", lw=0.8)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("experiment1_out_sample.png")
    plt.clf()
    
def benchmark(symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    """
    The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM, and holding that position. 
    
    Parameters:
    symbol (str): The stock symbol to evaluate.
    sd (datetime): The start date of the evaluation period.
    ed (datetime): The end date of the evaluation period.
    sv (float): The starting value of the portfolio.
    
    Returns:
    pandas.DataFrame: A DataFrame representing the benchmark portfolio.
    """
    
    # Get the historical prices for the given symbol and date range
    prices_df = get_data([symbol], pd.date_range(sd, ed)) 
    prices_df = prices_df[symbol]     

    # Create a DataFrame to store the trades
    df_trades = pd.DataFrame(prices_df, index=prices_df.index)
    df_trades.drop(symbol, axis=1, inplace=True)
    df_trades["Symbol"] = symbol
    df_trades["Shares"] = 0

    # Create a DataFrame to store the benchmark portfolio
    benchmark_df = pd.DataFrame(prices_df, index=prices_df.index)
    benchmark_df.drop(symbol, axis=1, inplace=True)

    benchmark_df["Symbol"] = symbol
    benchmark_df["Shares"] = 0
    
    # Set the initial trade as a "BUY" order for 1000 shares
    benchmark_df.iloc[0] = [symbol, 1000]
    
    return benchmark_df

def convert_to_orders_df_for_marketsim(old_trades_df):
    new_df_trades = pd.DataFrame(index=old_trades_df.index, columns=['Symbol', 'Shares'])
    new_df_trades['Symbol'] = old_trades_df.columns.values[0]
    new_df_trades['Shares'] = old_trades_df.iloc[:, 0].values
    return new_df_trades