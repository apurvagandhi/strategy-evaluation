"""
This file should be considered the entry point to the project. 
Code initializing/running all necessary files for the report.

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

def test_code():
    start_date_in_sample = dt.datetime(2008, 1, 1)
    end_date_in_sample = dt.datetime(2009, 12, 31)
    start_date_out_sample = dt.datetime(2010, 1, 1)
    end_date_out_sample = dt.datetime(2011, 12, 31)
    start_value = 100000
    symbol = "JPM"

    # Manual Strategy in-sample Testing
    df_manual_strategy_trades_in_sample = ms.testPolicy(symbol, start_date_in_sample, end_date_in_sample, start_value)
    manual_strategy_trades_portfolio_value_in_sample = msc.compute_portvals(df_manual_strategy_trades_in_sample, start_value, 0.0, 0.0)
    manual_strategy_cumulative_return_in_sample, manual_strategy_stdev_daily_return_in_sample, manual_strategy_average_daily_return_in_sample = get_statistics(manual_strategy_trades_portfolio_value_in_sample)
    manual_strategy_trades_portfolio_value_in_sample = manual_strategy_trades_portfolio_value_in_sample / manual_strategy_trades_portfolio_value_in_sample[0]
    # # Add vertical lines for LONG entry points
    # manual_strategy_long_entries_in_sample = df_manual_strategy_trades_in_sample[df_manual_strategy_trades_in_sample['Order'] == "BUY"].index
    # # Add vertical lines for SHORT entry points
    # manual_strategy_short_entries_in_sample = df_manual_strategy_trades_in_sample[df_manual_strategy_trades_in_sample['Order'] == "SELL"].index

    # Manual Strategy out-sample Testing
    df_manual_strategy_trades_out_sample = ms.testPolicy(symbol, start_date_out_sample, end_date_out_sample, start_value)
    manual_strategy_trades_portfolio_value_out_sample = msc.compute_portvals(df_manual_strategy_trades_out_sample, start_value, 0.0, 0.0)
    manual_strategy_cumulative_return_out_sample, manual_strategy_stdev_daily_return_out_sample, manual_strategy_average_daily_return_out_sample = get_statistics(manual_strategy_trades_portfolio_value_out_sample)
    manual_strategy_trades_portfolio_value_out_sample = manual_strategy_trades_portfolio_value_out_sample / manual_strategy_trades_portfolio_value_out_sample[0]
    # # Add vertical lines for LONG entry points
    # long_entries = df_manual_strategy_trades[df_manual_strategy_trades['Order'] == "BUY"].index
    # # Add vertical lines for SHORT entry points
    # short_entries = df_manual_strategy_trades[df_manual_strategy_trades['Order'] == "SELL"].index

    # Benchmark In Smaple Testing
    df_benchmark_in_sample = benchmark(symbol, start_date_in_sample, end_date_in_sample, start_value)
    benchMark_portfolio_value_in_sample = msc.compute_portvals(df_benchmark_in_sample, start_value, 0, 0)
    benchmark_cumulative_return_in_sample, benchmark_stdev_daily_return_in_sample, benchmark_average_daily_return_in_sample = get_statistics(benchMark_portfolio_value_in_sample)
    benchMark_portfolio_value_in_sample = benchMark_portfolio_value_in_sample / benchMark_portfolio_value_in_sample[0]

     # Benchmark Out Smaple Testing
    df_benchmark_out_sample = benchmark(symbol, start_date_out_sample, end_date_out_sample, start_value)
    benchMark_portfolio_value_out_sample = msc.compute_portvals(df_benchmark_out_sample, start_value, 0, 0)
    benchmark_cumulative_return_out_sample, benchmark_stdev_daily_return_out_sample, benchmark_average_daily_return_out_sample = get_statistics(benchMark_portfolio_value_out_sample)
    benchMark_portfolio_value_out_sample = benchMark_portfolio_value_out_sample / benchMark_portfolio_value_out_sample[0]

    #Strategy Learner in-sample Testing
    learner_in_sample = sl.StrategyLearner()
    learner_in_sample.add_evidence(symbol, start_date_in_sample,end_date_in_sample, start_value)
    
    df_strategy_learner_trades_in_sample = learner_in_sample.testPolicy(symbol, start_date_in_sample,end_date_in_sample, start_value)
    print("df_strategy_learner_trades_in_sample",df_strategy_learner_trades_in_sample)
    strategy_learner_trades_portfolio_value_in_sample = msc.compute_portvals(df_strategy_learner_trades_in_sample, start_value, 0.0, 0.0)
    strategy_learner_cumulative_return_in_sample, strategy_learner_stdev_daily_return_in_sample, strategy_learner_average_daily_return_in_sample = get_statistics(strategy_learner_trades_portfolio_value_in_sample)
    strategy_learner_trades_portfolio_value_in_sample = strategy_learner_trades_portfolio_value_in_sample / strategy_learner_trades_portfolio_value_in_sample[0]

    # Strategy Learner out-sample Testing
    learner_out_sample = sl.StrategyLearner()
    learner_out_sample.add_evidence(symbol, start_date_out_sample, end_date_out_sample, start_value)
    df_strategy_learner_trades_out_sample = learner_out_sample.testPolicy(symbol, start_date_out_sample, end_date_out_sample, start_value)
    strategy_learner_trades_portfolio_value_out_sample = msc.compute_portvals(df_strategy_learner_trades_out_sample, start_value, 0.0, 0.0)
    strategy_learner_cumulative_return_out_sample, strategy_learner_stdev_daily_return_out_sample, strategy_learner_average_daily_return_out_sample = get_statistics(strategy_learner_trades_portfolio_value_out_sample)
    strategy_learner_trades_portfolio_value_out_sample = strategy_learner_trades_portfolio_value_out_sample / strategy_learner_trades_portfolio_value_out_sample[0]

    # Create a DataFrame for the summary table
    summary_table = pd.DataFrame({
        'Benchmark In-Sample': [benchmark_cumulative_return_in_sample, benchmark_stdev_daily_return_in_sample, benchmark_average_daily_return_in_sample],
        'Benchmark Out-Sample': [benchmark_cumulative_return_out_sample, benchmark_stdev_daily_return_out_sample, benchmark_average_daily_return_out_sample],
        'Manual Strategy In-Sample': [manual_strategy_cumulative_return_in_sample, manual_strategy_stdev_daily_return_in_sample, manual_strategy_average_daily_return_in_sample],
        'Manual Strategy Out-Sample': [manual_strategy_cumulative_return_out_sample, manual_strategy_stdev_daily_return_out_sample, manual_strategy_average_daily_return_out_sample],
        'Strategy Learner In-Sample': [strategy_learner_cumulative_return_in_sample, strategy_learner_stdev_daily_return_in_sample, strategy_learner_average_daily_return_in_sample],
        'Strategy Learner Out-Sample': [strategy_learner_cumulative_return_out_sample, strategy_learner_stdev_daily_return_out_sample, strategy_learner_average_daily_return_out_sample],
    }, index=['Cumulative Return', 'STDEV of Daily Returns', 'Mean of Daily Returns'])

    print(summary_table)

    plt.figure(figsize=(10, 5))
    plt.title("Benchmark vs. Manual Strategy In Sample")
    plt.xticks(rotation=20)
    plt.xlim(manual_strategy_trades_portfolio_value_in_sample.index.min(), manual_strategy_trades_portfolio_value_in_sample.index.max())
    plt.ylabel("Normalized Values")
    plt.plot(benchMark_portfolio_value_in_sample, label="Benchmark", color="purple", lw=0.8)
    plt.plot(manual_strategy_trades_portfolio_value_in_sample, label="Manual Strategy", color="red", lw=0.8)
    # for entry in manual_strategy_long_entries_in_sample:
    #     plt.axvline(x=entry, label="LONG", color='blue', linestyle='--')
    # for entry in manual_strategy_short_entries_in_sample:
    #     plt.axvline(x=entry, label="SHORT", color='black', linestyle='--')
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("images/manual_strategy_in_sample.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Benchmark vs. Manual Strategy Out Sample")
    plt.xticks(rotation=20)
    plt.xlim(manual_strategy_trades_portfolio_value_out_sample.index.min(), manual_strategy_trades_portfolio_value_out_sample.index.max())
    plt.ylabel("Normalized Values")
    plt.plot(benchMark_portfolio_value_out_sample, label="Benchmark", color="purple", lw=0.8)
    plt.plot(manual_strategy_trades_portfolio_value_out_sample, label="Manual Strategy", color="red", lw=0.8)
    # for entry in long_entries:
    #     plt.axvline(x=entry, color='blue', linestyle='--')
    # for entry in short_entries:
    #     plt.axvline(x=entry, color='black', linestyle='--')
    plt.grid()
    plt.legend(loc="best")
    plt.savefig("images/manual_strategy_out_sample.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Strategy Learner vs. Benchmark In Sample")
    plt.xticks(rotation=20)
    plt.xlim(strategy_learner_trades_portfolio_value_in_sample.index.min(), strategy_learner_trades_portfolio_value_in_sample.index.max())
    plt.ylabel("Normalized Values")
    plt.plot(benchMark_portfolio_value_in_sample, label="Benchmark", color="purple", lw=0.8)
    plt.plot(strategy_learner_trades_portfolio_value_in_sample, label="Strategy Learner", color="red", lw=0.8)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("images/strategy_learner_in_sample.png")
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.title("Strategy Learner vs. Benchmark Out Sample")
    plt.xticks(rotation=20)
    plt.xlim(strategy_learner_trades_portfolio_value_out_sample.index.min(), strategy_learner_trades_portfolio_value_out_sample.index.max())
    plt.ylabel("Normalized Values")
    plt.plot(benchMark_portfolio_value_out_sample, label="Benchmark", color="purple", lw=0.8)
    plt.plot(strategy_learner_trades_portfolio_value_out_sample, label="Strategy Learner", color="red", lw=0.8)
    plt.grid()
    plt.legend(loc="best")
    plt.savefig("images/strategy_learner_out_sample.png")
    plt.clf()

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
    plt.savefig("images/experiment1_in_sample.png")
    plt.clf()

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
    plt.savefig("images/experiment1_out_sample.png")
    plt.clf()

def benchmark(symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    """
    The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM, and holding that position. 
    """
    
    prices_df = get_data([symbol], pd.date_range(sd, ed)) 
    prices_df = prices_df[symbol]     

    df_trades = pd.DataFrame(prices_df, index=prices_df.index)
    df_trades.drop(symbol, axis=1, inplace=True)
    df_trades["Order"] = None
    df_trades["Shares"] = 0
    df_trades["Symbol"] = symbol

    benchmark_df = pd.DataFrame(prices_df, index=prices_df.index)
    benchmark_df.drop(symbol, axis=1, inplace=True)
    benchmark_df["Order"] = None
    benchmark_df["Shares"] = 0
    benchmark_df["Symbol"] = symbol
    benchmark_df.iloc[0] = ["BUY", 1000, symbol]
    return benchmark_df

def get_statistics(portfolio_values):
    daily_ret = portfolio_values.copy()
    daily_ret[1:] = (daily_ret[1:] / daily_ret[:-1].values) - 1
    daily_ret.loc[daily_ret.index[0]] = 0

    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    average_daily_return = daily_ret[1:].mean()
    stdev_daily_return = daily_ret[1:].std()

    return cumulative_return, stdev_daily_return, average_daily_return

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    test_code()