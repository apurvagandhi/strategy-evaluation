import datetime as dt
from matplotlib import pyplot as plt
import pandas as pd
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as msc

def author():
    """
    : return: The GT username of the student
    : rtype: str
    """
    return "agandhi301"

def conduct_experiment2():
    start_date_in_sample = dt.datetime(2008, 1, 1)
    end_date_in_sample = dt.datetime(2009, 12, 31)
    start_value = 100000
    symbol = "JPM"

    #Strategy Learner in-sample Testing impact of 0.0
    learner_in_sample_1 = sl.StrategyLearner(verbose=False, impact=0.005, commission=0.0)
    learner_in_sample_1.add_evidence(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample_impact1 = learner_in_sample_1.testPolicy(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample_impact1 = convert_to_orders_df_for_marketsim(df_strategy_learner_trades_in_sample_impact1)
    strategy_learner_trades_portfolio_value_in_sample_impact1 = msc.compute_portvals(df_strategy_learner_trades_in_sample_impact1, start_value, 0.0, 0.005)
    strategy_learner_cumulative_return_in_sample_impact1, strategy_learner_stdev_daily_return_in_sample_impact1, strategy_learner_average_daily_return_in_sample_impact1 = get_statistics(strategy_learner_trades_portfolio_value_in_sample_impact1)
    strategy_learner_trades_portfolio_value_in_sample_impact1 = strategy_learner_trades_portfolio_value_in_sample_impact1 / strategy_learner_trades_portfolio_value_in_sample_impact1[0]

    #Strategy Learner in-sample Testing impact of 0.005
    learner_in_sample_2 = sl.StrategyLearner(verbose=False, impact=0.01, commission=0.0)
    learner_in_sample_2.add_evidence(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample_impact2 = learner_in_sample_2.testPolicy(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample_impact2 = convert_to_orders_df_for_marketsim(df_strategy_learner_trades_in_sample_impact2)
    strategy_learner_trades_portfolio_value_in_sample_impact2 = msc.compute_portvals(df_strategy_learner_trades_in_sample_impact2, start_value, 0.0, 0.01)
    strategy_learner_cumulative_return_in_sample_impact2, strategy_learner_stdev_daily_return_in_sample_impact2, strategy_learner_average_daily_return_in_sample_impact2 = get_statistics(strategy_learner_trades_portfolio_value_in_sample_impact2)
    strategy_learner_trades_portfolio_value_in_sample_impact2 = strategy_learner_trades_portfolio_value_in_sample_impact2 / strategy_learner_trades_portfolio_value_in_sample_impact2[0]

    #Strategy Learner in-sample Testing impact of 0.05
    learner_in_sample_3 = sl.StrategyLearner(verbose=False, impact=0.025, commission=0.0)
    learner_in_sample_3.add_evidence(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample_impact3 = learner_in_sample_3.testPolicy(symbol, start_date_in_sample,end_date_in_sample, start_value)
    df_strategy_learner_trades_in_sample_impact3 = convert_to_orders_df_for_marketsim(df_strategy_learner_trades_in_sample_impact3)
    strategy_learner_trades_portfolio_value_in_sample_impact3 = msc.compute_portvals(df_strategy_learner_trades_in_sample_impact3, start_value, 0.0, 0.025)
    strategy_learner_cumulative_return_in_sample_impact3, strategy_learner_stdev_daily_return_in_sample_impact3, strategy_learner_average_daily_return_in_sample_impact3 = get_statistics(strategy_learner_trades_portfolio_value_in_sample_impact3)
    strategy_learner_trades_portfolio_value_in_sample_impact3 = strategy_learner_trades_portfolio_value_in_sample_impact3 / strategy_learner_trades_portfolio_value_in_sample_impact3[0]

    # Create a DataFrame for the summary table
    summary_table = pd.DataFrame({
        'Strategy Learner In-Sample Impact 0.005': [strategy_learner_cumulative_return_in_sample_impact1, strategy_learner_stdev_daily_return_in_sample_impact1, strategy_learner_average_daily_return_in_sample_impact1],
        'Strategy Learner In-Sample Impact 0.01': [strategy_learner_cumulative_return_in_sample_impact2, strategy_learner_stdev_daily_return_in_sample_impact2, strategy_learner_average_daily_return_in_sample_impact2],
        'Strategy Learner In-Sample Impact 0.025': [strategy_learner_cumulative_return_in_sample_impact3, strategy_learner_stdev_daily_return_in_sample_impact3, strategy_learner_average_daily_return_in_sample_impact3],
    }, index=['Cumulative Return', 'STDEV of Daily Returns', 'Mean of Daily Returns'])

    with open("p8_results.txt", "a") as file:
        file.write("Summary table of experiment 2\n\n")
        file.write(summary_table.to_string())
        file.write("\n\n")

     # Plotting strategy learner in-sample with three different impacts
    plt.figure(figsize=(10, 5))
    plt.title("Strategy Learner In Sample Impact Comparision")
    plt.xticks(rotation=20)
    plt.xlim(strategy_learner_trades_portfolio_value_in_sample_impact1.index.min(), strategy_learner_trades_portfolio_value_in_sample_impact1.index.max())
    plt.ylabel("Normalized Values")
    plt.plot(strategy_learner_trades_portfolio_value_in_sample_impact1, label="Impact of 0.005", color="purple", lw=0.8)
    plt.plot(strategy_learner_trades_portfolio_value_in_sample_impact2, label="Impact of 0.01", color="red", lw=0.8)
    plt.plot(strategy_learner_trades_portfolio_value_in_sample_impact3, label="Impact of 0.025", color="black", lw=0.8)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("experiment2_strategy_learner_in_sample_impact.png")
    plt.clf()

def get_statistics(portfolio_values):
    daily_ret = portfolio_values.copy()
    daily_ret[1:] = (daily_ret[1:] / daily_ret[:-1].values) - 1
    daily_ret.loc[daily_ret.index[0]] = 0

    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    average_daily_return = daily_ret[1:].mean()
    stdev_daily_return = daily_ret[1:].std()

    return cumulative_return, stdev_daily_return, average_daily_return

def convert_to_orders_df_for_marketsim(old_trades_df):
    new_df_trades = pd.DataFrame(index=old_trades_df.index, columns=['Symbol', 'Shares'])
    new_df_trades['Symbol'] = old_trades_df.columns.values[0]
    new_df_trades['Shares'] = old_trades_df.iloc[:, 0].values
    return new_df_trades