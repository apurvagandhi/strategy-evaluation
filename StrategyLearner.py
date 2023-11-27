""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Apurva Gandhi
GT User ID: agandhi301
GT ID: 903862828		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		   	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
import util as ut  		  	
import BagLearner as bl   		  		 		  		  		    	 		 		   		 		  
import indicators as indicators
import RTLearner as rt
  		  	   		  		 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # constructor  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.005, commission=9.95):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  		 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		  		 		  		  		    	 		 		   		 		  
        self.commission = commission  	
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)	  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # this method uses RTLearner and trains it for trading given the training data x and training data y		  	   		  		 		  		  		    	 		 		   		 		  
    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        # Get the stock prices for the given symbol and time frame
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_df = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Calculate the Technical Indicators for Portfolio Symbols
        simple_moving_average, simple_moving_average_ratio = indicators.calculate_simple_moving_average(prices_df)
        momentum = indicators.calculate_momentum(prices_df)
        commodity_channel_index = indicators.calculate_commodity_channel_index(prices_df)
        bollinger_band_percentage = indicators.calculate_bollinger_band_percentage(prices_df)
        macd_histogram = indicators.calculate_moving_average_convergence_divergence(prices_df)
        lookback_window = 5

        # Prepare the training data
        x_train = pd.concat((simple_moving_average_ratio, bollinger_band_percentage, momentum, commodity_channel_index, macd_histogram), axis=1)
        x_train.fillna(0, inplace=True)
        x_train = x_train[:(lookback_window * -1)].values
        y_train = np.zeros(prices_df.shape[0] - lookback_window, dtype=int)

        buy_signal = 0.02 + self.impact
        sell_signal = (0.02 + self.impact) * -1

        # Generate the training labels based on the returns
        for i in range(prices_df.shape[0] - lookback_window):
            ret = (prices_df.iloc[i + lookback_window, 0] / prices_df.iloc[i, 0]) - 1
            if ret > buy_signal:
                y_train[i] = 1
            elif ret < sell_signal:
                y_train[i] = -1
            else:
                y_train[0] = 0

        # Print the training data if verbose mode is enabled
        if self.verbose:
            print("buy_signal:", buy_signal)
            print("sell_signal:", sell_signal)
            print("x_train shape:", x_train.shape)
            print("x_train", x_train)
            print("y_train shape:", y_train.shape)
            print("y_train:", y_train)

        # Add the training data to the strategy learner
        self.learner.add_evidence(x_train, y_train)
  		  	   		  		 		  		  		    	 		 		   		 		  
    # This method tests the learner using data outside of the training data
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):
        """
        :param symbol: The stock symbol that you trained on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day.
        :rtype: pandas.DataFrame
        """

        # Get the price data for the given symbol and dates
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices_df = prices_all[syms]
        prices_SPY = prices_all["SPY"]

        # Calculate the Technical Indicators for Portfolio Symbols
        simple_moving_average, simple_moving_average_ratio = indicators.calculate_simple_moving_average(prices_df)
        momentum = indicators.calculate_momentum(prices_df, window_size=10)
        commodity_channel_index = indicators.calculate_commodity_channel_index(prices_df)
        bollinger_band_percentage = indicators.calculate_bollinger_band_percentage(prices_df)
        macd_histogram = indicators.calculate_moving_average_convergence_divergence(prices_df)

        # Concatenate the indicators into the feature matrix
        x_test = pd.concat((simple_moving_average_ratio, bollinger_band_percentage, momentum, commodity_channel_index, macd_histogram), axis=1)
        x_test = x_test.values

        # Query the learner to get the predicted labels
        y_test = self.learner.query(x_test)

        # Create a DataFrame to store the trades
        df_trades_new = prices_df.copy()
        df_trades_new.values[:, :] = 0

        holding = 0
        for i in range(y_test.shape[0]):
            if holding == 0:
                if y_test[i] > 1:
                    df_trades_new.iloc[i, 0] = 1000
                    holding += 1000
                elif y_test[i] < 0:
                    df_trades_new.iloc[i, 0] = -1000
                    holding -= 1000
            elif holding == 1000:
                if y_test[i] < 0:
                    df_trades_new.iloc[i, 0] = -2000
                    holding -= 2000
                elif y_test[i] == 0:
                    df_trades_new.iloc[i, 0] = -1000
                    holding -= 1000
            else:
                if y_test[i] > 0:
                    df_trades_new.iloc[i, 0] = 2000
                    holding += 2000
                elif y_test[i] == 0:
                    df_trades_new.iloc[i, 0] = 1000
                    holding += 1000

        # Print debug information if verbose is True
        if self.verbose:
            print("x_test shape:", x_test.shape)
            print("x_test", x_test)
            print("y_test shape:", y_test.shape)
            print("y_test:", y_test)
            print("df_trades", df_trades_new)

        return df_trades_new

    # This method returns the GT username of the student
    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "agandhi301"
  		  	   		  		 		  		  		    	 		 		   		 		    		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)  	   		  		 		  		  		    	 		 		   		 		  