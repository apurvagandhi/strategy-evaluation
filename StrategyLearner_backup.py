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
import random
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
  		  	   		  		 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		  		 		  		  		    	 		 		   		 		  
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
        # Reading prices from the csv file into data frame
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
        
        x_train = pd.concat((simple_moving_average_ratio, bollinger_band_percentage, momentum), axis=1)
        x_train = x_train[:lookback_window * -1].values
        print("x_train shape:", x_train.shape)
        print("x_train:\n", x_train[:])
        # ********************************
        # Possible problem that I think is that x_train data is 2d array.
        # I think it should be 1D array so need to convert x_train to 1D Array
        # If I don't train it right, how am I going to get test result back correct. 
        # 
        # 
        # **********************************
        # for x in x_train:
        #     print(x)

        y_train = np.zeros(prices_df.shape[0] - lookback_window, dtype = int)
                
        buy_signal = 0.015 + self.impact
        print("buy_signal:", buy_signal)
        sell_signal = -0.015 - self.impact
        print  ("sell_signal:", sell_signal)        
        for i in range(prices_df.shape[0] - lookback_window):
            threshold = (prices_df.iloc[i + lookback_window, 0] / prices_df.iloc[i, 0]) - 1.0
            print("threshold: ", threshold)
            if threshold > buy_signal:
                y_train[i] = 1
            elif threshold < sell_signal:
                y_train[i] = -1
            else: 
                y_train[0] = 0
        print("y_train.shape", y_train.shape)
        print("y_train", y_train)
        
        self.learner.add_evidence(x_train, y_train) 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		  		 		  		  		    	 		 		   		 		  
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  		 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        # Reading prices from the csv file into data frame
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

        x_test = pd.concat((simple_moving_average_ratio, bollinger_band_percentage, momentum), axis=1)
        x_test = x_test.values	
        print("x_test:\n")
        print(x_test)
        # for x in x_test:
        #     print(x)
        y_test = self.learner.query(x_test)
        print("y_test", y_test)
        df_trades = pd.DataFrame(index=prices_df.index, columns=['Symbol', 'Order', 'Shares'])
        # df_trades['Symbol'] = 'JPM'
        # df_trades['Order'] = np.NaN
        # df_trades['Shares'] = 1000
        # print("df_trades", df_trades)

        # # Generate buy and sell signals based on the model's prediction
        # df_trades.loc[y_test == 1, 'Order'] = 'BUY'
        # df_trades.loc[y_test == -1, 'Order'] = 'SELL'
        
        # # Generate buy and sell signals based on the model's prediction
        # df_trades.loc[y_test == 1, 'Shares'] = 1000.0
        # df_trades.loc[y_test == -1, 'Shares'] = -1000.0

        # # Adjust the trades to ensure net holdings are constrained to -1000, 0, and 1000
        # df_trades['Shares'] = df_trades['Shares'].diff()
        # df_trades.iloc[0] = 1000.0 if y_test[0] == 1 else -1000.0
       
        return df_trades		 
    
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