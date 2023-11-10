# Code implementing a ManualStrategy object (your Manual Strategy) in the strategy_evaluation/ directory. 
# It should implement testPolicy() which returns a trades data frame (see below). 
# The main part of this code should call marketsimcode as necessary to generate the plots used in the report. 
# NOTE: You will have to create this file yourself. 
import datetime as dt 
import pandas as pd
from util import get_data
import indicators as indicators

class ManualStrategy(object):

    def __init__(self):
        pass 

    def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        return "agandhi301"
    
    def testPolicy(self, symbol="JPM", start_date=dt.datetime(2008, 1, 1), end_date=dt.datetime(2009, 1, 31), starting_value=100000,  		  	   		  		 		  		  		    	 		 		   		 		  
    ):  
        # Date Range
        dates = pd.date_range(start_date, end_date)	 
        # Read in adjusted closing prices for given symbols, date range, automatically adds SPY  	
        prices_all = get_data(symbol, dates)  
        # Just the portfolio symbol
        prices_df = prices_all[symbol]	   		
        # Just the spy 		
        df_prices_SPY = prices_all['SPY']	  	 
            
        # Calculate the Technical Indicators for Portfolio Symbols     
        simple_moving_average = indicators.calculate_simple_moving_average(prices_df)
        momentum = indicators.calculate_momentum(prices_df)
        commodity_channel_index = indicators.calculate_commodity_channel_index(prices_df)
        bollinger_band_percentage = indicators.calculate_bollinger_band_percentage(prices_df)
        macd_histogram = indicators.calculate_moving_average_convergence_divergence(prices_df)

        