# Code implementing a ManualStrategy object (your Manual Strategy) in the strategy_evaluation/ directory. 
# It should implement testPolicy() which returns a trades data frame (see below). 
# The main part of this code should call marketsimcode as necessary to generate the plots used in the report. 
# NOTE: You will have to create this file yourself. 
import datetime as dt
import numpy as np 
import pandas as pd
from util import get_data
import indicators as indicators

def __init__(self):
    pass 

def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    return "agandhi301"

def testPolicy(symbol="JPM", start_date=dt.datetime(2008, 1, 1), end_date=dt.datetime(2009, 1, 31), starting_value=100000,  		  	   		  		 		  		  		    	 		 		   		 		  
):  
    debug = False
    # Date Range
    dates = pd.date_range(start_date, end_date)	 
    # Read in adjusted closing prices for given symbols, date range, automatically adds SPY  	
    prices_all = get_data([symbol], dates)  
    # Just the portfolio symbol
    prices_df = prices_all[symbol]	   		
    # Just the spy 		
    df_prices_SPY = prices_all['SPY']	  	 
        
    # Calculate the Technical Indicators for Portfolio Symbols     
    simple_moving_average, _ = indicators.calculate_simple_moving_average(prices_df)
    momentum = indicators.calculate_momentum(prices_df)
    commodity_channel_index = indicators.calculate_commodity_channel_index(prices_df)
    bollinger_band_percentage = indicators.calculate_bollinger_band_percentage(prices_df)
    macd_histogram = indicators.calculate_moving_average_convergence_divergence(prices_df)

    # Create a DataFrame to hold the trading signals
    signals = pd.DataFrame(index=prices_df.index)
    signals['sma_signal'] = 0.0
    signals['momentum_signal'] = 0.0
    signals['chi_signal'] = 0.0
    signals['bollinger_signal'] = 0.0
    signals['macd_signal'] = 0.0
    signals['final_signal'] = 0.0

    # Generate trading signals based on the indicators
    signals['sma_signal'][simple_moving_average > prices_df] = 1.0 # buy signal
    signals['sma_signal'][simple_moving_average < prices_df] = -1.0 # sell signal

    signals['momentum_signal'][momentum < -0.15] = 1.0 # buy signal
    signals['momentum_signal'][momentum > 0.15] = -1.0 # sell signal

    signals['chi_signal'][commodity_channel_index > 100] = -1.0 # sell signal
    signals['chi_signal'][commodity_channel_index < -100] = 1.0 # buy signal

    signals['bollinger_signal'][bollinger_band_percentage < 0.2] = 1.0  # buy signal
    signals['bollinger_signal'][bollinger_band_percentage > 0.8] = -1.0  # sell signal

    signals['macd_signal'][macd_histogram > 0] = 1.0
    signals['macd_signal'][macd_histogram < 0] = -1.0

    # Create a temporary DataFrame with the individual signals
    temp_df = signals[['sma_signal', 'momentum_signal', 'chi_signal', 'bollinger_signal', 'macd_signal']]
    # Calculate the mode along the row axis
    signals['final_signal'] = temp_df.mode(axis=1)[0]

    #create trade dataframe
    trades_df = pd.DataFrame(index = prices_df.index)
    trades_df['Symbol'] = 'JPM'
    trades_df['Order'] = np.NaN
    trades_df['Shares'] = 0
        
    share_holding = 0
    for date in prices_df.index:
        trades_df.loc[date,"Order"] = "SELL"
        if signals["final_signal"][date] == -1:
            if share_holding == 0:
                trades_df.loc[date,"Shares"] = -1000
                share_holding -= 1000
            elif share_holding == 1000:
                trades_df.loc[date, "Shares"] = -2000
                share_holding -= 2000
            else: trades_df.loc[date, "Shares"] = 0
        elif signals["final_signal"][date] == 1:
            trades_df.loc[date,"Order"] = "BUY"
            if share_holding == 0:
                trades_df.loc[date,"Shares"] = 1000
                share_holding += 1000
            elif share_holding == -1000:
                trades_df.loc[date, "Shares"] = 2000
                share_holding += 2000
            else: trades_df.loc[date, "Shares"] = 0
        else: 
            trades_df.loc[date,"Order"] = "NO TRADE"
            trades_df.loc[date, "Shares"] = 0
    
    if debug:
        print("prices df ")
        print(prices_df)
        print("simple moving average")
        print(simple_moving_average)
        print("momentum")
        print(momentum)
        print("commodity channel index")
        print(commodity_channel_index)
        print("bollinger band percentage")
        print(bollinger_band_percentage)
        print("macd histogram")
        print(macd_histogram)
        print("signals df")
        print(signals)
        print("trades df")
        print(trades_df)

    return trades_df