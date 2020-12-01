### Utility module with useful functions ###

import pandas as pd
import numpy as np
import math
import os

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import yfinance as yf
#from yahoofinancials import YahooFinancials

def download_price(ticker='SPY', start_dt='1980-01-01', end_dt='2020-12-31'):
    ''' 
    Download ticker historical prices from Yahoo Finance
    Rename the Adj Close price column name to ticker, drop index so Date becomes a column
    Save as csv file
    '''
    df_prc = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
    
    if df_prc.shape[0]==0:
        raise ValueError('Please double check the ticker, start date and end date.')
    
    df_prc.rename(columns={'Adj Close': ticker}, inplace=True)
    df_prc.reset_index(drop=False, inplace=True)
    df_prc.sort_values(by=['Date'])

    price_data_path = os.getcwd() + '/price_data/'
    price_file_name = \
        '{0}_{1}_{2}.csv'.format(ticker, start_dt.replace("-",""), end_dt.replace("-",""))
    df_prc.to_csv(price_data_path + price_file_name, sep='|', index=False)

    return df_prc



def read_price(ticker, start_dt, end_dt):
    '''
    Read already downloaded historical price file for ticker with specified start and end dates
    '''
    price_data_path = os.getcwd() + '/price_data/'
    price_file_name = \
        '{0}_{1}_{2}.csv'.format(ticker, start_dt.replace("-",""), end_dt.replace("-",""))
    df_prc = pd.read_csv(price_data_path + price_file_name, sep='|', parse_dates=['Date'])

    return df_prc



def line_plot(df, ticker):
    '''
    Line plot ticker price in df
    '''
    if ticker not in df.columns:
        raise ValueError('Ticker {} data is not present in the input DataFrame.'.format(ticker))

    dt_min = str(df['Date'].min().date())
    dt_max = str(df['Date'].max().date())
    
    plt.figure(figsize=(6,4))
    plt.plot(df['Date'], df[ticker])
    plt.xticks(rotation=45)
    plt.ylabel('Price ($)')
    plt.title('Historical Adjusted Price for {0}\n {1} to {2}'.format(ticker, dt_min, dt_max))
    plt.show()

    return None



def add_bbvalue(df, ticker, window=20):
    '''
    Add Bollinger Band value ((price - rolling mean) / rolling std) for ticker in df.
    '''
    if ticker not in df.columns:
        raise ValueError('Ticker {} data is not present in the input DataFrame.'.format(ticker))
       
    df['rm'] = df[ticker].rolling(window=window).mean()
    df['rstd'] = df[ticker].rolling(window=window).std()
    df['bbvalue'+str(window)] = (df[ticker] - df.rm) / df.rstd
    
    df.drop(columns=['rm', 'rstd'], inplace=True)
    
    return df
    

    
def add_so(df, ticker, window=20):
    '''
    Add Stochastic Oscillator for ticker in df.
    '''
    if ticker not in df.columns:
        raise ValueError('Ticker {} data is not present in the input DataFrame.'.format(ticker))

    df['price_max'] = df[ticker].rolling(window=window).max()
    df['price_min'] = df[ticker].rolling(window=window).min()
    df['so'+str(window)] = (df[ticker] - df.price_min) / (df.price_max - df.price_min)
    
    df.drop(columns=['price_max', 'price_min'], inplace=True)
    
    return df



def add_rtrn(df, ticker, mode='b', days=1):
    '''
    Add return of x days with mode of 'b' (backward) or 'f' (forward) looking.
    '''
    if ticker not in df.columns:
        raise ValueError('Ticker {} data is not present in the input DataFrame.'.format(ticker))
    if mode not in ['b', 'f']:
        raise ValueError('Input parameter mode needs to be either "b" (backward) or "f" (forward) looking.')
    if days<0:
        raise ValueError('Input parameter days needs to be a positive integer value.')
    
    if mode=='b':
        df['rtrn_{0}_{1}d'.format(mode, str(days))] = df[ticker].pct_change(periods=days)
    else:
        df['rtrn_{0}_{1}d'.format(mode, str(days))] = df[ticker].pct_change(periods=days).shift(-days)

    return df



def add_rsi(df, ticker, window=14):
    '''
    Add Relative Strength Index for ticker in df.
    '''
    if ticker not in df.columns:
        raise ValueError('Ticker {} data is not present in the input DataFrame.'.format(ticker))
    
    df_rsi = pd.DataFrame(df[ticker].copy())
    df_rsi = add_rtrn(df_rsi, ticker, 'b', 1)
    df_rsi['rtrn_b_1d_u'] = (np.abs(df_rsi.rtrn_b_1d) + df_rsi.rtrn_b_1d) / 2.0
    df_rsi['rtrn_b_1d_d'] = (np.abs(df_rsi.rtrn_b_1d) - df_rsi.rtrn_b_1d) / 2.0
    for idx in range(window, df.shape[0]):
        if idx==window:
            df_rsi.at[idx, 'avg_u'] = df_rsi.rtrn_b_1d_u[idx-window+1:idx+1].sum() / window
            df_rsi.at[idx, 'avg_d'] = df_rsi.rtrn_b_1d_d[idx-window+1:idx+1].sum() / window
        else:
            df_rsi.at[idx, 'avg_u'] = (df_rsi.at[idx, 'rtrn_b_1d_u'] + (window - 1) * df_rsi.at[idx-1, 'avg_u']) / window
            df_rsi.at[idx, 'avg_d'] = (df_rsi.at[idx, 'rtrn_b_1d_d'] + (window - 1) * df_rsi.at[idx-1, 'avg_d']) / window
    df_rsi['rs'] = df_rsi.avg_u / df_rsi.avg_d
    df_rsi['rsi'] = df_rsi.rs / (1.0 + df_rsi.rs)
    
    df['rsi'+str(window)] = df_rsi.rsi
        
    return df
    
    
    
    
def calc_bt_pl(df, ticker='SPY', penalty=0.002):
    '''
    Given df with columns price, log_rtrn (actual log return) and pstn (planned position, long 1 flat 0 short -1), calculate pl.
    Assuming all of the fund available will be traded, with return penalty of -0.02%;
    '''
    
    if ticker in df.columns:
        df.rename(columns={ticker: 'price'}, inplace=True)
    
    if not set(['price', 'pstn']).issubset(set(df.columns)):
        raise KeyError('Input DataFrame has to contain columns "price" and "pstn" for this calculation.')
    
    df.reset_index(drop=True, inplace=True)
    df['price_sod'] = df['price'].shift(1)
    df.at[0, 'price_sod'] = df.at[0, 'price']
    df['pstn_sod'] = df['pstn'].shift(1)
    df.at[0, 'pstn_sod'] = 0.0
    df['pstn_sod'] = df.pstn_sod.astype(int)
    df.at[df.shape[0]-1, 'pstn'] = 0
    
    df['rtrn'] = df.price / df.price_sod - 1.0
    df['pstn_rtrn'] = df.pstn_sod * df.rtrn
    df.loc[df.pstn_sod.isin([1,-1]) & (df.pstn_sod!=df.pstn), 'pstn_rtrn'] -= penalty  
    df['pstn_rtrn_log'] = np.log(1.0 + df.pstn_rtrn)
        
    return round(df.price[df.shape[0]-1] / df.price[0], 4), round(math.exp(df.pstn_rtrn_log.sum()), 4)

    