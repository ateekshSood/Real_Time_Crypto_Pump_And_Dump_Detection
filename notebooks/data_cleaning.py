# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: real-time-crypto-pump-and-dump-detection
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %%
df = pd.read_csv('data/raw/binace_btc_1yr_1minGap_data.csv')
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %%
# converting open time from unix millisecond to normal ms 
df['Open Time'] = pd.to_datetime(df['Open Time'] , unit="ms")
df.set_index('Open Time' , inplace=True)
df.head()

# %%
#missing 1 min frequency gaps 


#if theres a 1 min gap then pd will add a new row for it and 
#fill the row with na stuff 

df = df.resample('1min').asfreq() 


#forward fill will filll these na columns with the last filled value 

df[['Open Price' , 'High Price' , 'Low Price' , 'Close Price']] = df[['Open Price' , 'High Price' , 'Low Price' , 'Close Price']].ffill()

#fill vol with 0 
df['Volume'] = df['Volume'].fillna(0)


# %%
df.info()

# %%
# new features which will measure the percentage chnage over the last 1 min , 3 min and 5 min 

df['Change_1min'] = df['Close Price'].pct_change(periods=1)
df['Change_3min'] = df['Close Price'].pct_change(periods=3)
df['Change_5min'] = df['Close Price'].pct_change(periods=5)



# %%
# past 15 min normal vol 

df['Vol_avg_15min'] = df['Volume'].rolling(window=15).mean()

#volume spike ration 

df['Volume_spike_ratio'] = df['Volume']/df['Vol_avg_15min']

#high low apread 

df['High_Low_spread'] = (df['High Price'] - df['Low Price'])/df['Low Price']

# %%
import ta 

# this tells that the thing is normal , oversold , or overbought 

df['Relative_strength_index'] = ta.momentum.RSIIndicator(close=df['Close Price'] , window=14).rsi()

# %%
#target var

df['Target_Pump'] = (
    (df['Change_5min'] > 0.005) &
    (df['Volume_spike_ratio'] > 3.0)
).astype(int)

#its pump time when the change in last 5 min is greater than 2%
#and the volume has spiked 3 times

# %%
df['Target_Pump'].value_counts()

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.dropna(inplace=True)

# %%
df.isnull().sum()

# %%
import os

file_path = "data/processed"
file_name = "processed_data.csv"
final_path = os.path.join(file_path , file_name)
df.to_csv(final_path)


# %%

# %%
