from binance.client import Client
import os
from dotenv import load_dotenv
import pandas as pd
# import pytz 
# from tqdm import tqdm 
# from datetime import datetime , timedelta


# NEXT IMPROVEMENT : ADD THE PROGRESS BAR WHILE THE API IS FETCHING DATA USING pytz IN IST

load_dotenv()

api_key = os.getenv("binance_api_key")
secret_key = os.getenv("binance_secret_key")
client = Client(api_key , secret_key)


print("Getting data for model training (historical data via binance REST api client)")
print("Fetching bitcoin historical data(1 year ago) in US Dollar")

historical_data =  client.get_historical_klines(
                    symbol="BTCUSDT",
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    start_str="1 year ago UTC"
)

print(f"Download is completed the overall data lenght is {len(historical_data)}")

columns = ['Open Time', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume',
           'Close Time', 'Quote Asset Volume', 'Number of Trades', 
           'Taker Buy Base', 'Taker Buy Quote', 'Ignore']


df_main = pd.DataFrame(historical_data , columns=columns)

df_required = df_main[['Open Time' , 
                       'Open Price',
                       'High Price',
                       'Low Price',
                       'Close Price',
                       'Volume' 
                       ]]

save_dir = "/home/ateeksh/Real_Time_Crypto_Pump_And_Dump_Detection/data"
file_name = "binace_btc_1yr_1minGap_data.csv"
full_path = os.path.join(save_dir , file_name)
df_required.to_csv(full_path , index=False)

print(f"Data has been saved successfully , the file name is {full_path}")