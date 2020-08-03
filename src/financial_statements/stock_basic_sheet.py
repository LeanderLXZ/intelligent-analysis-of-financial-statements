import time
import threading
import argparse
import tushare as ts
import numpy as np
import pandas as pd
from pandas import datetime as dt
from tqdm import tqdm


with open('token.txt', 'r') as f:
    token = f.readline()

ts.set_token(token)
tushare_api = ts.pro_api()


# 股票列表
df_list = []
for list_status in ['L', 'D', 'P']:
    df_i = tushare_api.stock_basic(
        exchange='',
        list_status=list_status,
        fields='ts_code, symbol, name, area, industry, fullname, enname,'
               'market, exchange, curr_type, list_status, list_date,'
               'delist_date, is_hs')
    df_list.append(df_i)
    
df = pd.concat(df_list)

df.to_csv('../../data/financial_statements/stock_basic_sheet.csv', index=False)
