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
        fields='ts_code')
    df_list.append(df_i)
    
df_all = pd.concat(df_list)


# 利润表
df = pd.DataFrame()
for ts_code in tqdm(df_all['ts_code'].values):
    df_i = tushare_api.income(ts_code=ts_code)
    df_i = df_i.drop_duplicates()
    df_i = df_i.reindex(index=df_i.index[::-1])
    df_i.insert(0, 'code', [c[:6] for c in df_i['ts_code']])
    df = df.append(df_i)
df = df.reset_index(drop=True)

df.to_csv('../财务信息表/利润表.csv', index=False)