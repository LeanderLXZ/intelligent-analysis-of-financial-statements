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


# 概念分类表
df_all = tushare_api.concept(src='ts')


# 概念股明细表
df = pd.DataFrame()
for code in tqdm(df_all['code'].values):
    df_i = tushare_api.concept_detail(
        id=code, fields='id, concept_name, ts_code, name, in_date, out_date')
    df_i = df_i.drop_duplicates()
    df_i.insert(0, 'code', [c[:6] for c in df_i['ts_code']])
    df = df.append(df_i)
df = df.reset_index(drop=True)

df.to_csv('../../data/financial_statements/'
          'concept_details_sheet_by_concept.csv', index=False)


# 股票列表
df_list = []
for list_status in ['L', 'D', 'P']:
    df_i = tushare_api.stock_basic(
        exchange='',
        list_status=list_status,
        fields='ts_code')
    df_list.append(df_i)
    
df_all = pd.concat(df_list)


# 概念股明细表
interval = 0.16
df = pd.DataFrame()
for ts_code in tqdm(df_all['ts_code'].values):
    time_remaining = interval - time.time() % interval
    time.sleep(time_remaining)
    df_i = tushare_api.concept_detail(
        ts_code=ts_code, 
        fields='id, concept_name, ts_code, name, in_date, out_date')
    df_i = df_i.drop_duplicates()
    df_i.insert(0, 'code', [c[:6] for c in df_i['ts_code']])
    df = df.append(df_i)
df = df.reset_index(drop=True)

df.to_csv('../../data/financial_statements/'
          'concept_details_sheet_by_stocks.csv.csv', index=False)