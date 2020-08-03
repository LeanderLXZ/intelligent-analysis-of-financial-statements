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
df = tushare_api.concept(src='ts')

df.to_csv('../财务信息表/概念股分类表.csv', index=False)
