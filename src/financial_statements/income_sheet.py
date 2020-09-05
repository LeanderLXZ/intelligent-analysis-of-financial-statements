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
    df_i = tushare_api.income(
        ts_code=ts_code,
        fields=
        'ts_code, ann_date, f_ann_date, end_date, report_type, comp_type,'
        'basic_eps, diluted_eps, total_revenue, revenue, int_income,'
        'prem_earned, comm_income, n_commis_income, n_oth_income,'
        'n_oth_b_income, prem_income, out_prem, une_prem_reser,'
        'reins_income, n_sec_tb_income, n_sec_uw_income, n_asset_mg_income,'
        'oth_b_income, fv_value_chg_gain, invest_income, ass_invest_income,'
        'forex_gain, total_cogs, oper_cost, int_exp, comm_exp, biz_tax_surchg,'
        'sell_exp, admin_exp, fin_exp, assets_impair_loss, prem_refund,'
        'compens_payout, reser_insur_liab, div_payt, reins_exp, oper_exp,'
        'compens_payout_refu, insur_reser_refu, reins_cost_refund,'
        'other_bus_cost, operate_profit, non_oper_income, non_oper_exp,'
        'nca_disploss, total_profit, income_tax, n_income, n_income_attr_p,'
        'minority_gain, oth_compr_income, t_compr_income, compr_inc_attr_p,'
        'compr_inc_attr_m_s, ebit, ebitda, insurance_exp, undist_profit,'
        'distable_profit, update_flag'
        )
    df_i = df_i.drop_duplicates()
    df_i = df_i.reindex(index=df_i.index[::-1])
    df_i.insert(0, 'code', [c[:6] for c in df_i['ts_code']])
    df = df.append(df_i)
df = df.reset_index(drop=True)

df.to_csv('../../data/financial_statements/income_sheet.csv', index=False)
