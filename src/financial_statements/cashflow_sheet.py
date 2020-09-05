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

# 现金流量表
df = pd.DataFrame()
for ts_code in tqdm(df_all['ts_code'].values):
    df_i = tushare_api.cashflow(
        ts_code=ts_code,
        fields=
        'ts_code, ann_date, f_ann_date, end_date, comp_type, report_type,'
        'net_profit, finan_exp, c_fr_sale_sg, recp_tax_rends, n_depos_incr_fi,'
        'n_incr_loans_cb, n_inc_borr_oth_fi, prem_fr_orig_contr,'
        'n_incr_insured_dep, n_reinsur_prem, n_incr_disp_tfa, ifc_cash_incr,'
        'n_incr_disp_faas, n_incr_loans_oth_bank, n_cap_incr_repur,'
        'c_fr_oth_operate_a, c_inf_fr_operate_a, c_paid_goods_s,'
        'c_paid_to_for_empl, c_paid_for_taxes, n_incr_clt_loan_adv,'
        'n_incr_dep_cbob, c_pay_claims_orig_inco, pay_handling_chrg,'
        'pay_comm_insur_plcy, oth_cash_pay_oper_act, st_cash_out_act,'
        'n_cashflow_act, oth_recp_ral_inv_act, c_disp_withdrwl_invest,'
        'c_recp_return_invest, n_recp_disp_fiolta, n_recp_disp_sobu,'
        'stot_inflows_inv_act, c_pay_acq_const_fiolta, c_paid_invest,'
        'n_disp_subs_oth_biz, oth_pay_ral_inv_act, n_incr_pledge_loan,'
        'stot_out_inv_act, n_cashflow_inv_act, c_recp_borrow, proc_issue_bonds,'
        'oth_cash_recp_ral_fnc_act, stot_cash_in_fnc_act, free_cashflow,'
        'c_prepay_amt_borr, c_pay_dist_dpcp_int_exp,'
        'incl_dvd_profit_paid_sc_ms, oth_cashpay_ral_fnc_act,'
        'stot_cashout_fnc_act, n_cash_flows_fnc_act, eff_fx_flu_cash,'
        'n_incr_cash_cash_equ, c_cash_equ_beg_period, c_cash_equ_end_period,'
        'c_recp_cap_contrib, incl_cash_rec_saims, uncon_invest_loss,'
        'prov_depr_assets, depr_fa_coga_dpba, amort_intang_assets,'
        'lt_amort_deferred_exp, decr_deferred_exp, incr_acc_exp,'
        'loss_disp_fiolta, loss_scr_fa, loss_fv_chg, invest_loss,'
        'decr_def_inc_tax_assets, incr_def_inc_tax_liab, decr_inventories,'
        'decr_oper_payable, incr_oper_payable, others,'
        'im_net_cashflow_oper_act, conv_debt_into_cap,'
        'conv_copbonds_due_within_1y, fa_fnc_leases, end_bal_cash,'
        'beg_bal_cash, end_bal_cash_equ, beg_bal_cash_equ,'
        'im_n_incr_cash_equ, update_flag'
        )
    df_i = df_i.drop_duplicates()
    df_i = df_i.reindex(index=df_i.index[::-1])
    df_i.insert(0, 'code', [c[:6] for c in df_i['ts_code']])
    df = df.append(df_i)
df = df.reset_index(drop=True)

df.to_csv('../../data/financial_statements/cashflow_sheet.csv', index=False)
