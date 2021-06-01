import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime as dt
from utils import *


class StocksInfo(object):

  def __init__(self,
               api,
               date_now=None,
               period=20,
               today=False,
               batch_size=30,
               use_tqdm=True,
               cb_mode='tushare',
               uqer_cb_dir=None):
    """此类用于每日更新数据集
    api             tushare API
    date_now        指定当天日期，若不指定，则默认近日
    period          获取序列的周期(用于计算连板和连阳)
    today           若为今天，则置为True
    batch_size      实时行情的调用batch大小
    use_tqdm        使用tqdm显示进度条
    cb_mode         可转债列表调用方式
    uqer_cb_dir     优矿可转债列表目录
    """
    # tushare 接口
    self.api = api
    
    self.start_date = '20100104'
    if date_now:
      self.date_now = date_now
      self.today = False
    else:
      self.date_now = dt.now().strftime('%Y%m%d')
      self.today = True

    if today:
      self.today = True

    print('=' * 70)
    print('日期: ', self.date_now)
    print('时间: ', dt.now().time().strftime('%H:%M:%S'))
    print('-' * 70)

    # 计算周期
    self.period = period

    # 实时行情batch
    self.batch_size = batch_size

    # 交易日列表
    self.trade_date: np.ndarray = self._get_trade_date()
    self.trade_date_back: np.ndarray = self.trade_date[::-1]
    self.pre_date = self.trade_date[-2]

    # 今日股票列表
    self.ts_code_now = self.api.daily(trade_date=self.date_now)['ts_code'].values

    # 股票名称列表
    self.stock_name_dict: dict = self._get_name_dict()

    # 可转债信息和列表
    self.cb_mode = cb_mode
    self.uqer_cb_dir = uqer_cb_dir
    if cb_mode == 'tushare':
      self.stock2bond_dict, self.bond2stock_dict, \
          self.cb_stock_list, self.cb_bond_list, \
          self.cb_bond_name_dict = self._load_cvt_bond_tushare()
    elif cb_mode == 'tushare_old':
      self.stock2bond_dict, self.bond2stock_dict, \
          self.cb_stock_list, self.cb_bond_list, \
          self.cb_bond_name_dict = self._load_cvt_bond_tushare_old()
    elif cb_mode == 'uqer':
      self.stock2bond_dict, self.bond2stock_dict, \
          self.cb_stock_list, self.cb_bond_list, \
          self.cb_bond_name_dict = self._load_cvt_bond_uqer()
    else:
      raise ValueError(cb_mode)

    # 使用tqdm
    self.use_tqdm = use_tqdm

  def _get_trade_date(self):
    """获取交易日历"""
    print('获取交易日历...')
    df_sse = safe_get(
        self.api.trade_cal, exchange='SSE', start_date=self.start_date,
        end_date=self.date_now, is_open='1')
    df_szse = safe_get(
        self.api.trade_cal, exchange='SZSE', start_date=self.start_date,
        end_date=self.date_now, is_open='1')
    date_sse = df_sse['cal_date'].values
    date_szse = df_szse['cal_date'].values
    date = np.union1d(date_sse, date_szse)
    self.start_date = date[-self.period]
    return date

  def _get_name_dict(self):
    """股票名称"""
    print('获取股票名称...')
    df_list = []
    for list_status in ['L', 'D', 'P']:
      df_i = safe_get(
          self.api.stock_basic,
          exchange='',
          list_status=list_status,
          fields='ts_code, name')
      df_list.append(df_i)
    df = pd.concat(df_list)

    stock_name_dict = {}
    for _, info_row in df.iterrows():
      stock_name_dict[info_row['ts_code'][:6]] = info_row['name']
    return stock_name_dict

  def _stock_code_to_name(self, code):
    """安全地将股票代码转为名称"""
    try:
      name = self.stock_name_dict[code[:6]]
    except KeyError:
      print('股票名未找到！代码: ' + code)
      name = ''
    return name

  def _load_cvt_bond_uqer(self):
    """读取转债信息表 - 优矿"""
    file_names = [n for n in os.listdir(self.uqer_cb_dir) if n != '.DS_Store']
    file_names.sort()

    print('读取优矿转债信息表: {} ...'.format(file_names[-1]))
    cvt_bond_path = join(self.uqer_cb_dir, file_names[-1])
    df_cb = pd.read_csv(cvt_bond_path, index_col=0)

    cb_stock_list = []
    cb_bond_list = []
    stock2bond_dict = {}
    bond2stock_dict = {}
    cb_bond_name_dict = {}
    for _, row_i in df_cb.iterrows():
      stock_code = str(row_i['tickerEqu']).zfill(6)
      stock_name = row_i['secShortNameEqu']
      bond_code = str(row_i['tickerBond']).zfill(6)
      bond_name = row_i['secShortNameBond']

      # 不考虑EB
      if 'EB' in bond_name:
        continue

      cb_stock_list.append(stock_code)
      cb_bond_list.append(bond_code)
      stock2bond_dict[stock_code] = (bond_code, bond_name)
      bond2stock_dict[bond_code] = (stock_code, stock_name)
      cb_bond_name_dict[bond_code] = bond_name

    return stock2bond_dict, bond2stock_dict, \
           cb_stock_list, cb_bond_list, cb_bond_name_dict

  def _load_cvt_bond_tushare(self):
    """获取可转债表 - tushare"""
    print('获取tushare可转债表...')
    # 可转债列表
    df_cb_all = safe_get(self.api.cb_basic)
    df_cb_all = df_cb_all.drop_duplicates(subset='ts_code')
    cb_ts_code_list = df_cb_all['ts_code'].astype('str')
    cb_code_list = [code_i[:6] for code_i in cb_ts_code_list]
    df_cb_all['ts_code'] = cb_code_list
    df_cb_all.rename(columns={'ts_code': 'bond_code'}, inplace=True)

    # 行情数据
    cb_list_all = set(df_cb_all['bond_code'].values)
    df_cb = self._get_realtime_quotes_safe(cb_list_all)
    df_cb = df_cb[df_cb['price'] != 0]
    df_cb.set_index('code', inplace=True)
    df_cb.index.name = 'bond_code'

    # 前一天转债数据不存在，则调用更前一天
    if self.pre_date in ['20200701']:
      self.pre_date = self.trade_date[-3]

    df_cb_daily = safe_get(
      self.api.cb_daily,
      trade_date=self.pre_date,
      fields='ts_code, trade_date'
    )

    df_cb_daily = df_cb_daily.drop_duplicates(subset='ts_code')
    df_cb_daily = df_cb_daily.dropna()
    df_cb_daily_list = df_cb_daily['ts_code'].apply(lambda x: x[:6]).values
    # df_cb = df_cb.loc[df_cb_daily_list]
    df_cb = df_cb.reindex(df_cb_daily_list)

    # 选出目前可转债列表
    df_cb_all.set_index('bond_code', inplace=True)
    df = df_cb_all.loc[df_cb.index]
    df.reset_index(inplace=True)

    # 生成可转债列表和dict
    cb_stock_list = []
    cb_bond_list = []
    stock2bond_dict = {}
    bond2stock_dict = {}
    cb_bond_name_dict = {}

    for _, row_i in df.iterrows():
      stock_code = str(row_i['stk_code']).zfill(6)[:6]
      stock_name = row_i['stk_short_name']
      bond_code = str(row_i['bond_code']).zfill(6)[:6]
      bond_name = row_i['bond_short_name']

      # 不考虑EB
      if 'EB' in bond_name:
        continue

      cb_stock_list.append(stock_code)
      cb_bond_list.append(bond_code)
      stock2bond_dict[stock_code] = (bond_code, bond_name)
      bond2stock_dict[bond_code] = (stock_code, stock_name)
      cb_bond_name_dict[bond_code] = bond_name

    return stock2bond_dict, bond2stock_dict, \
           cb_stock_list, cb_bond_list, cb_bond_name_dict

  def _load_cvt_bond_tushare_old(self):
    """获取可转债表 - tushare"""
    print('获取tushare可转债表...')
    # 可转债列表
    df_cb_all = safe_get(ts.new_cbonds, default=0)
    df_cb_all = df_cb_all.drop_duplicates(subset='bcode')
    df_cb_all['bcode'] = df_cb_all['bcode'].astype('str')

    # 行情数据
    cb_list_all = set(df_cb_all['bcode'].values)
    df_cb = self._get_realtime_quotes_safe(cb_list_all)
    df_cb = df_cb[df_cb['price'] != 0]
    df_cb.set_index('code', inplace=True)
    df_cb.index.name = 'bcode'

    # 选出目前可转债列表
    df_cb_all.set_index('bcode', inplace=True)
    df = df_cb_all.loc[df_cb.index]
    df.reset_index(inplace=True)

    # 生成可转债列表和dict
    cb_stock_list = []
    cb_bond_list = []
    stock2bond_dict = {}
    bond2stock_dict = {}
    cb_bond_name_dict = {}
    for _, row_i in df.iterrows():
      stock_code = str(row_i['scode']).zfill(6)
      stock_name = self._stock_code_to_name(stock_code)
      bond_code = str(row_i['bcode']).zfill(6)
      bond_name = row_i['bname']

      # 不考虑EB
      if 'EB' in bond_name:
        continue

      cb_stock_list.append(stock_code)
      cb_bond_list.append(bond_code)
      stock2bond_dict[stock_code] = (bond_code, bond_name)
      bond2stock_dict[bond_code] = (stock_code, stock_name)
      cb_bond_name_dict[bond_code] = bond_name

    return stock2bond_dict, bond2stock_dict, \
           cb_stock_list, cb_bond_list, cb_bond_name_dict

  @staticmethod
  def _get_realtime_quotes_safe(code_list):
    """以安全方式获取行情数据(避免因timeout报错)"""
    if type(code_list) in [list, set, np.ndarray]:
      code_list = list(code_list)
      if len(list(code_list)[0]) != 6:
        code_list = [code[:6] for code in code_list]
    else:
      if len(code_list) != 6:
        code_list = code_list[:6]
    return safe_get(ts.get_realtime_quotes, code_list)

  def _get_data_batch_generator(self, code_list, batch_size=None):
    """获取数据batch - 用于加速接口调"""
    for start in range(0, len(code_list), batch_size):
      end = start + batch_size
      code_list_batch = code_list[start:end]
      data_batch = self._get_realtime_quotes_safe(code_list_batch)
      yield code_list_batch, data_batch

  def _get_data_batch(self, code_list, i_batch):
    code_list_batch = code_list[i_batch * self.batch_size:
                                (i_batch + 1) * self.batch_size]
    data_batch = self._get_realtime_quotes_safe(code_list_batch)
    return code_list_batch, data_batch

  @staticmethod
  def _get_real_time_batch(i, code_list_batch, data_batch):

    code = code_list_batch[i]
    df = data_batch[data_batch['code'] == code]
    name = df['name'][i]
    sell1_v = df['a1_v'][i]
    sell1_v = 0 if sell1_v == '' else int(sell1_v)
    buy1_v = df['b1_v'][i]
    buy1_v = 0 if buy1_v == '' else int(buy1_v)

    price = float(df['price'][i])
    pre_close = float(df['pre_close'][i])

    # 自己计算涨幅
    pct_change = ((price - pre_close) / pre_close) * 100

    return code, name, price, pre_close, sell1_v, buy1_v, pct_change

  @staticmethod
  def _get_sector(code):
    """获取板块名称"""
    # 科创板
    if code[:3] == '688':
      return 'kcb'
    # 上交所
    elif code[0] == '6':
      return 'sh'
    # 深交所
    elif code[0] == '0':
      return 'sz'
    #创业板
    elif code[0] == '3':
      return 'cyb'
    else:
      return ''
