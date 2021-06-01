import os
import pickle
import requests
import socket
from decimal import Decimal
from os.path import isdir, join


def check_dirs(dir_list):
  for dir_path in dir_list:
    if not isdir(dir_path):
      os.makedirs(dir_path)

def round_exact(value, dec_digits=2):
  result = str(value).strip()
  if result != '':
    zero_count = dec_digits
    # index_dec 小数点位置
    index_dec = result.find('.')
    if index_dec > 0:
      # zero_count 小数点后有多少位
      zero_count = len(result[index_dec + 1:])
      # 小数点位数大于保留位数
      if zero_count > dec_digits:
        if int(result[index_dec + dec_digits + 1]) > 4:
          if value < 0:
            result = str(Decimal(result[:index_dec + dec_digits + 1])
                         - Decimal(str(pow(10, dec_digits * -1))))
          else:
            result = str(Decimal(result[:index_dec + dec_digits + 1])
                         + Decimal(str(pow(10, dec_digits * -1))))
        index_dec = result.find('.')
        result = result[:index_dec + dec_digits + 1]
        zero_count = 0
      else:
        zero_count = dec_digits - zero_count
    else:
      result += '.'
    for i in range(zero_count):
      result += '0'
  return float(result)

def save_df(df, file_dir, file_name, save_format, verbose=True, index=True):
  check_dirs([file_dir])
  if save_format == 'pickle':
    file_path = join(file_dir, file_name + '.p')
    if verbose:
      print('保存 {} ...'.format(file_path))
    df.to_pickle(file_path)
  elif save_format == 'csv':
    file_path = join(file_dir, file_name + '.csv')
    if verbose:
      print('保存 {} ...'.format(file_path))
    df.to_csv(file_path, index=index)
  elif save_format == 'excel':
    file_path = join(file_dir, file_name + '.xlsx')
    if verbose:
      print('保存 {} ...'.format(file_path))
    df.to_excel(file_path)
  else:
    raise ValueError

def load_pkl(data_path, verbose=True):
  """Load data from pickle file."""
  with open(data_path, 'rb') as f:
    if verbose:
      print('读取 {}...'.format(f.name))
    return pickle.load(f)

import dis

def safe_get(func, *args, **kwargs):
  result = None
  i = 0
  while result is None and i < 100:
    try:
      result = func(*args, **kwargs)
    except socket.error:
      print('Time Out!')
    except KeyboardInterrupt:
      exit()
    # except Exception:
    #   pass
    i += 1
  if result is not None:
    if i != 1:
      print('safe_get() retried: {} times'.format(i))
    return result
  else:
    raise ValueError('Tried more than 100 times ang got no response!')

def send_notice_ifttt(event_name, key, text):
    url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key+""
    payload = "{\n    \"value1\": \""+text+"\"\n}"
    headers = {
    'Content-Type': "application/json",
    'User-Agent': "PostmanRuntime/7.15.0",
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Postman-Token': "a9477d0f-08ee-4960-b6f8-9fd85dc0d5cc,"
                     "d376ec80-54e1-450a-8215-952ea91b01dd",
    'Host': "maker.ifttt.com",
    'accept-encoding': "gzip, deflate",
    'content-length': "63",
    'Connection': "keep-alive",
    'cache-control': "no-cache"
    }
 
    response = requests.request(
      "POST", url, data=payload.encode('utf-8'), headers=headers)
    