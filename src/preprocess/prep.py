import os
from re import I
import numpy as np
import pandas as pd
from datetime import date
from tqdm import tqdm
import xlrd
from os.path import join


# Check if directories exit or not
def check_dirs(path_list):
    for dir_path in path_list:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


# Remove a file
def remove_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
            

class DataPreprocess(object):
    
    def __init__(self,
                 data_path,
                 classification_path,
                 delist_path,
                 save_path,
                 info_path,
                 drop_list=[]):
        
        self.data_path = data_path
        self.classification_path = classification_path
        self.delist_path = delist_path
        self.save_path = save_path
        self.drop_list = drop_list
        check_dirs([self.save_path])
        
        df_cn_en = pd.read_csv(join(self.data_path, 'cn_to_en.csv'))
        self.cn_to_en = {
            c: e for c, e in zip(df_cn_en['cn_name'], df_cn_en['en_name'])}

        self.sheet_name_list = [
            'balance_industry', 'balance_bank', 'balance_security',
            'balance_insurance', 'income_industry', 'income_bank',
            'income_security', 'income_insurance', 'cash_flow_industry',
            'cash_flow_bank', 'cash_flow_security', 'cash_flow_insurance'
        ]
        self.feature_dict = self.get_feature_dict()
        
        self.df_info = pd.read_csv(info_path)
        self.df_info['ticker'] = \
            self.df_info['ticker'].apply(lambda x: str(x).zfill(6))
    
    def get_feature_dict(self):
        wb = xlrd.open_workbook(
            join(self.data_path, 'feature_dictionaries.xls'),
            formatting_info=True)
        
        feature_dict = dict()
        for sheet_name in self.sheet_name_list:
            feature_dict[sheet_name] = [[], []]
            ws = wb.sheet_by_name(sheet_name)
            for i in range(3, ws.nrows):
                feature_dict[sheet_name][0].append(ws.cell_value(i, 0))
                feature_dict[sheet_name][1].append(
                    [ws.cell_value(i, 0),
                     ws.cell_value(i, 1),
                     ws.cell_value(i, 2)])
        return feature_dict

    def get_new_listed(self, end_date):
        new_listed = set(self.df_info[
            (self.df_info['listDate'] >= end_date) |
            (self.df_info['listStatusCD'] == 'UN') |
            (self.df_info['listStatusCD'] == 'O')]['ticker'])
        return list(new_listed)
    
    def get_delisted_tickers(self):
        
        # Copy halt dataframe
        df_delist = self.df_info.copy()

        # Get delist stocks
        # df_delist = df_delist[df_delist['listStatusCD'] == 'DE']
        df_delist = df_delist[
            (df_delist['listStatusCD'] == 'DE') &
            (df_delist['ticker'].apply(lambda x: len(x) == 6))]

        # Drop columns
        df_delist = df_delist[['ticker', 'secShortName', 'delistDate']]

        # Normalize ticker
        df_delist['ticker'] = \
            df_delist['ticker'].apply(lambda x: str(x).zfill(6))

        # Delete duplicates (with different halt dates)
        df_delist.drop_duplicates(inplace=True)

        delisted = set(df_delist['ticker'])
        
        df_delist = pd.read_csv(self.delist_path, dtype={'TICKER': str})
        df_delist = df_delist[df_delist['DELETE_FLAG_FINAL'] == 1]
        delisted_delete = set(df_delist['TICKER'])
        
        return delisted, delisted_delete
    
    def get_concat_df(self,
                      data_path,
                      name,
                      post_fix='',
                      where=True,
                      columns=None):
        
        def _get_df(_sheet_type):
            _path = join(
                join(data_path, name + '_' + _sheet_type + post_fix + '.csv'))
            if columns:
                df =  pd.read_csv(_path, dtype={'ticker': str})[columns]
            else:
                df =  pd.read_csv(_path, dtype={'ticker': str})
            if where:
                df['from'] = [_sheet_type] * df.shape[0]
            return df
        
        df_1 = _get_df('bank')
        df_2 = _get_df('insurance' )
        df_3 = _get_df('security')
        df_4 = _get_df('industry')
        
        df = pd.concat([df_1, df_2, df_3, df_4])
        
        df['ticker'] = df['ticker'].apply(lambda x: str(x).zfill(6))
        
        df = df.reset_index()
        df = df.drop(columns=['index'])
        
        return df
    
    def normalize(self,
                  file_path,
                  start_date=None,
                  end_date=None):
    
        df = pd.read_csv(file_path, dtype={'ticker': str})

        # Select mergedFlag == 1
        if 'mergedFlag' in df.columns:
            df = df[df['mergedFlag'] == 1]
        
        # Set start date and end date
        if start_date:
            df = df[df['endDate'] >= start_date]
        if end_date:
            df = df[df['endDate'] <= end_date]
        
        # Normalize ticker
        df['ticker'] = df['ticker'].apply(lambda x: str(x).zfill(6))

        # Delete Companies
        df.drop(df[df['ticker'].apply(
            lambda x: (x[0] in ['9', 'A', '2']) | (x in self.drop_list)
        )].index, inplace=True)
        
        # If is not balance sheet, Drop Q3 and then rename report type
        if 'CQ3' in set(df['reportType']):
            df = df.drop(
                df[df['reportType']=='Q3'].index, axis=0, errors='ignore')
        
        # Rename CQ3 to Q3
        type_dict = {
            'Q1': 'Q1',
            'S1': 'S1',
            'Q3': 'Q3',
            'CQ3': 'Q3',
            'A': 'A'
        }
        df['reportType'] = df['reportType'].apply(lambda x: type_dict[x])

        # Delete Columns
        df = df.drop(columns=[
            'Unnamed: 0',
            'Unnamed: 0.1',
            'secID',
            'partyID',
            'publishDate',
            'fiscalPeriod',
            'mergedFlag',
            'accoutingStandards',
            'currencyCD',
            'industryCategory'
        ], errors='ignore')

        df = df.sort_values(
            by=['ticker', 'endDate', 'endDateRep', 'actPubtime'],
            ascending=[True, True, True, True]
        )

        return df
    
    def remove_duplicates(self,
                          df_orig):
    
        df = df_orig.sort_values(
            by=['ticker', 'endDate', 'endDateRep', 'actPubtime'],
            ascending=[True, True, True, True]
        )

        # Normalize ticker
        df['ticker'] = df['ticker'].apply(lambda x: str(x).zfill(6))

        # Drop duplicated rows by checking endDate -> endDateRep
        selected_1 = df.groupby(['ticker', 'endDate']).apply(
            lambda x: tuple(
                x[x['endDateRep'] == x['endDateRep'].values[-1]].index))
        df = df.loc[np.concatenate(selected_1.values)]
        
        # Drop duplicated rows by checking endDateRep -> actPubtime
        selected_2 = df.groupby(['ticker', 'endDate', 'endDateRep']).apply(
            lambda x: tuple(
                x[x['actPubtime'] == x['actPubtime'].values[-1]].index))
        df = df.loc[np.concatenate(selected_2.values)]
        
        # TODO: deal with absent data
        
        return df
    
    def drop_null_zero_columns(self, df, file_name, save_path):
        null_columns = []
        for col, n in df.isnull().all().iteritems():
            if n:
                null_columns.append(col)
        for col, n in (df == 0).all().iteritems():
            if n:
                null_columns.append(col)
        df = df.drop(columns=null_columns)
        with open(save_path, 'a') as f:
            f.write('=' * 70 + '\n')
            f.write(file_name + ':\n')
            f.write('-' * 20 + '\n')
            for col in null_columns:
                 f.write(col + '\n')
    
    def split_concat_df(self,
                        df_orig,
                        sheet_type):
        
        for cls in ['bank', 'insurance', 'security', 'industry']:

            file_name = sheet_type + '_' + cls
            
            df = df_orig[df_orig['from'] == cls]

            # Save sheet 2
            print(sheet_type, cls, 'sheet_2')
            df.to_csv(
                join(self.save_path, 'sheet_2/csv/' + file_name + '.csv'),
                index=False)
            df.to_excel(
                join(self.save_path, 'sheet_2/excel/' + file_name + '.xlsx'),
                index=False)
            
            # Sort columns using feature dict
            df = df[self.feature_dict[file_name][0]]
            
            # Drop null and 0 columns
            self.drop_null_zero_columns(df, file_name, join(
                self.save_path,
                'sheet_3(standard_1)/null_zero_columns.txt'))
            
            # Sort rows
            df = df.sort_values(
                by=['ticker', 'endDate', 'endDateRep', 'actPubtime'],
                ascending=[True, True, True, True]
            )
            
            # Save sheet 3
            print(sheet_type, cls, 'sheet_3(standard_1)')
            df.to_csv(
                join(
                    self.save_path,
                    'sheet_3(standard_1)/csv/' + file_name + '.csv'),
                index=False)
            df.to_excel(
                join(self.save_path,
                     'sheet_3(standard_1)/excel/' + file_name + '.xlsx'),
                index=False)
            
            # Save feature dict
            df_feature_dict = pd.DataFrame(
                columns=['COLUMN_NAME', 'CN_NAME', 'ANNOTATION'])
            for en_name, cn_name, annotation in self.feature_dict[file_name][1]:
                if en_name in df.columns:
                    df_feature_dict = df_feature_dict.append(
                        pd.DataFrame({
                                'COLUMN_NAME': [en_name],
                                'CN_NAME': [cn_name],
                                'ANNOTATION': [annotation]
                            }), ignore_index=True)
                    
            # print(df_feature_dict.shape[0], len(df.columns))
            # print(df_feature_dict['COLUMN_NAME'], df.columns)
            assert df_feature_dict.shape[0] == len(df.columns)
            df_feature_dict.to_csv(
                join(
                    self.save_path,
                    'sheet_3(standard_1)/feature_dictionaries/'
                        + file_name + '.csv'),
                index=False)
            
    def get_ts_df(self, df_total_2, sheet_type, df_ts):
            
            dict_into_date = {}
            for _, row in df_ts.iterrows():
                dict_into_date[row['ticker']] = row['intoDate']
                
            for cls in ['bank', 'security', 'insurance']:
                
                file_name = sheet_type + '_' + cls
                
                print(sheet_type , cls, 'total_sheet_3(standard_2)')
                
                # Copy time series data to new df
                df_ts_cls = df_ts[df_ts['category']==cls]
                df = df_total_2[df_total_2['ticker'].apply(
                    lambda x: x in df_ts_cls['ticker'].values)]
                
                # Drop data from total df
                df_total_2.drop(df.index)
                
                # Remove data before into_date
                for _, row in df_ts_cls.iterrows():
                    if not (type(row['intoDate']) == float and
                            np.isnan(row['intoDate'])):
                        df = df.drop(df[
                            (df['ticker'].apply(lambda x: x == row['ticker'])) &
                            (df['endDate'].apply(lambda x: x < row['intoDate']))
                        ].index)
                
                # Drop null and 0 columns
                self.drop_null_zero_columns(
                    df, file_name,
                    join(self.save_path,
                         'total_sheet_3(standard_2)/null_zero_columns.txt'))
                
                # Save time series data to sheet 3
                df.to_csv(
                    join(self.save_path, 'total_sheet_3(standard_2)/csv/')
                        + file_name + '.csv',
                    index=False)
                df.to_excel(
                    join(self.save_path, 'total_sheet_3(standard_2)/excel/')
                        + file_name + '.xlsx',
                    index=False)
            
            return df_total_2.reset_index()
    
    def get_sw_name_dict(self):
        df1 = pd.read_csv(
            join(self.classification_path, 'shenwan/shenwan_l1.csv'))
        df2 = pd.read_csv(
            join(self.classification_path, 'shenwan/shenwan_l2.csv'))
        df3 = pd.read_csv(
            join(self.classification_path, 'shenwan/shenwan_l3.csv'))
        sw_name_dict = {}
        for code, name in zip(df1['industrySymbol'], df1['industryName']):
            sw_name_dict[code] = name
        for code, name in zip(df2['industrySymbol'], df2['industryName']):
            sw_name_dict[code] = name
        for code, name in zip(df3['industrySymbol'], df3['industryName']):
            sw_name_dict[code] = name
        return sw_name_dict
    
    def get_sw_cls_dict(self, df_sw_cls, sw_name_dict):

        # Only use the 2014 shenwan classification standard
        df_sw_cls = df_sw_cls[
            ((df_sw_cls['outDate'] > '2014-01-01') |
             (df_sw_cls['outDate'].isnull())) &
            (df_sw_cls['industrySymbol'].apply(
                lambda x: x in sw_name_dict.keys()))]
        
        # Transfer shenwan class info from date range to end date
        sw_cls_dict = {}
        sw_earliest_date_dict = {}
        today_date = date.today().strftime("%Y-%m-%d")
        for _, row in df_sw_cls.iterrows():
            ticker = row['ticker']
            if ticker not in sw_earliest_date_dict:
                sw_earliest_date_dict[ticker] = today_date
            into_date = row['intoDate']
            out_date = row['outDate']
            into_year = int(into_date[:4])
            if type(out_date) == float and np.isnan(out_date):
                out_date = today_date
            out_year = int(out_date[:4])
            for year in range (into_year, out_year + 1):
                for month in ['-03-31', '-06-30', '-09-30', '-12-31']:
                    end_date = str(year) + month
                    if into_date < end_date <= out_date:
                        class_L3 = str(row['industrySymbol'])
                        class_L2 = class_L3[:4] +'00'
                        # print(ticker, class_L3, class_L2)
                        class_L2_CN = sw_name_dict[int(class_L2)]
                        class_L1 = class_L3[:2] +'0000'
                        class_L1_CN = sw_name_dict[int(class_L1)]
                        sw_cls_dict[(ticker, end_date)] = \
                            [class_L1, class_L1_CN, class_L2, class_L2_CN]
                        if end_date < sw_earliest_date_dict[ticker]:
                            sw_earliest_date_dict[ticker] = end_date
            
        # Record outlier classes in the sw_cls_outliers_dict
        df_sw_cls = df_sw_cls.sort_values(
            by=['ticker', 'intoDate'], ascending=[True, True])
        df_sw_cls = df_sw_cls.set_index('ticker')
        sw_cls_outliers_dict = {}
        for ticker, end_date in sw_earliest_date_dict.items():
            if end_date == today_date:
                class_L3 = df_sw_cls.loc[ticker, 'industrySymbol']
                if type(class_L3) != np.int64:
                    class_L3 = class_L3.values[0]
                class_L3 = str(class_L3)
                class_L2 = class_L3[:4] +'00'
                class_L2_CN = sw_name_dict[int(class_L2)]
                class_L1 = class_L3[:2] +'0000'
                class_L1_CN = sw_name_dict[int(class_L1)]
                sw_cls_outliers_dict[ticker] = \
                    [class_L1, class_L1_CN, class_L2, class_L2_CN]
        
        # Remove outliers from earliest_date_dict
        sw_earliest_date_dict = \
            {k: v for k, v in sw_earliest_date_dict.items() if v != today_date}
                    
        return sw_cls_dict, sw_earliest_date_dict, sw_cls_outliers_dict
        
    def add_classes(self,
                    sheet_type,
                    df_orig,
                    df_cls,
                    sw_cls_dict,
                    sw_name_dict,
                    sw_earliest_date_dict,
                    sw_cls_outliers_dict):
        
        # Original statements dataframes
        df_orig = df_orig.set_index(['ticker', 'endDate'])
        df_orig = df_orig.drop(columns=['index'], errors='ignore')
        
        # Get class from classification
        df_cls = df_cls.set_index(['ticker', 'endDate'])
        df_cls = df_cls.drop(columns=['index'], errors='ignore')
        
        # Outer join two dataframes to add classes columns
        df = pd.concat([df_orig, df_cls], axis=1, join='outer')
        df = df.loc[df_orig.index]
        print(df_orig.shape, df_cls.shape, df.shape)

        # Earliest
        earliest_date_dict = {}
        today_date = date.today().strftime("%Y-%m-%d")
        for (ticker, end_date), _ in df_cls.iterrows():
            if ticker not in earliest_date_dict:
                earliest_date_dict[ticker] = today_date
            if end_date < earliest_date_dict[ticker]:
                earliest_date_dict[ticker] = end_date
        earliest_date_dict = \
            {k: v for k, v in earliest_date_dict.items() if v != today_date}
                
        def _get_cls(_ticker, _end_date):
            _class_L1 = df.loc[(_ticker, _end_date), 'class_L1']
            _class_L1_CN = df.loc[(_ticker, _end_date), 'class_L1_CN']
            _class_L2 = df.loc[(_ticker, _end_date), 'class_L2']
            _class_L2_CN = df.loc[(_ticker, _end_date), 'class_L2_CN']
            return _class_L1, _class_L1_CN, _class_L2, _class_L2_CN
        
        # Fill in the rows with no class
        null_class_rows = []
        null_index = df['class_L1'].isnull()
        df_null = df[null_index].reset_index()
        for ticker, end_date in tqdm(
                zip(df_null['ticker'].values, df_null['endDate'].values),
                total=df_null['ticker'].shape[0]):
            if (ticker, end_date) not in sw_cls_dict:
                if (ticker in earliest_date_dict) and \
                        (ticker in sw_earliest_date_dict):
                    if end_date < earliest_date_dict[ticker] \
                            <= sw_earliest_date_dict[ticker]:
                        class_L1, class_L1_CN, class_L2, class_L2_CN = \
                            _get_cls(ticker, earliest_date_dict[ticker])
                    elif end_date < sw_earliest_date_dict[ticker] \
                            < earliest_date_dict[ticker]:
                        class_L1, class_L1_CN, class_L2, class_L2_CN = \
                            sw_cls_dict[(ticker, sw_earliest_date_dict[ticker])]
                    else:
                        print('Class not found 1:', (ticker, end_date))
                        null_class_rows.append((ticker, end_date))
                        continue
                elif (ticker in earliest_date_dict) and \
                        (ticker  not in sw_earliest_date_dict):
                    class_L1, class_L1_CN, class_L2, class_L2_CN = \
                            _get_cls(ticker, earliest_date_dict[ticker])
                elif (ticker not in earliest_date_dict) and \
                        (ticker  in sw_earliest_date_dict):
                    class_L1, class_L1_CN, class_L2, class_L2_CN = \
                            sw_cls_dict[(ticker, sw_earliest_date_dict[ticker])]
                else:
                    if ticker in sw_cls_outliers_dict:
                        class_L1, class_L1_CN, class_L2, class_L2_CN = \
                            sw_cls_outliers_dict[ticker]
                    else:
                        print('Class not found 2:', (ticker, end_date))
                        null_class_rows.append((ticker, end_date))
                        continue
            else:
                class_L1, class_L1_CN, class_L2, class_L2_CN = \
                    sw_cls_dict[(ticker, end_date)]
            df.loc[(ticker, end_date), 'class_L1'] = class_L1
            df.loc[(ticker, end_date), 'class_L1_CN'] = class_L1_CN
            df.loc[(ticker, end_date), 'class_L2'] = class_L2
            df.loc[(ticker, end_date), 'class_L2_CN'] = class_L2_CN
        
        with open(join(
                self.save_path,
                'total_sheet_3(standard_2)/null_class.txt'), 'a') as f:
            f.write('=' * 70 + '\n')
            f.write(sheet_type + '_general: \n'.format(null_class_rows))
            f.write('-' * 20 + '\n')
            for ticker, end_date in null_class_rows:
                f.write(ticker + ', ' + end_date + '\n')
        
        # Set all of the level 2 classes of 490000 to 490300
        for _, row in df.iterrows():
            if row['class_L1'] == 490000:
                row['class_L2'] = 490300
                row['class_L2_CN'] = sw_name_dict[490300]
        
        return df.reset_index()
    
    def process_l1(self, start_date=None, end_date=None):

        print('-' * 70)
        print('Preprocessing level 1...')
        
        orig_path = join(self.data_path, 'original/csv')
        check_dirs([
            join(self.save_path, 'sheet_1'),
            join(self.save_path, 'sheet_1/csv'),
            join(self.save_path, 'sheet_1/excel')
        ])

        # Drop new listed companies
        self.drop_list.extend(self.get_new_listed(end_date))

        for en_name in self.cn_to_en.values():
            print(en_name + '_sheet_1')

            df_l1 = self.normalize(
                join(orig_path, en_name) + '.csv', start_date, end_date)
            df_l1.to_csv(
                join(join(self.save_path, 'sheet_1/csv'), en_name) + '.csv',
                index=False)
            df_l1.to_excel(
                join(join(self.save_path, 'sheet_1/excel'), en_name) + '.xlsx',
                index=False)
            
    def process_l2(self, start_date=None, end_date=None):
        
        print('-' * 70)
        print('Preprocessing level 2...')
        
        remove_file(join(
            self.save_path, 'sheet_3(standard_1)/null_zero_columns.txt'))
        check_dirs([
            join(self.save_path, 'total_sheet_1'),
            join(self.save_path, 'total_sheet_1/csv'),
            join(self.save_path, 'total_sheet_1/excel'),
            join(self.save_path, 'sheet_2'),
            join(self.save_path, 'sheet_2/csv'),
            join(self.save_path, 'sheet_2/excel'),
            join(self.save_path, 'sheet_3(standard_1)'),
            join(self.save_path, 'sheet_3(standard_1)/csv'),
            join(self.save_path, 'sheet_3(standard_1)/excel'),
            join(self.save_path, 'sheet_3(standard_1)/feature_dictionaries'),
            join(self.save_path, 'total_sheet_2'),
            join(self.save_path, 'total_sheet_2/csv'),
            join(self.save_path, 'total_sheet_2/excel')
        ])
        
        for sheet_type in ['income', 'cash_flow', 'balance']:
            
            # Total sheet 1
            print(sheet_type + ' total_sheet_1')
            df_total_1 = self.get_concat_df(
                join(self.save_path, 'sheet_1/csv'), sheet_type)
            df_total_1.to_csv(
                join(self.save_path, 'total_sheet_1/csv/')
                    + sheet_type + '.csv',
                index=False)
            df_total_1.to_excel(
                join(self.save_path, 'total_sheet_1/excel/')
                    + sheet_type + '.xlsx',
                index=False)
            
            # Set start date and end date
            if start_date:
                df_total_1 = df_total_1[df_total_1['endDate'] >= start_date]
            if end_date:
                df_total_1 = df_total_1[df_total_1['endDate'] <= end_date]
            
            # Normalize the repeated data
            print(sheet_type + ' total_sheet_2')
            df_total_2 = self.remove_duplicates(df_total_1)
            
            # Split the total sheet 2 to get sheet 2,
            # then check null columns and get sheet sheet3
            self.split_concat_df(df_total_2, sheet_type)
            
            # Total sheet 2
            print(sheet_type + ' total_sheet_2')
            df_total_2 = df_total_2.drop(columns=['from'])
            
            # Save total sheet 2
            df_total_2.to_csv(
                join(self.save_path, 'total_sheet_2/csv/')
                    + sheet_type + '.csv',
                index=False)
            df_total_2.to_excel(
                join(self.save_path, 'total_sheet_2/excel/')
                    + sheet_type + '.xlsx',
                index=False)
        
    def process_l3(self):
        
        print('-' * 70)
        print('Preprocessing level 3...')

        remove_file(join(
            self.save_path, 'total_sheet_3(standard_2)/null_zero_columns.txt'))
        remove_file(join(
            self.save_path, 'total_sheet_3(standard_2)/null_class.txt'))
        check_dirs([
            join(self.save_path, 'total_sheet_3(standard_2)'),
            join(self.save_path, 'total_sheet_3(standard_2)/csv'),
            join(self.save_path, 'total_sheet_3(standard_2)/excel')
        ])
        
        df_ts = pd.read_csv(
            join(self.data_path, 'bank_security_insurance.csv'),
            dtype={'ticker': str})
        
        df_cls = pd.read_csv(
            join(self.classification_path, 'classification.csv'),
            dtype={'ticker': str})
        
        df_sw_cls = pd.read_csv(
            join(self.classification_path, 'shenwan_classification.csv'),
            dtype={'ticker': str})
        sw_name_dict = self.get_sw_name_dict()
        sw_cls_dict, sw_earliest_date_dict, sw_cls_outliers_dict = \
            self.get_sw_cls_dict(df_sw_cls, sw_name_dict)
        
        for sheet_type in ['income', 'cash_flow', 'balance']:
            
            # Get total data 2
            df_total_2 = pd.read_csv(
                join(self.save_path,
                     'total_sheet_2/csv/' + sheet_type + '.csv'),
                dtype={'ticker': str})
            
            # TODO: remove delisted companies
            delisted, delisted_delete = self.get_delisted_tickers()
            delete_tickers = []
            for t, f in df_total_2.groupby('ticker').apply(
                    lambda x: ('2020-12-31' not in tuple(x['endDate']))
                    and (len(tuple(x['endDate'])) > 0)).iteritems():
                if f and t not in delisted:
                    delete_tickers.append(t)
            delisted_delete = delisted_delete.union(delete_tickers)
                    
            df_total_2.drop(df_total_2[
                df_total_2['ticker'].apply(
                    lambda x: x in delisted_delete)
                ].index, inplace=True)
            
            # TODO: remove ST companies
            
            # TODO: remove pre-listed data
            
            # TODO: remove companies with revenue less than a threshold
            
            # Get and save time series data
            df_total_3 = self.get_ts_df(df_total_2, sheet_type, df_ts)

            # Drop null and 0 columns
            self.drop_null_zero_columns(
                df_total_3, sheet_type + '_' + 'general',
                join(self.save_path,
                     'total_sheet_3(standard_2)/null_zero_columns.txt'))
            
            # Add Shenwan classification
            df_total_3 = self.add_classes(
                sheet_type, df_total_3, df_cls, sw_cls_dict, sw_name_dict,
                sw_earliest_date_dict, sw_cls_outliers_dict)
            
            # Save total sheet 3
            df_total_3.to_csv(
                join(self.save_path, 'total_sheet_3(standard_2)/csv/')
                    + sheet_type + '_general.csv',
                index=False)
            df_total_3.to_excel(
                join(self.save_path, 'total_sheet_3(standard_2)/excel/')
                    + sheet_type + '_general.xlsx',
                index=False)

    def main(self):
        self.process_l1(end_date='2021-01-01')
        self.process_l2()
        self.process_l3()


if __name__ == '__main__':
    
    DataPreprocess(
        data_path='../../data/financial_statements',
        classification_path='../../data/classification',
        save_path='../../data/standard_dataset',
        info_path='../../data/stocks_information/stocks_info-20210611.csv',
        delist_path='../../data/stocks_information/delist-delete-20210611.csv',
        drop_list=[
            '000991', '002257', '002525',
            '300060', '001872', '001914', '601360']
    ).main()
    