import os
import numpy as np
import pandas as pd
from os.path import join

# Check if directories exit or not
def check_dirs(path_list):
    for dir_path in path_list:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


class DataPreprocess(object):

    def __init__(self, data_path, save_path, drop_list=[]):
        
        self.data_path = data_path
        self.drop_list = drop_list
        self.save_path = save_path
        
        df_cn_en = pd.read_csv(join(self.data_path, 'cn_to_en.csv'))
        self.cn_to_en = {c: e for c, e in zip(df_cn_en['cn_name'], df_cn_en['en_name'])}

    def get_delist(self, file_path):

        df_info = pd.read_csv(file_path)

        # Copy halt dataframe
        df_delist = df_info.copy()

        # Get delist stocks
        # df_delist = df_delist[df_delist['listStatusCD'] == 'DE']
        df_delist = df_delist[(df_delist['listStatusCD'] == 'DE') & (df_delist['ticker'].apply(lambda x: len(x) == 6))]

        # Drop columns
        df_delist = df_delist[['ticker', 'secShortName', 'delistDate']]

        # Normalize ticker
        df_delist['ticker'] = df_delist['ticker'].apply(lambda x: str(x).zfill(6))

        # Delete duplicates (with different halt dates)
        df_delist.drop_duplicates(inplace=True)

        return set(df_delist['ticker'])
        

    def normalize_l1(self, file_path, start_date=None, end_date=None):

        df = pd.read_csv(file_path)

        # Select mergedFlag == 1
        if 'mergedFlag' in df.columns:
            df = df[df['mergedFlag'] == 1]
        
        # Set start date and end date
        if start_date:
            df = df[df['endDate'] >= start_date]
        if end_date:
            df = df[df['endDate'] <= end_date]
        
        # Nomalize ticker
        df['ticker'] = df['ticker'].apply(lambda x: str(x).zfill(6))

        # Delete Companies
        df.drop(df[df['ticker'].apply(
            lambda x: (x[0] in ['9', 'A', '2']) | (x in self.drop_list)
        )].index, inplace=True)

        # Check fiscalPeriod = 3
        # df = df.drop(df[(df['reportType']=='Q3') & (df['fiscalPeriod']==3)].index, axis=0, errors='ignore')

        # If is not balance sheet, Drop Q3 and then rename report type
        if 'CQ3' in set(df['reportType']):
            df = df.drop(df[df['reportType']=='Q3'].index, axis=0, errors='ignore')
        
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
    
    def normalize_l2(self, file_path, start_date=None, end_date=None):

        df = pd.read_csv(file_path)
        df = df.sort_values(
            by=['ticker', 'endDate', 'endDateRep', 'actPubtime'], 
            ascending=[True, True, True, True]
        )

        # Set start date and end date
        if start_date:
            df = df[df['endDate'] >= start_date]
        if end_date:
            df = df[df['endDate'] <= end_date]

        # Normalize ticker
        df['ticker'] = df['ticker'].apply(lambda x: str(x).zfill(6))
        df_orig = df.copy()

        # Drop duplicated rows by checking endDate -> endDateRep
        selected_1 = df.groupby(['ticker', 'endDate']).apply(
            lambda x: tuple(x[x['endDateRep'] == x['endDateRep'].values[-1]].index))
        df = df.loc[np.concatenate(selected_1.values)]

        # Print out duplicated endDateRep
        idx_dup_1 = []
        for i in selected_1.values:
            if len(i) > 1:
                idx_dup_1.extend(i)
        # if idx_dup_1:
        #     print('Duplicated endDateRep: ')
        #     print(df.loc[idx_dup_1][['ticker', 'secShortName', 'endDate', 'endDateRep', 'actPubtime']])

        # Drop duplicated rows by checking endDateRep -> actPubtime
        selected_2 = df.groupby(['ticker', 'endDate', 'endDateRep']).apply(lambda x: tuple(x[x['actPubtime'] == x['actPubtime'].values[-1]].index))
        df = df.loc[np.concatenate(selected_2.values)]

        # removed data
        removed = np.setdiff1d(df_orig.index.values, np.concatenate(selected_2.values))
        df_removed = df_orig.loc[removed]

        # Print out duplicated actPubtime
        idx_dup_2 = []
        for i in selected_2.values:
            if len(i) > 1:
                idx_dup_2.extend(i)
        if idx_dup_2:
            df_dup_act = df.loc[idx_dup_2]
            # print('Duplicated actPubtime: ')
            # print(df_dup_act[['ticker', 'secShortName', 'endDate', 'endDateRep', 'actPubtime']])
        else:
            df_dup_act = None

        # # Change column names
        # df = df.rename(columns={
        #     'aop': 'AOP',
        #     'aor': 'AOR',
        #     'cogs': 'COGS',
        #     'bizTaSurchg': 'bizTaxSurchg',
        #     'atoc': 'ATOC'
        # }, errors='ignore')
        
        return df, df_removed, df_dup_act
    
    def process_l1(self, start_date=None, end_date=None):

        orig_path = join(self.data_path, 'original/csv')
        check_dirs([
            join(self.save_path, 'normalized_l1'),
            join(self.save_path, 'normalized_l1/csv'),
            join(self.save_path, 'normalized_l1/excel'),
            join(self.save_path, 'normalized_l1/statistics'),
            join(self.save_path, 'normalized_l1/statistics/feature_info')
        ])

        df_info = pd.DataFrame(columns=(
            'SHEET_NAME', 'NUMBER_OF_COMPANIES'
        ))

        for en_name in self.cn_to_en.values():
            print(en_name)

            df_l1 = self.normalize_l1(join(orig_path, en_name) + '.csv', start_date, end_date)
            df_l1.to_csv(join(join(self.save_path, 'normalized_l1/csv'), en_name) + '.csv', index=False)
            df_l1.to_excel(join(join(self.save_path, 'normalized_l1/excel'), en_name) + '.xlsx', index=False)
            df_l1.count().to_excel(join(join(self.save_path, 'normalized_l1/statistics/feature_info'), en_name) + '_feature_info.xlsx', header=False)
            df_info = df_info.append(pd.DataFrame({
                'SHEET_NAME': [en_name], 
                'NUMBER_OF_COMPANIES': [len(set(df_l1['ticker']))]
            }), ignore_index=True)

        df_info.to_excel(join(join(self.save_path, 'normalized_l1/statistics'), 'number_of_companies.xlsx'), index=False)

    def process_l2(self, start_date=None, end_date=None):

        orig_path = join(self.data_path, 'normalized_l1/csv')
        check_dirs([
            join(self.save_path, 'normalized_l2'),
            join(self.save_path, 'normalized_l2/csv'),
            join(self.save_path, 'normalized_l2/excel'),
            join(self.save_path, 'normalized_l2/statistics'),
            join(self.save_path, 'normalized_l2/statistics/feature_info')
        ])

        df_info = pd.DataFrame(columns=(
            'SHEET_NAME', 'NUMBER_OF_COMPANIES'
        ))

        for en_name in self.cn_to_en.values():
            print('-'*70)
            print(en_name)

            df_l2, df_removed_l2, df_dup_act_l2 = self.normalize_l2(join(orig_path, en_name) + '.csv', start_date, end_date)
            df_l2.to_csv(join(join(self.save_path, 'normalized_l2/csv'), en_name) + '.csv', index=False)
            df_l2.to_excel(join(join(self.save_path, 'normalized_l2/excel'), en_name) + '.xlsx', index=False)
            if df_removed_l2.shape[0] > 0:
                check_dirs([
                    join(self.save_path, 'normalized_l2/removed')
                ])
                df_removed_l2.to_excel(join(join(self.save_path, 'normalized_l2/removed'), en_name) + '_removed.xlsx', index=False)
            if df_dup_act_l2 is not None:
                check_dirs([
                    join(self.save_path, 'normalized_l2/duplicates')
                ])
                df_dup_act_l2.to_excel(join(join(self.save_path, 'normalized_l2/duplicates'), en_name) + '_duplicates.xlsx', index=False)
            df_l2.count().to_excel(join(join(self.save_path, 'normalized_l2/statistics/feature_info'), en_name) + '_feature_info.xlsx', header=False)
            df_info = df_info.append(pd.DataFrame({
                'SHEET_NAME': [en_name], 
                'NUMBER_OF_COMPANIES': [len(set(df_l2['ticker']))]
            }), ignore_index=True)

        df_info.to_excel(join(join(self.save_path, 'normalized_l2/statistics'), 'number_of_companies.xlsx'), index=False)

    def find_zeros(self, file_path):

        save_path = join(file_path, 'zeros')
        check_dirs([save_path])

        for en_name in self.cn_to_en.values():
            print('File:', en_name)
            df = pd.read_csv(join(join(file_path, 'csv'), en_name) + '.csv')
            df_zero = []
            for _, row in df.iterrows():
                for k, v in row.items():
                    if v == 0:
                        df_zero.append(row)
            if df_zero:
                df_zero = pd.concat(df_zero, axis=1).T
                df_zero.to_excel(join(save_path, en_name) + '.xlsx', index=False)
    
    def check_Q3_fiscal_period(self, file_path, end_date_rep=False):
        save_path = join(file_path, 'Q3-3-not-9_endDate_endDateRep')
        check_dirs([save_path])

        for en_name in self.cn_to_en.values():
            print('File:', en_name)
            df = pd.read_csv(join(join(self.data_path, 'original/csv'), en_name) + '.csv')

            if end_date_rep:
                df_Q3_3 = df[(df['reportType']=='Q3') | (df['reportType']=='CQ3')].groupby(
                    ['ticker', 'endDate', 'endDateRep']).apply(
                        lambda x: 9 not in set(x['fiscalPeriod']))
                df_Q3_3_not_9 = []
                for t, ed, edr in df_Q3_3[df_Q3_3.values].index.to_list():
                    df_Q3_3_not_9.append(
                        df[(df['ticker']==t) \
                        & (df['endDate']==ed) \
                        & (df['endDateRep']==edr) \
                        & (df['mergedFlag']==1)])
            else:
                df_Q3_3 = df[(df['reportType']=='Q3') | (df['reportType']=='CQ3')].groupby(
                    ['ticker', 'endDate']).apply(
                        lambda x: 9 not in set(x['fiscalPeriod']))
                df_Q3_3_not_9 = []
                for t, ed in df_Q3_3[df_Q3_3.values].index.to_list():
                    df_Q3_3_not_9.append(
                        df[(df['ticker']==t) \
                        & (df['endDate']==ed) \
                        & (df['mergedFlag']==1)])
            if df_Q3_3_not_9:
                df_Q3_3_not_9 = pd.concat(df_Q3_3_not_9)
                df_Q3_3_not_9.to_excel(join(save_path, en_name) + '.xlsx', index=False)
                
    def get_year_revenue_sheets(self, df_income, file_path, file_name):
        
        df_income['ticker'] = df_income['ticker'].apply(lambda x: str(x).zfill(6))
        df_income = df_income[df_income['endDate'].apply(lambda x: str(x)[-6:]=='-12-31')]
        df_income.drop_duplicates(subset=['ticker', 'endDate'], keep='last', inplace=True)
        df_income['endDate'] = df_income['endDate'].apply(lambda x: int(x[:4]))

        df_income.set_index(['endDate', 'ticker'], inplace=True)
        df_income.index.names = [None, 'TICKER']

        df_income_r = df_income[['revenue']]
        df_income_r = df_income_r.unstack(1).T
        df_income_r.reset_index(inplace=True)
        df_income_r = df_income_r.drop(columns=['level_0'])
        df_income_r.set_index('TICKER', inplace=True)

        df_income_r_count_list = pd.DataFrame(columns=('TICKER', 'COUNTS', 'YEARS'))
        df_income_r_not_ct_list = pd.DataFrame(columns=('TICKER', 'YEARS'))
        last_year = int(sorted(df_income_r.columns)[-1])

        for ticker, row in df_income_r.iterrows():
            row = row.notnull()
            counts = sum(row)
            years_list = []
            for k in row.keys():
                if row[k]:
                    years_list.append(k)
            df_income_r_count_list = df_income_r_count_list.append(
                pd.DataFrame({'TICKER': [ticker], 'COUNTS': [counts], 'YEARS': [years_list]}), ignore_index=True)

            if not years_list:
                print('No data:', ticker)
                df_income_r_not_ct_list = df_income_r_not_ct_list.append(
                    pd.DataFrame({'TICKER': [ticker], 'YEARS': [[]]}), ignore_index=True)
            # elif years_list[-1] - years_list[0] != counts - 1:
            elif last_year - years_list[0] != counts - 1:
                df_income_r_not_ct_list = df_income_r_not_ct_list.append(
                    pd.DataFrame({'TICKER': [ticker], 'YEARS': [years_list]}), ignore_index=True)

        save_path = join(file_path, 'revenue')
        check_dirs([save_path])
        df_income_r.to_excel(join(save_path, file_name + '.xlsx'), index=True)
        df_income_r_count_list.to_excel(join(save_path, file_name + '_counts.xlsx'), index=False)
        df_income_r_not_ct_list.to_excel(join(save_path, file_name + '_discontinuous.xlsx'), index=False)

    def get_quarter_revenue_sheets(self, df_income, file_path, file_name):
        
        df_income['ticker'] = df_income['ticker'].apply(lambda x: str(x).zfill(6))
        df_income.drop_duplicates(subset=['ticker', 'endDate'], keep='last', inplace=True)
        df_income['endDate'] = df_income['endDate'].apply(lambda x: x[:7])

        df_income.set_index(['endDate', 'ticker'], inplace=True)
        df_income.index.names = [None, 'TICKER']

        df_income_r = df_income[['revenue']]
        df_income_r = df_income_r.unstack(1).T
        df_income_r.reset_index(inplace=True)
        df_income_r = df_income_r.drop(columns=['level_0'])
        df_income_r.set_index('TICKER', inplace=True)

        df_income_r_count_list = pd.DataFrame(columns=('TICKER', 'COUNTS', 'QUARTERS'))
        df_income_r_not_ct_list = pd.DataFrame(columns=('TICKER', 'QUARTERS'))
        quarters_all = sorted(df_income_r.columns)
        last_quarter = quarters_all[-1]

        for ticker, row in df_income_r.iterrows():
            row = row.notnull()
            counts = sum(row)
            quarters_list = []
            for k in row.keys():
                if row[k]:
                    quarters_list.append(k)
            df_income_r_count_list = df_income_r_count_list.append(
                pd.DataFrame({'TICKER': [ticker], 'COUNTS': [counts], 'QUARTERS': [quarters_list]}), ignore_index=True)

            if not quarters_list:
                print('No data:', ticker)
                df_income_r_not_ct_list = df_income_r_not_ct_list.append(
                    pd.DataFrame({'TICKER': [ticker], 'QUARTERS': [[]]}), ignore_index=True)
            # elif quarters_list[-1] - quarters_list[0] != counts - 1:
            elif len(quarters_all) - quarters_all.index(quarters_list[0]) != counts:
                df_income_r_not_ct_list = df_income_r_not_ct_list.append(
                    pd.DataFrame({'TICKER': [ticker], 'QUARTERS': [quarters_list]}), ignore_index=True)

        save_path = join(file_path, 'revenue')
        check_dirs([save_path])
        df_income_r.to_excel(join(save_path, file_name + '.xlsx'), index=True)
        df_income_r_count_list.to_excel(join(save_path, file_name + '_counts.xlsx'), index=False)
        df_income_r_not_ct_list.to_excel(join(save_path, file_name + '_discontinuous.xlsx'), index=False)
    
    def get_concat_df(self, file_path, name, post_fix=''):
        df_1 = pd.read_csv(join(join(file_path, 'csv'), name + '_bank' + post_fix + '.csv'))[['ticker', 'endDate']]
        df_2 = pd.read_csv(join(join(file_path, 'csv'), name + '_insurance' + post_fix + '.csv'))[['ticker', 'endDate']]
        df_3 = pd.read_csv(join(join(file_path, 'csv'), name + '_security' + post_fix + '.csv'))[['ticker', 'endDate']]
        df_4 = pd.read_csv(join(join(file_path, 'csv'), name + '_industry' + post_fix + '.csv'))[['ticker', 'endDate']]
        df = pd.concat([df_1, df_2, df_3, df_4])
        df['ticker'] = df['ticker'].apply(lambda x: str(x).zfill(6))

        df.drop(df[df['ticker'].apply(lambda x: x in new_listed)].index, inplace=True)

        df = df.set_index(['ticker', 'endDate'])
        return set(df.index.to_list())
    
    def run(self):
        l2_path = join(self.data_path, 'normalized_l2')
        
        # Get delist companies
        delist = self.get_delist()
        self.drop_list = self.drop_list.extend(delist)
        
        # Normalization - level 1
        self.process_l1(end_date='2020-10-01')
        
        # Normalization - level 2
        self.process_l2(start_date='2011-01-01')

        # Find Zeros
        self.find_zeros(l2_path)
        
        # Check Q3
        self.check_Q3_fiscal_period(self.data_path)
        
        # Get revenue
        df_1 = pd.read_csv(join(join(l2_path, 'csv'), 'income_bank.csv'))[['ticker', 'endDate', 'revenue']]
        df_2 = pd.read_csv(join(join(l2_path, 'csv'), 'income_insurance.csv'))[['ticker', 'endDate', 'revenue']]
        df_3 = pd.read_csv(join(join(l2_path, 'csv'), 'income_security.csv'))[['ticker', 'endDate', 'revenue']]
        df_4 = pd.read_csv(join(join(l2_path, 'csv'), 'income_industry.csv'))[['ticker', 'endDate', 'revenue']]
        df_income = pd.concat([df_1, df_2, df_3, df_4])
        self.get_year_revenue_sheets(df_income, l2_path, 'revenue_year')
        self.get_quarter_revenue_sheets(df_income, l2_path, 'revenue_quarter')
        
        