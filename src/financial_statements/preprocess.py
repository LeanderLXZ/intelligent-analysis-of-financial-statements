import os
import re
import pandas as pd
from os.path import join

# Check if directories exit or not
def check_dirs(path_list):
    for dir_path in path_list:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


class DataPreprocess(object):

    def __init__(self, data_path, drop_list):
        self.data_path = data_path
        df_cn_en = pd.read_csv(join(self.data_path, 'cn_to_en.csv'))
        self.cn_to_en = {c: e for c, e in zip(df_cn_en['cn_name'], df_cn_en['en_name'])}
        self.drop_list = drop_list

    def normalize_l1(self, file_path, is_balance_sheet=False):

        df = pd.read_csv(file_path)

        # Nomalize ticker
        df['ticker'] = df['ticker'].apply(lambda x: str(x).zfill(6))

        # Delete Companies
        df.drop(df[df['ticker'].apply(
            lambda x: (x[0] in ['9', 'A', '2']) | (x in self.drop_list)
        )].index, inplace=True)

        # If is not balance sheet, Drop Q3 and then rename report type
        if not is_balance_sheet:
            df = df.drop(df[df['reportType']=='Q3'].index, axis=0)
        type_dict = {
            'Q1': 'Q1',
            'S1': 'Q2',
            'Q3': 'Q3',
            'CQ3': 'Q3',
            'A': 'A'
        }
        df['reportType'] = df['reportType'].apply(lambda x: type_dict[x])

        # Select mergedFlag == 1
        if 'mergedFlag' in df.columns:
            df = df[df['mergedFlag'] == 1]

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

        return df
    
    def normalize_l2(self, file_path):

        df = pd.read_csv(file_path)
        df = df.sort_values(
            by=['ticker', 'endDate', 'endDateRep', 'actPubtime'], 
            ascending=[True, True, True, True]
        )

        # Set start year to 2011
        df = df[df['endDate'] >= '2011-01-01']

        # Nomalize ticker
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
        
        return df, df_removed, df_dup_act
    
    def process_l1(self):

        orig_path = join(self.data_path, 'original')
        check_dirs([
            join(self.data_path, 'normalized_l1'),
            join(self.data_path, 'normalized_l1/csv'),
            join(self.data_path, 'normalized_l1/excel'),
            join(self.data_path, 'normalized_l1/statistics'),
            join(self.data_path, 'normalized_l1/statistics/feature_info')
        ])

        df_info = pd.DataFrame(columns=(
            'SHEET_NAME', 'NUMBER_OF_COMPANIES'
        ))

        for f in os.listdir(orig_path):
            m = re.match(r'(.*).csv', f)
            if m:
                cn_name = m.groups()[0]
                en_name = self.cn_to_en[cn_name]
                print(cn_name, en_name)

                df_l1 = self.normalize_l1(join(orig_path, cn_name) + '.csv', 'balance' in en_name)
                df_l1.to_csv(join(join(self.data_path, 'normalized_l1/csv'), en_name) + '.csv', index=False)
                df_l1.to_excel(join(join(self.data_path, 'normalized_l1/excel'), en_name) + '.xlsx', index=False)
                df_l1.count().to_excel(join(join(self.data_path, 'normalized_l1/statistics/feature_info'), en_name) + '_feature_info.xlsx', header=False)
                df_info = df_info.append(pd.DataFrame({
                    'SHEET_NAME': [en_name], 
                    'NUMBER_OF_COMPANIES': [len(set(df_l1['ticker']))]
                }), ignore_index=True)

        df_info.to_excel(join(join(self.data_path, 'normalized_l1/statistics'), 'number_of_companies.xlsx'), index=False)

    def process_l2(self):

        orig_path = join(self.data_path, 'normalized_l1/csv')
        check_dirs([
            join(self.data_path, 'normalized_l2'),
            join(self.data_path, 'normalized_l2/removed'),
            join(self.data_path, 'normalized_l2/duplicated'),
            join(self.data_path, 'normalized_l2/csv'),
            join(self.data_path, 'normalized_l2/excel'),
            join(self.data_path, 'normalized_l2/statistics'),
            join(self.data_path, 'normalized_l2/statistics/feature_info')
        ])

        df_info = pd.DataFrame(columns=(
            'SHEET_NAME', 'NUMBER_OF_COMPANIES'
        ))

        for f in sorted(os.listdir(orig_path)):
            m = re.match(r'(.*).csv', f)
            if m:
                en_name = m.groups()[0]
                print('='*70)
                print(en_name)

                df_l2, df_removed_l2, df_dup_act_l2 = self.normalize_l2(join(orig_path, en_name) + '.csv')
                df_l2.to_csv(join(join(self.data_path, 'normalized_l2/csv'), en_name) + '.csv', index=False)
                df_l2.to_excel(join(join(self.data_path, 'normalized_l2/excel'), en_name) + '.xlsx', index=False)
                if df_removed_l2.shape[0] > 0:
                    df_removed_l2.to_excel(join(join(self.data_path, 'normalized_l2/removed'), en_name) + '_removed.xlsx', index=False)
                if df_dup_act_l2 is not None:
                    df_dup_act_l2.to_excel(join(join(self.data_path, 'normalized_l2/duplicated'), en_name) + '_duplicated.xlsx', index=False)
                df_l2.count().to_excel(join(join(self.data_path, 'normalized_l2/statistics/feature_info'), en_name) + '_feature_info.xlsx', header=False)
                df_info = df_info.append(pd.DataFrame({
                    'SHEET_NAME': [en_name], 
                    'NUMBER_OF_COMPANIES': [len(set(df_l2['ticker']))]
                }), ignore_index=True)

        df_info.to_excel(join(join(self.data_path, 'normalized_l2/statistics'), 'number_of_companies.xlsx'), index=False)