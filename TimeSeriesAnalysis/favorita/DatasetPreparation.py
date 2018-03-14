#     cut = 0.+1e-12 # 0.+1e-15

'''

Created on Nov 21, 2017



@author: vladimirfux

'''

import numpy as np

import pandas as pd

import matplotlib

from math import log, sqrt

matplotlib.use('TkAgg')

from sklearn.cluster import KMeans

import gc; gc.enable()

from sklearn import preprocessing


import time

# decode date, add payday field
def df_transform(df):

    df['date'] = pd.to_datetime(df['date'])
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.dayofweek

    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})
    
    # distance to the payday
    df['payday'] = np.minimum(31 - df['day'], np.abs(15 - df['day']))
    df['payday'] = np.minimum(df['day'] - 1, df['payday'])
    gc.collect()
    df.fillna(-1, inplace=True)

    return df



# encode factor as numerical (better use embedding)
def df_lbl_enc(df):

    for c in df.columns:

        if df[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            df[c] = lbl.fit_transform(df[c])

            print(c)

    return df





# prepare the dataset
def prepare(path, remove_days_after_earthquake=0):

    # track time
    start_time = time.time()

    

    # read datasets
    print('Read dataset')


    dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}
    data = {

        'tra': pd.read_csv(path + 'train.csv', dtype=dtypes, parse_dates=['date']),
        'tes': pd.read_csv(path + 'test.csv', dtype=dtypes, parse_dates=['date']),
        'ite': pd.read_csv(path + 'items.csv'),
        'sto': pd.read_csv(path + 'stores.csv'),

#         'trn': pd.read_csv(path+'transactions.csv', parse_dates=['date']),

        'hol': pd.read_csv(path + 'holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),

    }

    print('Process dataset')

    

    # select one month only
    train = data['tra'][(data['tra']['date'].dt.year > 2015) & (data['tra']['date'].dt.month >= 6) & (data['tra']['date'].dt.month <= 9)\
#                         & (data['tra']['store_nbr']==1)\
                        ]

    del data['tra']; gc.collect();

    train = train[train['unit_sales'] >= 0]

    # join data
    print('---Join items')
    data['ite'] = df_lbl_enc(data['ite'])

    train = pd.merge(train, data['ite'], how='left', on=['item_nbr'])

    test = pd.merge(data['tes'], data['ite'], how='left', on=['item_nbr'])

    del data['tes']; gc.collect();

    del data['ite']; gc.collect();

    

#     train = pd.merge(train, data['trn'], how='left', on=['date','store_nbr'])
# 
#     test = pd.merge(test, data['trn'], how='left', on=['date','store_nbr'])
# 
#     del data['trn']; gc.collect();
# 
#     target = train['transactions'].values
# 
#     target[target < 0.] = 0.
# 
#     train['transactions'] = np.log1p(target) 

    print('---Join stores')


    data['sto'] = df_lbl_enc(data['sto'])

    train = pd.merge(train, data['sto'], how='left', on=['store_nbr'])

    test = pd.merge(test, data['sto'], how='left', on=['store_nbr'])

    del data['sto']; gc.collect();

    
    print('---Join holidays')

    data['hol'] = data['hol'][['date', 'transferred']]

    data['hol']['transferred'] = data['hol']['transferred'].map({'False': 0, 'True': 1})

    train = pd.merge(train, data['hol'] , how='left', on=['date'])

    test = pd.merge(test, data['hol'] , how='left', on=['date'])

    del data['hol']; gc.collect();


    print('transforming dates')
    train = df_transform(train)
    gc.collect();

    test = df_transform(test)
    gc.collect();

    

#     col = [c for c in train if c not in ['id', 'unit_sales','perishable','transactions']]

    print('Dataset prepared, elapsed time ', (time.time() - start_time))
  

    return train, test

    

    
# normalize column with idx indices 
def normalize(data, idx):

    mean = []

    sds = []

    for i in idx:        

        m = np.mean(data.loc[:, i])

        sd = np.std(data.loc[:, i])

        data.loc[:, i] = (data.loc[:, i] - m) / sd

        mean.append(m)

        sds.append(sd)

    return mean, sds



# split in train and test by date using the ratio
def split_by_date(df, ratio=0.95):

    min_date = np.min(df['date'])

    max_date = np.max(df['date'])

    period = max_date - min_date

    split = min_date + period * ratio

    train = df[df['date'] <= split]

    test = df[df['date'] > split]

    return train, test


# split in test and train by specific month (which is taken for test)
def split_by_month_in_test(df, year=2016, month=8):

    idx = (df['date'].dt.month == month) & (df['date'].dt.year == year)

    train = df[~idx]

    test = df[idx]

    return train, test



# Normalized Weighted Root Mean Squared Logarithmic Error 
def nwrmsle(weights, real, pred):

    norm = np.sum(weights)

    err_sum = 0 

    for i in range(1, len(real)):

        s1 = max(real[i], 0)

        s2 = max(pred[i], 0)

        s = pow(log(s1 + 1) - log(1 + s2), 2) * weights[i]

        err_sum += s 

        
    return sqrt(err_sum / norm)



# cluster items with Kmeans in specified number of clusters
def cluster_items(train, test, clusters=100):

    aggr = train.groupby(['item_nbr', 'weekday'], as_index=False)['unit_sales'].agg({'av_sales':'mean'})

    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(aggr)

    aggr['kcluster'] = kmeans.labels_

    train = pd.merge(train, aggr, how='left', on=['item_nbr', 'weekday'])

    test = pd.merge(test, aggr, how='left', on=['item_nbr', 'weekday'])

    return train, test


# make equal binning of items by sales unit
def bin_items(train, test, clusters=100):

    aggr = train.groupby(['item_nbr'], as_index=False)['unit_sales'].agg({'av_sales':'mean'})

    bins = pd.qcut(aggr['av_sales'], clusters, labels=False, duplicates='drop')
    aggr['bin'] = bins
    train = pd.merge(train, aggr, how='left', on=['item_nbr'])
 
    new_items = set(test['item_nbr'].unique()) - set(train['item_nbr'].unique())
 
    new_items = pd.DataFrame(list(new_items), columns=['item_nbr'])
 
    new_items = new_items.merge(test[~test['item_nbr'].duplicated()], how='left', on=['item_nbr', ])
 
    # compute average bin value for the following grouping
    aggr2 = train.groupby(['family', 'class', 'perishable'], as_index=False)['bin'].agg({'av_bin':'mean'})
    new_items = pd.merge(new_items, aggr2, how='left', on=['family', 'class', 'perishable'])
    new_items['bin'] = new_items.round({'av_bin':0})['av_bin']
 
    all_items = aggr[['item_nbr', 'bin']].append(new_items[['item_nbr', 'bin']])

#     del aggr,aggr2,new_items,bins; gc.collect();
#     all_items = pd.concat([aggr[['item_nbr','bin']], new_items[['item_nbr','bin']]], axis = 0)
#     test = pd.merge(test, all_items, how='left', on=['item_nbr'])
    test = pd.merge(test, aggr, how='left', on=['item_nbr'])

    del all_items; gc.collect();

#     test['bin']

    
    test['bin'] = test['bin'].fillna(clusters / 2)
    return train, test



