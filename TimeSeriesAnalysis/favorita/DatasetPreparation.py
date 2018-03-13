#     cut = 0.+1e-12 # 0.+1e-15

'''

Created on Nov 21, 2017



@author: vladimirfux

'''

import numpy as np

import pandas as pd

import random

import matplotlib

from math import log, sqrt

matplotlib.use('TkAgg')

from calendar import monthrange

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction import DictVectorizer as DV

from sklearn.preprocessing import LabelBinarizer



from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.cluster import KMeans

import gc; gc.enable()

from sklearn import preprocessing, linear_model, metrics

from sklearn.metrics import mean_squared_error, r2_score

import time

from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor



path = '~/Documents/tensorflow/Kaggle/Favorita/'

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

#     train = data['tra']



    del data['tra']; gc.collect();

    train = train[train['unit_sales'] >= 0]

#     target = train['unit_sales'].values
# 
#     target[target < 0.] = 0. # better drop them or imput with mean
# 
#     train['unit_sales'] = np.log1p(target)    

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

    sum = 0 

    for i in range(1, len(real)):

        s1 = max(real[i], 0)

        s2 = max(pred[i], 0)

        s = pow(log(s1 + 1) - log(1 + s2), 2) * weights[i]

        sum += s 

        
    return sqrt(sum / norm)



# cluster items with Kmeans in specified number of clusters
def cluster_items(train, test, clusters=100):

    aggr = train.groupby(['item_nbr', 'weekday'], as_index=False)['unit_sales'].agg({'av_sales':'mean'})

    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(aggr)

    aggr['kcluster'] = kmeans.labels_

    cols = ['kcluster', 'av_sales']

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
    

#     aggr = train.groupby(['family', 'class', 'perishable'], as_index=False)['bin'].agg({'av_bin':'mean'})

#     test = pd.merge(test, new_items, how='left', on=['item_nbr',])



    # for unseen items - find the average bin from similar products 
 

    return train, test



if __name__ == '__main__':

    loadPrepared = True
    

    random.seed(234)
    dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', \

              'yea':'uint16', 'mon':'uint8', 'day':'uint8', 'weekday':'uint8'}
    if(not loadPrepared):
        print('Preparing data')
        train, real_test = prepare(path)
        print('Saving prepared data')
        
        train.to_csv(path + "prepared_train_2017_6_9.csv")
        real_test.to_csv(path + "prepared_test.csv")
        
        train_all = train
        real_test_all = real_test
    else:
        print('Reading files')
        train_all = pd.read_csv(path + 'prepared_train_2017_6_9.csv', dtype=dtypes, parse_dates=['date'])
        real_test_all = pd.read_csv(path + 'prepared_test.csv', dtype=dtypes, parse_dates=['date'])
#         eval_test_all = train_all[(train_all['mon'] == 8)&(train_all['yea'] == 2017)]
    
    # TODO this remove 
#     train_all = train_all[~ (train_all['mon'] == 8)&(train_all['yea'] == 2017)]
#     train_all = pd.read_csv(path+'prepared_train_2017_8.csv', dtype=dtypes, parse_dates=['date'])
# 
#     real_test_all = pd.read_csv(path+'prepared_test.csv', dtype=dtypes, parse_dates=['date'])
    
    
    
#     eval_test_all = train_all[(train_all['mon'] == 8)&(train_all['yea'] == 2017)]
    
    # TODO this remove 
#     train_all = train_all[~ (train_all['mon'] == 8)&(train_all['yea'] == 2017)]
    
    
    print('Data loaded')



    stores = np.sort(train_all['store_nbr'].unique())  # [range(0,3)]

    predictions = pd.DataFrame()

    eval_predictions = pd.DataFrame()

    eval_true = []

    weights = []

    # model to use
    model = 3

    # consider all stores separately
    for s in stores:

        print('processing store ', s, ' from ', len(stores))
        train = train_all[train_all['store_nbr'] == s]
#         eval_test = eval_test_all[eval_test_all['store_nbr']==s]
        real_test = real_test_all[real_test_all['store_nbr'] == s]

        ids = train[(train['mon'] == 8) & (train['yea'] == 2016)]['id']

        

        print('Clustering')

        train, real_test = bin_items(train, real_test, 150)
        real_test = real_test.fillna(-1)    

        print('Clustered')

        

        print('Vectorizing')

        cols_to_vector = ['weekday', 'mon', 'yea']

        train = pd.get_dummies(train, columns=cols_to_vector)

        real_test = pd.get_dummies(real_test, columns=cols_to_vector)

        missing_cols = set(train.columns) - set(real_test.columns)

        for c in missing_cols:

            real_test[c] = 0

            

        missing_cols = set(real_test.columns) - set(train.columns)

        for c in missing_cols:

            train[c] = 0

        print('Vectorizing is done')

       

       

        print('splitting in x and y')

        columns_to_excl = ['unit_sales', 'date', 'id', 'item_nbr', 'store_nbr']   
        
        # split into train and evaluation
        
        eval_test = train[train['id'].isin(ids)]
        train = train[~train['id'].isin(ids)]


        trainY = train['unit_sales']
        trainX = train[[i for i in list(train.columns) if i not in columns_to_excl]]
        real_testX = real_test[[i for i in list(real_test.columns) if i not in columns_to_excl]]
        eval_testY = eval_test['unit_sales']
        eval_testX = eval_test[[i for i in list(eval_test.columns) if i not in columns_to_excl]]



#         weights = real_testX['perishable']
        
        if model == 1:

            name = 'btr'

            print('Fitting ' + name + ' regression model')

            regr = BaggingRegressor(DecisionTreeRegressor(max_depth=10, max_features=0.6), n_jobs=2, verbose=True)      

        elif model == 2:

            name = 'AdaBoostRegressor'

            print('Fitting ' + name + ' regression model')

            regr = AdaBoostRegressor()     

        elif model == 3:

            name = 'GradientBoostingRegressor'

            print('Fitting ' + name + ' regression model')

            regr = GradientBoostingRegressor(n_estimators=100, verbose=1)  

        elif model == 4:

            name = 'RandomForestRegressor'

            print('Fitting ' + name + ' regression model')

            regr = RandomForestRegressor(verbose=1, n_jobs=2)

        

        regr.fit(trainX, trainY)

        

        print('Predict eval_test')

        if len(eval_testX) != 0:

            pred_y = regr.predict(eval_testX)

            weights.extend(eval_testX['perishable'])

            cut = 0. + 1e-12  # 0.+1e-15

            ids = eval_test['id']

#             pred_y= np.clip(np.exp(pred_y) - 1, a_min=cut,a_max=None)

    

            store_prediction = pd.DataFrame({'id':ids, 'unit_sales':pred_y})

            eval_predictions = eval_predictions.append(store_prediction)

            eval_true.extend(eval_testY)

        

        print('Predict real test')

        if len(real_testX) != 0:

            pred_y = regr.predict(real_testX)
            #     pred_y = np.exp(pred_y)

            ids = real_test['id']
#             real_testX.to_csv(path+'Predictions/t.csv')
#             pred_y.to_csv(path+'Predictions/t2.csv')
#             pred_y= np.clip(np.exp(pred_y) - 1, a_min=cut,a_max=None)

        

            store_prediction = pd.DataFrame({'id':ids, 'unit_sales':pred_y})

            predictions = predictions.append(store_prediction)



    

    eval_predictions = eval_predictions.fillna(0)    
    score = 0
    score = nwrmsle(weights, eval_true, eval_predictions['unit_sales'].tolist())
#     eval_predictions.to_csv(path+'Predictions/eval_s' \
# 
#                                                + str(np.around(score,2))+'.csv', index=False)
    print("nwrmsle score for regression on eval set is: ", score)    

    
#     real_test_all = real_test_all.drop('unit_sales', 1)
    real_test_all = pd.merge(real_test_all, predictions, how='left', on=['id'])
    print(real_test_all.isnull().sum())

    # #     to_save.to_csv(path+'Predictions/submission_lr_s' + str(np.around(score,2))+'_' + str(time.time()) +'.csv', index=False)

    real_test_all = real_test_all.fillna(0)    
    real_test_all['unit_sales'] = np.abs(real_test_all['unit_sales'])


    real_test_all[['id', 'unit_sales']].to_csv(path + 'Predictions/submission_lr_s' \

                                               + str(np.around(score, 2)) + '_' + str(time.time()) + '.csv', index=False)


    print('Prediction results saved')
        
