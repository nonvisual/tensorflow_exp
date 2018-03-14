'''
Created on Mar 13, 2018

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
import favorita.DatasetPreparation as dp

path = '~/Documents/tensorflow/Kaggle/Favorita/'


if __name__ == '__main__':

    loadPrepared = True
    

    random.seed(234)
    dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', \

              'yea':'uint16', 'mon':'uint8', 'day':'uint8', 'weekday':'uint8'}
    if(not loadPrepared):
        print('Preparing data')
        train, real_test = dp.prepare(path)
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

        train, real_test = dp.bin_items(train, real_test, 150)
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
    score = dp.nwrmsle(weights, eval_true, eval_predictions['unit_sales'].tolist())
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
        