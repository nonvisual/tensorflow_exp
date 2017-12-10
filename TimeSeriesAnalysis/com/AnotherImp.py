'''
Created on Nov 12, 2017

@author: ubuntu
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import random
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn




def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test

# random.seed(111)
# rng = pd.date_range(start='2000', periods=209, freq='M')
# ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
# ts.plot(c='b', title='Exaplte Time Series')
# plt.show()
# ts.head(10)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
rawdata = pd.read_csv("~/Downloads/energydata_original.csv", 
               date_parser=dateparse)

ts =rawdata.loc[:,"Appliances"]

t =series_to_supervised(ts, 1, 3)
TS = np.array(ts)
num_periods = 50
f_horizon = 1  #forecast horizon, one period into the future

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[f_horizon:(len(TS)-(len(TS) % num_periods))+f_horizon]

y_batches = y_data.reshape(-1, num_periods, 1)
print (len(x_batches))
print (x_batches.shape)
print (x_batches[0:2])

print (y_batches[0:1])
print (y_batches.shape)


def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = TS[-(num_periods):].reshape(-1, num_periods, 1)
    return testX,testY

def plot(loss_list, predictions_series, batchX, batchY):

    plt.subplot(2, 3, 1)

    plt.cla()

    plt.plot(loss_list)



    for batch_series_idx in range(5):

        one_hot_output_series = np.array(predictions_series)[ batch_series_idx, :]

        single_output_series = np.array([out for out in one_hot_output_series])



        plt.subplot(2, 3, batch_series_idx + 2)

        plt.cla()

        plt.axis([0, num_periods, 0, 200])

        left_offset = range(num_periods)

        plt.plot(left_offset, batchX[batch_series_idx, :],  color="blue")

        plt.plot(left_offset, batchY[batch_series_idx, :] , color="red")

        plt.plot(left_offset, single_output_series , color="green")



    plt.draw()

    plt.pause(0.0001)
    
    
X_test, Y_test = test_data(TS,f_horizon,num_periods )
print (X_test.shape)
print (X_test)

tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs

inputs = 1            #number of vectors submitted
hidden = 150          #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors

X = tf.placeholder(tf.float32, [None, num_periods, inputs])   #create variable objects
y = tf.placeholder(tf.float32, [None, num_periods, output])


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static

learning_rate = 0.001   #small learning rate so we don't overshoot the minimum

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()         



epochs = 100   #number of iterations or training cycles, includes both the FeedFoward and Backpropogation

with tf.Session() as sess:
    init.run()
    loss_list= []
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})

        if ep % 10 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})

            print(ep, "\tMSE:", mse)
            loss_list.append(mse)
            y_pred = sess.run(outputs, feed_dict={X: x_batches})

            plot(loss_list, y_pred, x_batches, y_batches)

    y_pred = sess.run(outputs, feed_dict={X: X_test})
    
    print(y_pred)

plt.close()
plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), label="Actual")
#plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
plt.plot(pd.Series(np.ravel(y_pred)), label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")

plt.show()


























