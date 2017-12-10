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







# random.seed(111)

# rng = pd.date_range(start='2000', periods=209, freq='M')

# ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()

# ts.plot(c='b', title='Exaplte Time Series')

# plt.show()

# ts.head(10)



def create_dataset_from_array(array, input_size, output_size):

    dataX, dataY = [],[]

    for i in range(len(array)- input_size-output_size-1):

        a= array[i:(i+input_size)]

        b = array[(i+input_size):(i+input_size+output_size)]

        dataX.append(a)

        dataY.append(b)

    return np.array(dataX),np.array(dataY)

        

# ---------------------read data

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')

# data_for_net = pd.read_csv("~/Documents/DataAnalysisClient3/cp_dates_prepared21.csv", 

#                    date_parser=dateparse)

data_for_net = pd.read_csv("~/Documents/Tensorflow/energydata_complete.csv", 

                   date_parser=dateparse)

#



#TODO: for our data it is necessary to create batches for each cpid, cid separately!



ts =data_for_net.loc[:,"Appliances"]

TS = np.array(ts)





#----------------------params

truncated_backprop_length = 20

f_horizon = 5  #forecast horizon, one period into the future

inputs =  15            #number of vectors submitted

hidden = 100          #number of neurons we will recursively work through, can be changed to improve accuracy

output = f_horizon           #number of output vectors





x_data,y_data = create_dataset_from_array(ts, inputs, output)



x_data =  x_data[:(len(x_data)-(len(x_data) % truncated_backprop_length)),:]

# x_batches = x_data.reshape(-1, truncated_backprop_length, 1)

x_batches = np.reshape(x_data, (-1, truncated_backprop_length, x_data.shape[1]))



y_data = y_data[:(len(y_data)-(len(y_data) % truncated_backprop_length)),:]

y_batches = np.reshape(y_data, (-1, truncated_backprop_length, y_data.shape[1]))







X_test, Y_test = create_dataset_from_array(ts[-50:], inputs, output)





X_test =  X_test[:truncated_backprop_length,:]

# x_batches = x_data.reshape(-1, truncated_backprop_length, 1)

X_test = np.reshape(X_test, (1, truncated_backprop_length, X_test.shape[1]))



Y_test = Y_test[:truncated_backprop_length,:]

Y_test = np.reshape(Y_test, (1, truncated_backprop_length, Y_test.shape[1]))







print ("Number of batches:", len(x_batches))

print ("Batch input shape", x_batches.shape)

# print (x_batches[0:2])



# print (y_batches[0:1])

print ("Batch output shape",y_batches.shape)





def test_data(series,forecast,truncated_backprop_length):

    test_x_setup = TS[-(truncated_backprop_length + forecast):]

    testX = test_x_setup[:truncated_backprop_length].reshape(-1, truncated_backprop_length, 1)

    testY = TS[-(truncated_backprop_length):].reshape(-1, truncated_backprop_length, 1)

    return testX,testY



def plot(loss_list, predictions_series, batchX, batchY, input, output):



    plt.subplot(2, 3, 1)



    plt.cla()



    plt.plot(loss_list)



    ran = random.sample(range(len(predictions_series)),5)

    i = 0

    for batch_series_idx in ran:



        one_hot_output_series = np.array(predictions_series)[ batch_series_idx, -output,:]

            

        single_output_series = np.array([out for out in one_hot_output_series])

        for j in range(0,output):

            single_output_series= np.insert(single_output_series,0,None)

#         single_output_series = np.roll



        i+=1



        plt.subplot(2, 3, (i+1 ))



        plt.cla()



        plt.axis([0, truncated_backprop_length, 0, np.max(batchY[batch_series_idx, :,:] )])



        left_offset = range(truncated_backprop_length)



#         plt.plot(left_offset, batchX[batch_series_idx, :],  color="blue")



        plt.plot(left_offset, batchY[batch_series_idx, :,0] , color="red")



        plt.plot(left_offset, single_output_series , color="green")







    plt.draw()



    plt.pause(0.0001)

    

    





tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs





X = tf.placeholder(tf.float32, [None, truncated_backprop_length, inputs])   #create variable objects

y = tf.placeholder(tf.float32, [None, truncated_backprop_length, output])





basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object

rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static



learning_rate = 0.001   #small learning rate so we don't overshoot the minimum



stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor

stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)

outputs = tf.reshape(stacked_outputs, [-1, truncated_backprop_length, output])          #shape of results

 

loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method

training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 



init = tf.global_variables_initializer()         







epochs = 5000   #number of iterations or training cycles, includes both the FeedFoward and Backpropogation



with tf.Session() as sess:

    init.run()

    loss_list= []

    for ep in range(epochs):

        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})



        if ep % 100 == 0:

            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})



            print(ep, "\tMSE:", mse)

            loss_list.append(mse)

            y_pred = sess.run(outputs, feed_dict={X: x_batches})



#             plot(loss_list, y_pred, x_batches, y_batches,inputs,output)



    y_pred = sess.run(outputs, feed_dict={X: X_test})

    

    print(y_pred)



plt.close()



plt.title("Forecast vs Actual", fontsize=14)

yt = y_pred[0,-output,:]



for j in range(0,output):

    yt= np.insert(yt,0,None)

    

p = Y_test[0,:,0]

            

plt.plot(pd.Series(np.ravel(p)), label="Actual")

#plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)

plt.plot(pd.Series(np.ravel(yt)), label="Forecast")



plt.legend(loc="upper left")

plt.xlabel("Time Periods")



plt.show()