'''
Created on Oct 28, 2017

@author: ubuntu
'''

import tensorflow as tf
from numpy import shape
import matplotlib.pyplot as plt
import math
import numpy as np
from six.moves import xrange
import os
import sys
import time
import pickle
from random import shuffle, random
from tensorflow.contrib.batching.ops.gen_batch_ops import batch
from pip.utils import logging
from random import randint
import pandas as pd
from numpy.random import sample
from pyglet.graphics import Batch
from multiprocessing.spawn import prepare

def build_fw_graph(input_placeholder,  output_size, layers=[50,50], activation=tf.nn.relu):
    previous_layer=input_placeholder
    previous_size=int(shape(input_placeholder)[1])
    print(int(previous_size))
    #add hidden layers
    drop_rate=0.2
    count=0
    for size in layers:
#         print(count)
        count+=1
        with tf.name_scope('hidden' + str(count)):
            weights = tf.get_variable('hidden'+ str(count) + "weights",[previous_size, size],
                regularizer=tf.contrib.layers.l2_regularizer(0.8))
            biases = tf.Variable(tf.zeros([size]),
                     name='biases')
            previous_layer= activation(tf.matmul(previous_layer, weights) + biases)
            previous_layer = tf.layers.dropout(inputs=previous_layer, rate=drop_rate)
            previous_size=int(shape(previous_layer)[1])
            
    # add output layer
    with tf.name_scope('output'):
        weights = tf.get_variable('output' + "weights",[previous_size, output_size],
                regularizer=tf.contrib.layers.l2_regularizer(0.8))
        biases = tf.Variable(tf.zeros([output_size]),
                 name='biases')
        output= tf.matmul(previous_layer, weights) + biases
    return output

def prepare_data(window=3):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    rawdata = pd.read_csv("~/Downloads/energydata_original.csv", 
                   date_parser=dateparse)

    print(type(rawdata))
    
    
    data_for_net =pd.DataFrame()
    data_for_net['energy']=""
    data_for_net['date']=""
    data_for_net['month']=""
    data_for_net['day']=""
    data_for_net['hour']=""
    data_for_net['minutes']=""

    for i in range(1,window+1):
        name = 'prev_energy'+str(i)
        data_for_net[name]=""
   
    for i in range(window+1, len(rawdata.index)):
        ix = i-window-1
        data_for_net.loc[ix, 'date'] = pd.to_datetime(rawdata.loc[i, 'date'])
        data_for_net.loc[ix, 'month'] = data_for_net.loc[ix, 'date'].month
        data_for_net.loc[ix, 'day'] =data_for_net.loc[ix, 'date'].day
        data_for_net.loc[ix, 'weekday'] = pd.to_datetime(data_for_net.loc[ix, 'date']).weekday()
        data_for_net.loc[ix, 'hour'] = data_for_net.loc[ix, 'date'].hour
        data_for_net.loc[ix, 'minutes'] = data_for_net.loc[ix, 'date'].minute
        data_for_net.loc[ix, 'energy'] = rawdata.loc[i, 'Appliances']

        for j in range(1,window+1):
            name = 'prev_energy'+str(j)
            data_for_net.loc[ix,name] = rawdata.loc[i-j, 'Appliances']
        print(i)
            
    msk = np.random.rand(len(data_for_net)) < 0.8
    train  = data_for_net[msk]
    test= data_for_net[~msk]
    
    data_for_net.to_csv(path_or_buf ='~/Downloads/energydata_prepared'+str(window)+'.csv', index = False) 
    
def loss(output_layer, labels,reg=0.0):
    """Calculates the loss from the output_layer and the labels.
    Args:
      output_layer: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, float[batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    mse = tf.losses.mean_squared_error(labels=labels, predictions=output_layer)
    return mse + reg*reg_losses

def training(loss, learning_rate):
    """Sets up the training Ops. Taken from tensorflow tutorial.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def normalize(data):
    mean = []
    sds= []
    idx = ["energy", "prev_energy1", "prev_energy2", "prev_energy3", "prev_energy4", "prev_energy5","prev_energy6", "prev_energy7"]
    for i in idx:        
        m = np.mean( data.loc[:, i])
        sd = np.std(data.loc[:, i])
        data.loc[:,i] = (data.loc[:,i]-m)/sd
        mean.append(m)
        sds.append(sd)
    return mean,sds

def normalize_min_max(data):
    max_v = []
    min_v= []
    idx = ["energy", "prev_energy1", "prev_energy2", "prev_energy3", "prev_energy4", "prev_energy5","prev_energy6", "prev_energy7"]
    for i in idx:        
        max = np.max( data.loc[:, i])
        min = np.min(data.loc[:, i])
        data.loc[:,i] = (data.loc[:,i]-min)/(max-min)
        max_v.append(max)
        min_v.append(min)
    return max_v,min_v

def get_weights():
  return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
def placeholder_inputs(batch_size, input_size, output_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    Args:
      batch_size: The batch size will be baked into both placeholders.
      input_size: The size of input vector
      output_size: the size of output vector
    Returns:
      input_placeholder: Input placeholder.
      output_placeholder: Output placeholder.
    """
    input_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           input_size), name='input_pl')
    labels_placeholder = tf.placeholder(tf.float32, shape=(None,output_size), name='output_pl')
    return input_placeholder, labels_placeholder

def evaluation(output_layer, labels):
    """Evaluate the quality of the outptus at predicting the label.
    Args:
      output_layer: Logits tensor, float - [batch_size, output_size].
      labels: Labels tensor, float - [batch_size, output_size].
    Returns:
      Mean squared error
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all output_layer for that example.
    mse = tf.losses.mean_squared_error(labels=labels, predictions=output_layer)
    return mse    

if __name__ == '__main__':
    use_norm= True
    window=7
    use_real=True
    real_path =5
#     prepare_data(7)
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')

    data_for_net = pd.read_csv("~/Downloads/energydata_prepared7.csv", 
                   date_parser=dateparse)
    
    points = data_for_net.ix[:, 'energy'].tolist()
    
   
    
    if use_norm:
#         max, min = normalize_min_max(data_for_net)
        m, sd = normalize(data_for_net)
    
    batch_size=50
    msk = np.random.rand(len(data_for_net)) < 0.8
    train_size= 19500
    train  = data_for_net.loc[0:train_size]
    test= data_for_net.loc[train_size:len(data_for_net)]
    
    input_size = window+12+31+7+6+24
    sy_input,sy_labels = placeholder_inputs(batch_size, input_size, 1)   
    sy_output = build_fw_graph(sy_input, 1,activation=tf.nn.relu, layers=[30,30,20])
    
    loss_fn = loss(sy_output, sy_labels)
    eval = evaluation(sy_output, sy_labels)   
    train_op = training(loss_fn,0.001)    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    steps =  20000
    
    
    for i in range(0,steps):
        sample =train.sample(batch_size)
        inputs = []
        outputs = []
        for j in range(0,len(sample)):
            inp = []
            mnths = np.zeros((12))
            mnths[sample.iloc[j].loc[ 'month']] = 1
            inp.extend(mnths)
            days = np.zeros((31))
            days[sample.iloc[j].loc[  'day']-1] = 1
            inp.extend(days)
            weekdays = np.zeros((7))
            weekdays[int(sample.iloc[j].loc[  'weekday'])] = 1
            inp.extend(weekdays)
            hours = np.zeros((24))
            hours[sample.iloc[j].loc[  'hour']] = 1
            inp.extend(hours)
            mins = np.zeros((6))

            mins[int(sample.iloc[j].loc[  'minutes']/10)] = 1
            inp.extend(mins)
            for k in range(1,window+1):
                name = 'prev_energy'+str(k)
                inp.append(sample.iloc[j].loc[name])
            inputs.append(inp)
            
            out= []
            out.append(sample.iloc[j].loc[ 'energy'])
            outputs.append(out)
        feed_dict = {sy_input: inputs,
                                       sy_labels:outputs}
        if i%10==0:
            print("Training step", i)
        _, loss_value = sess.run([train_op, loss_fn],
                                     feed_dict=feed_dict)
        
#         if (i + 1) % 100 == 0 :
#                 # Evaluate against the training set.
#                 print('Training Data Eval:')
#                 do_eval(sess,
#                 eval,
#                 inputs_placeholder,
#                 labels_placeholder,
#                 data)
    # evaluate on test set
    av_error=0
    real_points = []
    pred_points = []
    
    prev_energy = []
    for i in range(1,window+1):
        name = 'prev_energy'+str(i)
        prev_energy.append(test.iloc[0].loc[name])
   

    for i in range(0,len(test)):    
        inputs = []
        outputs = []
        inp = []
        mnths = np.zeros((12))
        mnths[test.iloc[i].loc[  'month']] = 1
        inp.extend(mnths)
        days = np.zeros((31))
        days[test.iloc[i].loc[ 'day']-1] = 1
        inp.extend(days)
        weekdays = np.zeros((7))
        weekdays[int(test.iloc[i].loc[ 'weekday'])] = 1
        inp.extend(weekdays)
        hours = np.zeros((24))
        hours[test.iloc[i].loc[  'hour']] = 1
        inp.extend(hours)
        mins = np.zeros((6))
        
        mins[int(test.iloc[i].loc[  'minutes']/10)] = 1
        inp.extend(mins)
        
        for k in range(0,window-real_path):
            inp.append(prev_energy[k])
            
        # last real_path observations are real
        for k in range(window-real_path,window):
            name = 'prev_energy'+str(k+1)
            inp.append(test.iloc[i].loc[name])
      
           
               
       
#         inp.append(test.iloc[i].loc[ 'prev_energy1'])
#         inp.append(test.iloc[i].loc[ 'prev_energy2'])
#         inp.append(test.iloc[i].loc[ 'prev_energy3'])
        inputs.append(inp)
           
        out = [] 
        out.append(test.iloc[i].loc[ 'energy'])    
        outputs.append(out)
        feed_dict = {sy_input: inputs}
        prediction =  sess.run([sy_output],
                                     feed_dict=feed_dict) [0][0][0]  
        
#         error = sess.run([eval], {sy_output : prediction,
#                                   sy_labels : outputs})

        # update saved energy list
        prev_energy.pop(window-1)
        prev_energy.insert(0, prediction)
        real_energy = outputs[0][0]
        pred_energy = prediction
        
       
        if use_norm:
            real_energy = outputs[0][0]*sd[0]+m[0]
            pred_energy = prediction*sd[0]+m[0]            
#             real_energy = outputs[0][0]*(max[0]-min[0])+min[0]
#             pred_energy = prediction*(max[0]-min[0])+min[0]
        error =np.square((real_energy- pred_energy))
        
        real_points.append(real_energy)
        pred_points.append(pred_energy)
        print("Real energy", real_energy, "prediction",pred_energy, "error", error )
        av_error+=error
    av_error/=len(test)
    print("Average error", av_error)
    
    plt.figure() 
    plt.plot(real_points)
    plt.plot(pred_points) 
    plt.title('energy_'+str(steps)+"_"+str(real_path))
    plt.legend(['real', 'pred'], loc='best') 
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        