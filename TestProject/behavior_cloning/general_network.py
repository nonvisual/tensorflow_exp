'''
Created on Oct 1, 2017
General network implementation
without estimator use
@author: ubuntu
'''
import tensorflow as tf
from numpy import shape
import math
import numpy as np
from six.moves import xrange
import os
import sys
import time
import pickle
from random import shuffle, random
from tensorflow.contrib.batching.ops.gen_batch_ops import batch
import behavior_cloning.utils
from behavior_cloning.utils import normalize
from pip.utils import logging
from random import randint

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


def build_graph(input_placeholder,  output_size, layers=[50,50], activation=tf.nn.relu):
    previous_layer=input_placeholder
    previous_size=int(shape(input_placeholder)[1])
    print(int(previous_size))
    #add hidden layers
    drop_rate=0.2
    count=0
    for size in layers:
        print(count)
        count+=1
        with tf.name_scope('hidden' + str(count)):
            weights = tf.Variable(
                tf.truncated_normal([previous_size, size],
                        stddev=1.0/math.sqrt(previous_size)),
                                  name='weights')
            biases = tf.Variable(tf.zeros([size]),
                     name='biases')
            previous_layer= activation(tf.matmul(previous_layer, weights) + biases)
            previous_layer = tf.layers.dropout(inputs=previous_layer, rate=drop_rate)
            previous_size=int(shape(previous_layer)[1])
            
    # add output layer
    with tf.name_scope('output'):
        weights = tf.Variable(
            tf.truncated_normal([previous_size, output_size],
                    stddev=1.0/math.sqrt(previous_size)) ,
                              name='weights')
        biases = tf.Variable(tf.zeros([output_size]),
                 name='biases')
        output= tf.matmul(previous_layer, weights) + biases
    res = tf.identity(output, name="logits")
    return output


def loss(output_layer, labels):
    """Calculates the loss from the output_layer and the labels.
    Args:
      output_layer: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, float[batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    #labels = tf.to_int64(labels)
    mse = tf.losses.mean_squared_error(labels=labels, predictions=output_layer)
    return mse


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
    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

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
  
def prediction(output_layer):
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
    #mse = tf.losses.mean_squared_error(labels=labels, predictions=output_layer)
    return output_layer

def fill_feed_dict(data_set, input_pl, output_pl, batch_size):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      input_pl: The images placeholder, from placeholder_inputs().
      output_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    inputs_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
        input_pl: inputs_feed,
        output_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            mse,
            input_placeholder,
            labels_placeholder,
            data):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    mse: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  count = 0  # Counts the number of correct predictions.
  steps_per_epoch = 10
  num_examples = 100
  for step in xrange(steps_per_epoch):
    feed_dict = {input_placeholder: data['inputs'][step*num_examples:step*num_examples+num_examples],
                                       labels_placeholder:data['labels'][step*num_examples:step*num_examples+num_examples]}
    count+=1
    val = sess.run(mse, feed_dict=feed_dict)
    #print(' MSE ' , val)
  
   
def run_training(data,  batch_size, steps=1000,learning_rate=0.001):
    """Data contains inputs and labels"""

    input_size= shape(data['inputs'])[1]
    output_size= shape(data['labels'])[1]
    print(data)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        inputs_placeholder, labels_placeholder = placeholder_inputs(batch_size,input_size=input_size, output_size=output_size)
        
        # Build a Graph that computes predictions from the inference model.
        logits = build_graph(inputs_placeholder,
                                 output_size=output_size,activation=tf.nn.tanh)
        
        # Add to the Graph the Ops for loss_fn calculation.
        loss_fn = loss(logits, labels_placeholder)
        
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss_fn, learning_rate)
        
        # Add the Op to compare the logits to the labels during evaluation.
        eval = evaluation(logits, labels_placeholder)
        
        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()
        
        # Add the variable initializer Op.
        init = tf.global_variables_initializer()
        
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter('./models/', sess.graph)
        
        # And then after everything is built:
        
        # Run the Op to initialize the variables.
        sess.run(init)
        #shuffle(data)
        start=0
        # Start the training loop.
        for step in xrange(steps):
            start_time = time.time()
        
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            start =randint(0,int(shape(data['inputs'])[0])-batch_size)
            feed_dict = {inputs_placeholder: data['inputs'][start:start+batch_size],
                                       labels_placeholder:data['labels'][start:start+batch_size]}
            #print(data['labels'][start:start+batch_size])
            start+=batch_size
            #print(shape(data['inputs'])[0])

            if start > int(shape(data['inputs'])[0]):
                start=0
            #print(start)
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss_fn` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss_fn],
                                     feed_dict=feed_dict)
        
            duration = time.time() - start_time
        
            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss_fn = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
        
            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 100 == 0 or (step + 1) == steps:
                checkpoint_file = os.path.join('./models', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                eval,
                inputs_placeholder,
                labels_placeholder,
                data)
    saver.save(sess, './models/mymodel')
    
    return logits
                
            
if __name__ == '__main__':
    with open('Walkerxpert100.pkl', 'rb') as input:
        expert=pickle.load(input)
    data_to_feed= {'inputs': np.array(expert['observations']),
                                      'labels': np.array(expert['actions'][:,0])}
    mean, sd = normalize(data_to_feed['inputs'])
    run_training(data_to_feed, 500, 30000, 0.005)    
    
    tf.reset_default_graph()
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    

    # load model
    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('./models/mymodel.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./models/'))  
    
    graph=tf.get_default_graph()
    inputs= graph.get_tensor_by_name("input_pl:0")
    logits = graph.get_tensor_by_name("logits:0")
                       
    # predictions
    import gym
    env = gym.make('Walker2d-v1')
    max_steps=1000
    returns = []
    observations = []
    actions = []
    for i in range(1):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs=(obs - mean) / (sd)
            #print('obs ', obs[0:10])
            action = sess.run([logits], feed_dict={inputs:obs[None,:]})
            #print('action ', action)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        print(totalr)
        returns.append(totalr)
                                  
                                      
                                      
                                      