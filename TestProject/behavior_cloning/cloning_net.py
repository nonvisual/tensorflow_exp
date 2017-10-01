
'''
Created on Sep 23, 2017
Neural network doing imitation learning
Clone expert policy for Humanoid-v1 by Mujoco library
@author: ubuntu
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import shape

import numpy as np
import tensorflow as tf
from pip._vendor.webencodings import labels
from tensorflow.python.training.training_util import global_step
from pyglet.extlibs.png import itertools
from behavior_cloning import tf_util
from tensorflow.python.ops import init_ops
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import SummarySaverHook
from behavior_cloning import utils
from behavior_cloning.utils import ReportLossHook
 

def clone_model_fn(features, labels, mode):
    """Model function for DN."""
    
    # Input Layer
    input_layer = tf.convert_to_tensor(features['x'])

    hidden_layer1 =  tf.layers.dense(inputs=input_layer, activation=tf.nn.tanh,kernel_initializer=init_ops.glorot_uniform_initializer(),bias_initializer=init_ops.glorot_normal_initializer(), units=64)
    drop1 = tf.layers.dropout(inputs=hidden_layer1, rate=0.2)


    output_layer = tf.layers.dense(inputs=drop1, activation=tf.nn.tanh, bias_initializer=init_ops.glorot_uniform_initializer(),units=17)
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)      
      "actions": output_layer,
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(output_layer, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

   
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=output_layer)
    
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "mse": tf.metrics.mean_squared_error(
          labels=labels, predictions=predictions["actions"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    
def normalize (observations):  
    # normalizes observations data by mean and sd
    mean = np.mean(observations, axis= 0)
    sd = np.std(observations, axis=0)
    sd=sd+0.0001
    for idx, val in enumerate(observations):
        observations[idx] = (val-mean)/ (sd)
    return mean, sd



if __name__ == '__main__':
    import pickle
    # load expert policy
    with open('expert150.pkl', 'rb') as input:
        expert=pickle.load(input)  
    mean, sd = normalize(expert['observations'])    
    tf.reset_default_graph()
    train_ratio=0.8
    size = int(shape(expert['observations'][:,0])[0])
    train_size= int(train_ratio*size);
    print('size:',size)
    print('train size:',train_size)
    
    beh_clone = tf.estimator.Estimator(
    model_fn=clone_model_fn)
    
    train_data = np.array(expert['observations'][0:train_size-1])# Returns nps.array
    train_data = train_data.astype(np.float32)
    train_actions = np.array(expert['actions'][0:train_size-1,0])
   
    eval_data = np.array(expert['observations'][train_size:size])
    eval_data = eval_data.astype(np.float32)
    eval_actions = np.array(expert['actions'][train_size:size,0])
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(train_data)},
        y=np.array(train_actions),
        batch_size=500,         
        num_epochs=None,
        shuffle=True)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(eval_data)},
        y=np.array(eval_actions),
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    
    tensors_to_log = {'loss': 'mserror'}
    tf.logging.set_verbosity(tf.logging.INFO)
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    beh_clone.train(
        input_fn=train_input_fn,
        steps=3000,hooks=[])
    
    print('---------------model trained')
    eval_results = beh_clone.evaluate(input_fn=eval_input_fn,steps=100)
    
    print(eval_results.keys())
    print(eval_results.values())

    returns = []
    observations = []
    actions = []
    # try on env
   

    import gym
    env = gym.make('Humanoid-v1')
    for i in range(1):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = (obs - mean) / (sd)
            predict_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(obs[None, :]).astype(np.float32)},
            num_epochs=None,
            shuffle=True)
            #print('obs:', obs[0:20])
            action = beh_clone.predict(input_fn=predict_input)

            predictions = list(p["actions"] for p in itertools.islice(action, 1))
            
            #print("Predictions: ", predictions[0][None, :])
            observations.append(obs)
            actions.append(predictions[0][None, :])
            obs, r, done, _ = env.step(predictions[0][None, :])
            totalr += r
            steps += 1     
            env.render() 
            if steps > 1000:
                break  
        returns.append(totalr)
        
    print(returns)  
    with open('actions_clone.pkl', 'wb') as output:
        pickle.dump(actions, output, pickle.HIGHEST_PROTOCOL)
    print('actions saved')     
    print("Done")
    
    
    
    
