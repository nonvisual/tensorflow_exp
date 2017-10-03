#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3 run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import sys
import behavior_cloning
from behavior_cloning.cloning_net import clone_model_fn
from tensorflow.contrib.learn.python.learn.estimators.estimator import infer_real_valued_columns_from_input,\
    infer_real_valued_columns_from_input_fn
from tensorflow.contrib.layers.python.layers.feature_column import real_valued_column
from pyglet.extlibs.png import itertools
sys.path.append('/home/ubuntu/git/tensorflow_exp/TestProject/behavior_cloning')

import gym
import pickle

from behavior_cloning import load_policy
from behavior_cloning import tf_util
from behavior_cloning import cloning_net
import mujoco_py
import numpy as np
import tensorflow as tf
from numpy import shape

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        
    return expert_data


def train_and_save(expertfile, actionfile='action.pkl',iterations=10000):
    with open(expertfile, 'rb') as input:
        expert=pickle.load(input)
        
    train_ratio=0.8
    size = int(shape(expert['observations'][:,0])[0])
    train_size= int(train_ratio*size);
    print('size:',size)
    print('train size:',train_size)
    
    
    tf.reset_default_graph()
    
    train_data = np.array(expert['observations'][0:train_size-1])# Returns nps.array
    train_actions = np.array(expert['actions'][0:train_size-1,0])
    eval_data = np.array(expert['observations'][train_size:size])
    eval_actions = np.array(expert['actions'][train_size:size,0])
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(train_data)},
    y=np.array(train_actions),
    batch_size=100, 
    num_epochs=None,
    shuffle=True)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(eval_data)},
    y=np.array(eval_actions),
    batch_size=200,
    num_epochs=None,
    shuffle=True)
    
    obs_column =real_valued_column(
    "x",
    dimension=376,
    default_value=None,
    dtype=tf.float32
    )
    print('shape of train actions', shape(train_actions))
    print('shape of train data', shape(train_data))
    
    
    
   
    tensors_to_log = {"loss"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=10)
    
    estimator = tf.estimator.DNNRegressor(
        feature_columns=[obs_column],
        label_dimension=17, 
        dropout=0.1,
        activation_fn=tf.nn.tanh,
        hidden_units=[120,54],
        optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.01
    ))
    
    estimator.train(input_fn=train_input_fn, steps=iterations)
    print('---------------------Model is trained')
    estimator.export_savedmodel('./trained_model', train_input_fn)
    #saver = tf.train.Saver(max_to_keep=1)
    
    #with open('model.pl;', 'wb') as output:
    #        pickle.dump(estimator,output, pickle.HIGHEST_PROTOCOL)
    
    metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1000)
    print(metrics.keys())
    print(metrics.values())
    
    
    returns = []
    observations = []
    actions = []
    # try on env
    with tf.Session() as sess:
        #savePath = saver.save(sess, 'my_model.ckpt')
        tf_util.initialize()

        import gym
        env = gym.make('Humanoid-v1')
        for i in range(1):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                predict_input=tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array(obs[None,:])},
                num_epochs=None,
                shuffle=True)
                #print('predicting action, obs shape:', shape(obs))
                action = estimator.predict( input_fn=predict_input)
                #predictions = list(itertools.islice(action,1))
                predictions = list(p["predictions"] for p in itertools.islice(action, 1))
                #print('pr shape: ',shape(predictions))
                #print('obs:', obs[0:10])
                #print("Predictions: {}".format(str(predictions)))
                observations.append(obs)
                actions.append(predictions[0][None,:])
                #print('current actions:',actions)
                obs, r, done, _ = env.step(predictions[0][None,:])
                totalr += r
                steps += 1     
                env.render()           
            returns.append(totalr)
  
        
        with open(actionfile, 'wb') as output:
            pickle.dump(actions,output, pickle.HIGHEST_PROTOCOL)
        print('actions saved')     
    return

def play_actions(action_file):
    #===========================================================================
    # with tf.Session() as sess:
    # saver = tf.train.import_meta_graph('someDir/my_model.ckpt.meta')
    # saver.restore(sess, pathModel + 'someDir/my_model.ckpt')
    # # access a variable from the saved Graph, and so on:
    # someVar = sess.run('varName:0')
    #===========================================================================
    
    with open(action_file, 'rb') as input:
        actions=pickle.load(input)
    #tf.reset_default_graph()
    totalr = 0.

    #with tf.Session():
        #tf_util.initialize()
    import gym
    env = gym.make('Humanoid-v1')
    obs = env.reset()
    print(actions)
    for a in actions:
        print(a)
        obs, r, done, _ = env.step(a)
        env.render() 
            #import time
            #time.sleep(0.01) 
        totalr += r
        if done:
            break   
    print('reward:', totalr)  
    return





def generate_expert_file(filetogenerate,policyfile='./experts/Humanoid-v1.pkl', envname='Humanoid-v1', max_steps=1000, num_rollouts=20):
    #generates expert_file(provided by berkley DRL class) with policy for environments
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(policyfile)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(envname)

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                #env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        with open(filetogenerate, 'wb') as output:
            pickle.dump(expert_data,output, pickle.HIGHEST_PROTOCOL)
    return


if __name__ == '__main__':
    #generate rollouts
    generate_expert_file(filetogenerate='Walkerxpert100.pkl',policyfile='./experts/Walker2d-v1.pkl',envname='Walker2d-v1', num_rollouts=150)
    #train_and_save('expert150.pkl', 'action150.pkl',1000000)
    #play_actions('actions_clone.pkl')
   
    


# for interactive console
#sys.argv = input('./experts/Humanoid-v1.pkl Humanoid-v1 --render  --num_rollouts 1').split()
#expert=main()



#===============================================================================
# 
# expert= main()
#      
#     train_ratio=0.8
#     size = int(shape(expert['observations'][:,0])[0])
#     train_size= int(train_ratio*size);
#     print('size:',size)
#     print('train size:',train_size)
#     
#     with open('expert'+ str(size)+'.pkl', 'wb') as output:
#         pickle.dump(expert,output, pickle.HIGHEST_PROTOCOL)
#     
#     with open('expert10000.pkl', 'rb') as input:
#         expert=pickle.load(input)
#     
#     
#     
#     
# 
# 
#     #tf.logging.set_verbosity(tf.logging.INFO)
#     tf.reset_default_graph()
#     
#     train_data = np.array(expert['observations'][0:train_size-1])# Returns nps.array
#     train_actions = np.array(expert['actions'][0:train_size-1,0])
#     eval_data = np.array(expert['observations'][train_size:size])
#     eval_actions = np.array(expert['actions'][train_size:size,0])
#     
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": np.array(train_data)},
#     y=np.array(train_actions),
#     batch_size=100, 
#     num_epochs=None,
#     shuffle=True)
#     
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": np.array(eval_data)},
#     y=np.array(eval_actions),
#     batch_size=100,
#     num_epochs=None,
#     shuffle=True)
#     
#     obs_column =real_valued_column(
#     "x",
#     dimension=376,
#     default_value=None,
#     dtype=tf.float32,
#     normalizer=None
#     )
#     
#     print('shape of train actions', shape(train_actions))
#     print('shape of train data', shape(train_data))
#     
#     
#     
#    
#     tensors_to_log = {"loss"}
#     logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=10)
#     
#     estimator = tf.estimator.DNNRegressor(
#         feature_columns=[obs_column],
#         label_dimension=17, 
#         dropout=0.2,
#         activation_fn=tf.nn.tanh,
#         hidden_units=[100,40],
#         optimizer=tf.train.ProximalAdagradOptimizer(
#         learning_rate=0.01,
#         l1_regularization_strength=0.001
#     ))
#     
#     estimator.train(input_fn=train_input_fn, steps=2000)
#     print('---------------------Model is trained')
#     metrics = estimator.evaluate(input_fn=eval_input_fn, steps=100)
#     print(metrics.keys())
#     print(metrics.values())
#     
#     
#     returns = []
#     observations = []
#     actions = []
#     # try on env
#     with tf.Session():
#         tf_util.initialize()
# 
#         import gym
#         env = gym.make('Humanoid-v1')
#         for i in range(1):
#             print('iter', i)
#             obs = env.reset()
#             done = False
#             totalr = 0.
#             steps = 0
#             while not done:
#                 predict_input=tf.estimator.inputs.numpy_input_fn(
#                 x={"x": np.array(obs[None,:])},
#                 num_epochs=None,
#                 shuffle=True)
#                 print('predicting action, obs shape:', shape(obs))
#                 action = estimator.predict( input_fn=predict_input)
#                 #predictions = list(itertools.islice(action,1))
#                 predictions = list(p["predictions"] for p in itertools.islice(action, 1))
#                 print('pr shape: ',shape(predictions))
#                 print('obs:', obs[0:10])
#                 print("Predictions: {}".format(str(predictions)))
#                 observations.append(obs)
#                 actions.append(predictions[0][None,:])
#                 obs, r, done, _ = env.step(predictions[0][None,:])
#                 totalr += r
#                 steps += 1
#                 env.render()
#             returns.append(totalr)
#         print("--------------------------Now playing")
#         obs = env.reset()    
#         
#         with open('actions'+ str(size)+'.pkl', 'wb') as output:
#              pickle.dump(expert,output, pickle.HIGHEST_PROTOCOL)
# 
#         
#         for a in actions:
#             #predictions = list(itertools.islice(a,1)) 
#             #print("Predictions: {}".format(str(predictions)))
#             obs, r, done, _ = env.step(a)
#             env.render()  
#             if done:
#                 break 
#     
#    
#     print("Done")
#===============================================================================
