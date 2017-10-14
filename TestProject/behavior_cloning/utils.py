from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import six
import tensorflow as tf
import pickle


from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.training import  basic_session_run_hooks
from tensorflow.python.training.basic_session_run_hooks import LoggingTensorHook
from behavior_cloning import tf_util
from behavior_cloning import load_policy

class ReportLossHook(LoggingTensorHook):
    returns = {}
    def __init__(self, returns, tensors, every_n_iter):    
        self.returns = returns
        self.returns.clear()
        super().__init__(tensors=tensors, every_n_iter=every_n_iter)
        
    def _log_tensors(self, tensor_values):
        for tag in self._tag_order:
            self.returns[self._iter_count] = tensor_values[tag]
            
            
def normalize(observations):
    """Normalizes observations data by mean and sd
    Returns 
    mean - mean for observations (vector)
    sd - standard deviation for observations (vector)
    """
    
    mean = np.mean(observations, axis= 0)
    sd = np.std(observations, axis=0)
    sd=sd+0.00001
    for idx, val in enumerate(observations):
        observations[idx] = (val-mean)/ (sd)
    return mean, sd

def generate_expert_file(filetogenerate,policyfile='./experts/Humanoid-v1.pkl', envname='Humanoid-v1', max_steps=1000, num_rollouts=20):
    """generates expert_file(provided by berkley DRL class) with policy for environments
    Returns expert data with observations and actions
    """
    
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
    return expert_data

        
        
