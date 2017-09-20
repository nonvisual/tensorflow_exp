'''
Created on Sep 16, 2017

@author: ubuntu
'''
import tensorflow as tf


# initialize weight variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# initialize biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


