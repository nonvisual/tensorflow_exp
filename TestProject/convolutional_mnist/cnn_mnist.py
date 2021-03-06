'''
Created on Sep 19, 2017

@author: ubuntu
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import shape

import numpy as np
import tensorflow as tf


# Imports
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=4,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print(shape(conv1))
    print(shape(pool1))
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=12,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print(shape(conv2))
    print(shape(pool2))
  
  
  
    # Convolution layer 3
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=26,
      kernel_size=[4, 4],
      padding="valid",
      activation=tf.nn.relu)
    
    #print(shape(conv3))


    # Dense Layer
    pool2flat = tf.reshape(conv3, [-1, 26])
    print(shape(pool2flat))

    
    
    
    #dense = tf.layers.dense(inputs=pool2flat, units=15, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=pool2flat, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




if __name__ == "__main__":
    tf.reset_default_graph()
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_actions = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_actions = np.asarray(mnist.test.labels, dtype=np.int32)
    
    beh_clone = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model4")
    
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_actions,
    batch_size=200,
    num_epochs=None,
    shuffle=True)
    beh_clone.train(
    input_fn=train_input_fn,
    steps=30000,
    hooks=[logging_hook])
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_actions,
    num_epochs=1,
    shuffle=False)
    eval_results = beh_clone.evaluate(input_fn=eval_input_fn)
    
    print(eval_results)
    print("Done")
