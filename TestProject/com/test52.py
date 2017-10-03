'''
Created on Sep 14, 2017

@author: ubuntu
'''
from numpy import square

import numpy as np
import tensorflow as tf;


node1 = tf.constant(3.0);
node2 = tf.constant(5.0);

print("nodes:",node1,node2)

session = tf.Session()
print("nodes evaluated:", session.run([node1,node2]))

node3= tf.add(node1,node2,"addNode")
print(node3)
print(session.run(node3))

a=tf.placeholder(tf.float32,[None])
b=tf.placeholder(tf.float32,[None])

addNode=tf.add(a,b)
print(session.run(addNode,{a:[1,3],b:[2,4]}))


W=tf.Variable([.3],dtype=tf.float32)
b=tf.Variable([-.3],dtype=tf.float32)
x=tf.placeholder(tf.float32)
linear_model=W*x + b
init=tf.global_variables_initializer()
session.run(init)

print(session.run(linear_model,{x:[1,2,3,4]}))

y=tf.placeholder(tf.float32)
squared_deltas=tf.square(linear_model-y)
loss=tf.reduce_sum(squared_deltas)
print(session.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
session.run([fixW, fixb])
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


# Using train API
optimizer=tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
session.run(init)
for i in range(1000):
    session.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    
print(session.run([W,b]))


#===================Using estimator

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)



