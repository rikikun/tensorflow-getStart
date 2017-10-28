"""This module does blah blah."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

session = tf.Session()
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
session.run(init)

y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(10000):
    session.run(train, {x: [1, 2, 3, 4], y: [5, 9, 13, 17]})
print session.run([W, b])

print session.run(loss, {x: [1, 2, 3, 4], y: [5, 9, 13, 17]})

file_writer = tf.summary.FileWriter(
    '/Users/Thanakornthanprasit/ml/log-graph', session.graph)
