import tensorflow as tf
import numpy as np

training_steps = 30000

data = []
label = []

for i in range(200):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(0, 2)


    if x1 ** 2 + x2 ** 2 <= 1:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)


data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label).reshape(-1, 1)


def hidden_layer(input_tensor, weight1, bias1, weight2, bias2, weight3, bias3):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + bias1)
    layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)
    return tf.matmul(layer2, weight3) + bias3


xs = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
ys = tf.placeholder(tf.float32, shape=(None, 1), name="y-output")

weight1 = tf.Variable(tf.truncated_normal([2, 10], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[10]))
weight2 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
weight3 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
bias3 = tf.Variable(tf.constant(0.1, shape=[1]))

sample_size = len(data)

y = hidden_layer(xs, weight1, bias1, weight2, bias2, weight3, bias3)

error_loss = tf.reduce_sum(tf.pow(ys-y, 2))
tf.add_to_collection("losses", error_loss)

regularizer = tf.contrib.layers.l2_regularizer(0.01)
retularization = regularizer(weight1) + regularizer(weight2) + regularizer(weight3)
tf.add_to_collection("losses", retularization)

loss = tf.add_n(tf.get_collection("losses"))

train_op = tf.train.AdamOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(training_steps):
        sess.run(train_op, feed_dict={xs: data, ys: label})

        if i % 2000 == 0:
            loss_value = sess.run(loss, feed_dict={xs: data, ys: label})
            print("After %d steps, mse_loss: %f" %(i, loss_value))