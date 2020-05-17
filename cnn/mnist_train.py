# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference as mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
 
MODEL_SAVE_PATH = "./"
MODEL_NAME = "model.ckpt"
 
 
def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.OUTPUT_NODE], name='y-input')
 
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
 
    y = mnist_inference.inference(x, True, regularizer)
 
    global_step = tf.Variable(0, trainable=False)
 
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # print(tf.trainable_variables())
    # [<tf.Variable 'layer1-conv1/weight:0' shape=(5, 5, 1, 32) dtype=float32_ref>,
    # <tf.Variable 'layer1-conv1/bias:0' shape=(32,) dtype=float32_ref>,
    # <tf.Variable 'layer3-conv2/weight:0' shape=(5, 5, 32, 64) dtype=float32_ref>,
    # <tf.Variable 'layer3-conv2/bias:0' shape=(64,) dtype=float32_ref>,
    # <tf.Variable 'layer5-fc1/weight:0' shape=(3136, 512) dtype=float32_ref>,
    # <tf.Variable 'layer5-fc1/bias:0' shape=(512,) dtype=float32_ref>,
    # <tf.Variable 'layer6-fc2/weight:0' shape=(512, 10) dtype=float32_ref>,
    # <tf.Variable 'layer6-fc2/bias:0' shape=(10,) dtype=float32_ref>]
 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # print(tf.get_collection('losses'))
    # #[<tf.Tensor 'layer5-fc1/l2_regularizer:0' shape=() dtype=float32>,
    # <tf.Tensor 'layer6-fc2/l2_regularizer:0' shape=() dtype=float32>]
 
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
 
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, [BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS])
 
            _, loss_valuue, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
 

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_valuue))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
 
 
def main(argv=None):
    mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)
    train(mnist)
 
 
if __name__ == '__main__':
    tf.app.run()