

import tensorflow as tf
from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data


##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", one_hot=True)


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
dropout_keep_proba = 0.5
epochs = 3
batch_size = 32

# Architecture
input_size = 784
image_width, image_height = 28, 28
n_classes = 10

# Other
print_interval = 500
random_seed = 123


##########################
### WRAPPER FUNCTIONS
##########################

def conv2d(input_tensor, output_channels,
           kernel_size=(5, 5), strides=(1, 1, 1, 1),
           padding='SAME', activation=None, seed=None,
           name='conv2d'):

    with tf.name_scope(name):
        input_channels = input_tensor.get_shape().as_list()[-1]
        weights_shape = (kernel_size[0], kernel_size[1],
                         input_channels, output_channels)

        weights = tf.Variable(tf.truncated_normal(shape=weights_shape,
                                                  mean=0.0,
                                                  stddev=0.01,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=(output_channels,)), name='biases')
        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding)

        act = conv + biases
        if activation is not None:
            act = activation(conv + biases)
        return act


def fully_connected(input_tensor, output_nodes,
                    activation=None, seed=None,
                    name='fully_connected'):

    with tf.name_scope(name):
        input_nodes = input_tensor.get_shape().as_list()[1]
        weights = tf.Variable(tf.truncated_normal(shape=(input_nodes,
                                                         output_nodes),
                                                  mean=0.0,
                                                  stddev=0.01,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[output_nodes]), name='biases')

        act = tf.matmul(input_tensor, weights) + biases
        if activation is not None:
            act = activation(act)
        return act

    
##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)

    test_shape=(1,5,5,16)
    input_layer_test = tf.Variable(tf.truncated_normal(shape=test_shape,
                                                  mean=0.0,
                                                  stddev=0.01,
                                                  dtype=tf.float32,
                                                  seed=100),
                              name='input_layer_test')

    conv_test=conv2d(input_tensor=input_layer_test,
                    output_channels=400,
                    kernel_size=(5,5),
                    strides=(1,1,1,1),
                    padding='VALID',
                    activation=None,
                    name='conv_test')                       


# In[3]:


import numpy as np

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    print(conv_test)
    

