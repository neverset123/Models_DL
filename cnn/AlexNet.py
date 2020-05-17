#!/usr/bin/env python
# -*- coding:utf-8 -*-

 
import tensorflow as tf

def print_acrivations(t):
    print(t.op.name," ",t.get_shape().as_list())
 
 
NUM_CHANNELS = 1
CONV1_SIZE = 11
CONV1_DEEP = 96
 
CONV2_SIZE = 5
CONV2_DEEP = 256
 
CONV3_SIZE = 3
CONV3_DEEP = 384
 
CONV4_SIZE = 3
CONV4_DEEP = 384 #256
 
CONV5_SIZE = 3
CONV5_DEEP = 256

FC6_SIZE = 4096
FC7_SIZE = 4096
 
OUTPUT_NODE = 10
 
def AlexNet(input_tensor,train,regularizer):

    # 第一个卷积层
    with tf.variable_scope("conv1"):
        #conv1_weights = tf.get_variable(name="weight",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
        #                                dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_weights = tf.get_variable(name="weight",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                        dtype=tf.float32,initializer=tf.initializers.he_normal())                            
        conv1_biases = tf.get_variable(name="bias",shape=[CONV1_DEEP],
                                       dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        #conv1 = tf.nn.conv2d(input=input_tensor,filter=conv1_weights,strides=[1,4,4,1],padding="SAME")
        conv1 = tf.nn.conv2d(input=input_tensor,filter=conv1_weights,strides=[1,1,1,1],padding="SAME")
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        print_acrivations(conv1)
 
        lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn1")
        #pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        print_acrivations(pool1)
 
    with tf.variable_scope("conv2"):
        #conv2_weights = tf.get_variable(name="weight",shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
        #                                dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_weights = tf.get_variable(name="weight",shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                        dtype=tf.float32,initializer=tf.initializers.he_normal())                                
        conv2_biases = tf.get_variable(name="bias",shape=[CONV2_DEEP],
                                       dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(input=pool1,filter=conv2_weights,strides=[1,1,1,1],padding="SAME")
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
        print_acrivations(conv2)
        lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn2")
        #pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
        print_acrivations(pool2)
 
    with tf.variable_scope("conv3"):
        #conv3_weights = tf.get_variable(name="weight",shape=[CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP],
        #                                dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_weights = tf.get_variable(name="weight",shape=[CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP],
                                        dtype=tf.float32,initializer=tf.initializers.he_normal())
        conv3_biases = tf.get_variable(name="bias",shape=[CONV3_DEEP],
                                       dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(input=pool2,filter=conv3_weights,strides=[1,1,1,1],padding="SAME")
        conv3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
        print_acrivations(conv3)
 
    with tf.variable_scope("conv4"):
        #conv4_weights = tf.get_variable(name="weight",shape=[CONV4_SIZE,CONV4_SIZE,CONV3_DEEP,CONV4_DEEP],
        #                                dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_weights = tf.get_variable(name="weight",shape=[CONV4_SIZE,CONV4_SIZE,CONV3_DEEP,CONV4_DEEP],
                                        dtype=tf.float32,initializer=tf.initializers.he_normal())
        conv4_biases = tf.get_variable(name="bias",shape=[CONV4_DEEP],
                                       dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(input=conv3,filter=conv4_weights,strides=[1,1,1,1],padding="SAME")
        conv4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
        print_acrivations(conv4)

    with tf.variable_scope("conv5"):
        #conv5_weights = tf.get_variable(name="weight",shape=[CONV5_SIZE,CONV5_SIZE,CONV4_DEEP,CONV5_DEEP],
        #                                dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_weights = tf.get_variable(name="weight",shape=[CONV5_SIZE,CONV5_SIZE,CONV4_DEEP,CONV5_DEEP],
                                        dtype=tf.float32,initializer=tf.initializers.he_normal())
        conv5_biases = tf.get_variable(name="bias",shape=[CONV5_DEEP],
                                       dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(input=conv4,filter=conv5_weights,strides=[1,1,1,1],padding="SAME")
        conv5 = tf.nn.relu(tf.nn.bias_add(conv5,conv5_biases))
        print_acrivations(conv5)
        #pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pool5")
        pool5 = tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool5")
        print_acrivations(pool5)
 
    pool5_shape = pool5.get_shape().as_list()
    nodes = pool5_shape[1]*pool5_shape[2]*pool5_shape[3]
    dense = tf.reshape(pool5,shape=[-1,nodes])      #向量化
 
    with tf.variable_scope("fc6"):
        #fc6_weights = tf.get_variable(name="weight",shape=[nodes,FC6_SIZE],dtype=tf.float32,
        #                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc6_weights = tf.get_variable(name="weight",shape=[nodes,FC6_SIZE],dtype=tf.float32,
                                      initializer=tf.initializers.he_normal())                              
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc6_weights))
 
        fc6_biases = tf.get_variable(name="bias",shape=[FC6_SIZE],dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
        fc6 = tf.nn.relu(tf.add(tf.matmul(dense,fc6_weights),fc6_biases))
 
        #if train:
        #    fc6 = tf.nn.dropout(fc6,keep_prob=0.7)
        keep_prob=tf.cond(train, lambda: 0.7, lambda: 1.0)
        fc6 = tf.nn.dropout(fc6,keep_prob=keep_prob)


    with tf.variable_scope("fc7"):
        #fc7_weights = tf.get_variable(name="weight",shape=[FC6_SIZE,FC7_SIZE],dtype=tf.float32,
        #                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc7_weights = tf.get_variable(name="weight",shape=[FC6_SIZE,FC7_SIZE],dtype=tf.float32,
                                      initializer=tf.initializers.he_normal())
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc7_weights))
        fc7_biases = tf.get_variable(name="bias",shape=[FC7_SIZE],dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.relu(tf.add(tf.matmul(fc6,fc7_weights),fc7_biases))
        #if train:
        #    fc7 = tf.nn.dropout(fc7,keep_prob=0.7)
        keep_prob=tf.cond(train, lambda: 0.7, lambda: 1.0)
        fc7 = tf.nn.dropout(fc7,keep_prob=keep_prob)

    with tf.variable_scope("fc8"):
        #fc8_weights = tf.get_variable(name="weight",shape=[FC7_SIZE,OUTPUT_NODE],dtype=tf.float32,
        #                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc8_weights = tf.get_variable(name="weight",shape=[FC7_SIZE,OUTPUT_NODE],dtype=tf.float32,
                                      initializer=tf.initializers.he_normal())                              
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc8_weights))
        fc8_biases = tf.get_variable(name="bias",shape=[OUTPUT_NODE],dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
        fc8 = tf.add(tf.matmul(fc7,fc8_weights),fc8_biases)
 
    return fc8