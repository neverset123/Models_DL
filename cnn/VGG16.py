import tensorflow as tf
import numpy as np
 
NUM_CHANNELS = 1
#block1
CONV1_SIZE = 3
CONV1_DEEP = 64
#block2
CONV2_SIZE = 3
CONV2_DEEP = 128
#block3
CONV3_SIZE = 3
CONV3_DEEP = 256
#block4
CONV4_SIZE = 3
CONV4_DEEP = 512
#block5
CONV5_SIZE = 3
CONV5_DEEP = 512
#fc layer
FC6_SIZE = 4096
FC7_SIZE = 4096
 
OUTPUT_NODE = 10
#######################################################################################
def conv_layer(input, input_channels, output_channels, 
               kernel_size, strides, scope, padding='SAME'):
    with tf.variable_scope(scope):
        weights_shape = kernel_size + [input_channels, output_channels]
        weights = tf.get_variable(name="weights",shape=weights_shape,
                                            dtype=tf.float32,initializer=tf.initializers.he_normal()) 
        biases = tf.Variable(tf.zeros(shape=[output_channels]),
                             name='biases')
        conv = tf.nn.conv2d(input=input,
                            filter=weights,
                            strides=strides,
                            padding=padding,
                            name='convolution')
        out = tf.nn.bias_add(conv, biases, name='logits')
        out = tf.nn.relu(out, name='activation')
        return out

def fc_layer(input, output_nodes, scope, regularizer,
             activation=None, seed=None):
    with tf.variable_scope(scope):
        shape = int(np.prod(input.get_shape()[1:]))
        flat_input = tf.reshape(input, [-1, shape])
        weights = tf.get_variable(name="weights",shape=[shape, output_nodes],
                                        dtype=tf.float32, initializer=tf.initializers.he_normal()) 
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(weights))
        biases = tf.Variable(tf.zeros(shape=[output_nodes]),
                             name='biases')
        #biases = tf.get_variable(name=scope+"b",shape=[n_out],dtype=tf.float32,
        #                    initializer=tf.constant_initializer(0.1))
        act = tf.nn.bias_add(tf.matmul(flat_input, weights), biases, 
                             name='logits')

        if activation is not None:
            act = activation(act, name='activation')

        return act

def VGG_16(tf_x,keep_prob, regularizer):

#block1
    conv_layer_1 = conv_layer(input=tf_x,
                              input_channels=NUM_CHANNELS,
                              output_channels=CONV1_DEEP,
                              kernel_size=[CONV1_SIZE, CONV1_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv1')
    
    conv_layer_2 = conv_layer(input=conv_layer_1,
                              input_channels=CONV1_DEEP,
                              output_channels=CONV1_DEEP,
                              kernel_size=[CONV1_SIZE, CONV1_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv2')    
    
    pool_layer_1 = tf.nn.max_pool(conv_layer_2,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool1') 
#block2
    conv_layer_3 = conv_layer(input=pool_layer_1,
                              input_channels=CONV1_DEEP,
                              output_channels=CONV2_DEEP,
                              kernel_size=[CONV2_SIZE, CONV2_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv3')    
    
    conv_layer_4 = conv_layer(input=conv_layer_3,
                              input_channels=CONV2_DEEP,
                              output_channels=CONV2_DEEP,
                              kernel_size=[CONV2_SIZE, CONV2_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv4')    
    
    pool_layer_2 = tf.nn.max_pool(conv_layer_4,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool2') 
#block3
    conv_layer_5 = conv_layer(input=pool_layer_2,
                              input_channels=CONV2_DEEP,
                              output_channels=CONV3_DEEP,
                              kernel_size=[CONV3_SIZE, CONV3_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv5')        
    
    conv_layer_6 = conv_layer(input=conv_layer_5,
                              input_channels=CONV3_DEEP,
                              output_channels=CONV3_DEEP,
                              kernel_size=[CONV3_SIZE, CONV3_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv6')      
    
    conv_layer_7 = conv_layer(input=conv_layer_6,
                              input_channels=CONV3_DEEP,
                              output_channels=CONV3_DEEP,
                              kernel_size=[CONV3_SIZE, CONV3_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv7')
    
    pool_layer_3 = tf.nn.max_pool(conv_layer_7,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool3') 
#block4
    conv_layer_8 = conv_layer(input=pool_layer_3,
                              input_channels=CONV3_DEEP,
                              output_channels=CONV4_DEEP,
                              kernel_size=[CONV4_SIZE, CONV4_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv8')      
    
    conv_layer_9 = conv_layer(input=conv_layer_8,
                              input_channels=CONV4_DEEP,
                              output_channels=CONV4_DEEP,
                              kernel_size=[CONV4_SIZE, CONV4_SIZE],
                              strides=[1, 1, 1, 1],
                              scope='conv9')     
    
    conv_layer_10 = conv_layer(input=conv_layer_9,
                               input_channels=CONV4_DEEP,
                               output_channels=CONV4_DEEP,
                               kernel_size=[CONV4_SIZE, CONV4_SIZE],
                               strides=[1, 1, 1, 1],
                               scope='conv10')   
    
    pool_layer_4 = tf.nn.max_pool(conv_layer_10,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool4') 
#block5
    conv_layer_11 = conv_layer(input=pool_layer_4,
                               input_channels=CONV4_DEEP,
                               output_channels=CONV5_DEEP,
                               kernel_size=[CONV5_SIZE, CONV5_SIZE],
                               strides=[1, 1, 1, 1],
                               scope='conv11')   
    
    conv_layer_12 = conv_layer(input=conv_layer_11,
                               input_channels=CONV5_DEEP,
                               output_channels=CONV5_DEEP,
                               kernel_size=[CONV5_SIZE, CONV5_SIZE],
                               strides=[1, 1, 1, 1],
                               scope='conv12')   

    conv_layer_13 = conv_layer(input=conv_layer_12,
                               input_channels=CONV5_DEEP,
                               output_channels=CONV5_DEEP,
                               kernel_size=[CONV5_SIZE, CONV5_SIZE],
                               strides=[1, 1, 1, 1],
                               scope='conv13') 
    
    pool_layer_5 = tf.nn.max_pool(conv_layer_12,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool5')     
#block6
    fc_layer_1 = fc_layer(input=pool_layer_5, 
                          output_nodes=FC6_SIZE,
                          regularizer=regularizer,
                          activation=tf.nn.relu,
                          scope='fc1')
    fc_layer_1_drop=tf.nn.dropout(fc_layer_1, keep_prob=keep_prob, name='fc_layer_1_drop')
    
    fc_layer_2 = fc_layer(input=fc_layer_1_drop, 
                          output_nodes=FC7_SIZE,
                          regularizer=regularizer,
                          activation=tf.nn.relu,
                          scope='fc2')
    fc_layer_2_drop=tf.nn.dropout(fc_layer_2, keep_prob=keep_prob, name='fc_layer_2_drop')

#block output
    out_layer = fc_layer(input=fc_layer_2_drop, 
                         output_nodes=OUTPUT_NODE,
                         regularizer=regularizer,
                         activation=None,
                         scope='output_layer')
 
    return out_layer