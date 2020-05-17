##########################
### DATASET
##########################

# load utilities from ../helper.py
import sys
from helper import download_and_extract_cifar
from helper import Cifar10Loader

dest ='./data/cifar10'
cifar = Cifar10Loader(dest, normalize=True, 
                      zero_center=True,
                      channel_mean_center=False)
cifar.num_train

X, y = cifar.load_test()
half = cifar.num_test // 2
X_test, X_valid = X[:half], X[half:]
y_test, y_valid = y[:half], y[half:]

del X, y


# In[3]:


import tensorflow as tf
import numpy as np

tf.test.gpu_device_name()


# In[4]:


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.001
training_epochs = 30
batch_size = 32

# Other
print_interval = 200

# Architecture
image_width, image_height, image_depth = 32, 32, 3
n_classes = 10


##########################
### WRAPPER FUNCTIONS
##########################

def conv_layer(input, input_channels, output_channels, 
               kernel_size, strides, scope, padding='SAME'):
    with tf.name_scope(scope):
        weights_shape = kernel_size + [input_channels, output_channels]
        weights = tf.Variable(tf.truncated_normal(shape=weights_shape,
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32),
                                                  name='weights')
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


def fc_layer(input, output_nodes, scope,
             activation=None, seed=None):
    with tf.name_scope(scope):
        shape = int(np.prod(input.get_shape()[1:]))
        flat_input = tf.reshape(input, [-1, shape])
        weights = tf.Variable(tf.truncated_normal(shape=[shape,
                                                         output_nodes],
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32,
                                                  seed=seed),
                                                  name='weights')
        biases = tf.Variable(tf.zeros(shape=[output_nodes]),
                             name='biases')
        act = tf.nn.bias_add(tf.matmul(flat_input, weights), biases, 
                             name='logits')

        if activation is not None:
            act = activation(act, name='activation')

        return act


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, image_width, image_height, image_depth], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')
     
    ##########################
    ### VGG16 Model
    ##########################

    # =========
    # BLOCK 1
    # =========
    conv_layer_1 = conv_layer(input=tf_x,
                              input_channels=3,
                              output_channels=64,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv1')
    
    conv_layer_2 = conv_layer(input=conv_layer_1,
                              input_channels=64,
                              output_channels=64,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv2')    
    
    pool_layer_1 = tf.nn.max_pool(conv_layer_2,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool1') 
    # =========
    # BLOCK 2
    # =========
    conv_layer_3 = conv_layer(input=pool_layer_1,
                              input_channels=64,
                              output_channels=128,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv3')    
    
    conv_layer_4 = conv_layer(input=conv_layer_3,
                              input_channels=128,
                              output_channels=128,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv4')    
    
    pool_layer_2 = tf.nn.max_pool(conv_layer_4,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool2') 
    # =========
    # BLOCK 3
    # =========
    conv_layer_5 = conv_layer(input=pool_layer_2,
                              input_channels=128,
                              output_channels=256,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv5')        
    
    conv_layer_6 = conv_layer(input=conv_layer_5,
                              input_channels=256,
                              output_channels=256,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv6')      
    
    conv_layer_7 = conv_layer(input=conv_layer_6,
                              input_channels=256,
                              output_channels=256,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv7')
    
    pool_layer_3 = tf.nn.max_pool(conv_layer_7,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool3') 
    # =========
    # BLOCK 4
    # =========
    conv_layer_8 = conv_layer(input=pool_layer_3,
                              input_channels=256,
                              output_channels=512,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv8')      
    
    conv_layer_9 = conv_layer(input=conv_layer_8,
                              input_channels=512,
                              output_channels=512,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv9')     
    
    conv_layer_10 = conv_layer(input=conv_layer_9,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv10')   
    
    pool_layer_4 = tf.nn.max_pool(conv_layer_10,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool4') 
    # =========
    # BLOCK 5
    # =========
    conv_layer_11 = conv_layer(input=pool_layer_4,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv11')   
    
    conv_layer_12 = conv_layer(input=conv_layer_11,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv12')   

    conv_layer_13 = conv_layer(input=conv_layer_12,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv13') 
    
    pool_layer_5 = tf.nn.max_pool(conv_layer_12,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool5')     
    # ===========
    # CLASSIFIER
    # ===========
    
    fc_layer_1 = fc_layer(input=pool_layer_5, 
                          output_nodes=4096,
                          activation=tf.nn.relu,
                          scope='fc1')
    
    fc_layer_2 = fc_layer(input=fc_layer_1, 
                          output_nodes=4096,
                          activation=tf.nn.relu,
                          scope='fc2')

    out_layer = fc_layer(input=fc_layer_2, 
                         output_nodes=n_classes,
                         activation=None,
                         scope='output_layer')
    
    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1), 
                                  name='correct_predictions')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # Saver to save session for reuse
    saver = tf.train.Saver()

    
##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        
        avg_cost = 0.
        mbatch_cnt = 0
        for batch_x, batch_y in cifar.load_train_epoch(shuffle=True, batch_size=batch_size):
            
            mbatch_cnt += 1
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y})
            avg_cost += c

            if not mbatch_cnt % print_interval:
                print("Minibatch: %04d | Cost: %.3f" % (mbatch_cnt, c))
                

        # ===================
        # Training Accuracy
        # ===================
        n_predictions, n_correct = 0, 0
        for batch_x, batch_y in cifar.load_train_epoch(batch_size=batch_size):
        
            p = sess.run('correct_predictions:0', 
                         feed_dict={'features:0': batch_x,
                                    'targets:0':  batch_y})
            n_correct += np.sum(p)
            n_predictions += p.shape[0]
        train_acc = n_correct / n_predictions
        
        
        # ===================
        # Validation Accuracy
        # ===================
        #valid_acc = sess.run('accuracy:0', feed_dict={'features:0': X_valid,
        #                                              'targets:0': y_valid})
        # ---------------------------------------
        # workaround for GPUs with <= 4 Gb memory
        n_predictions, n_correct = 0, 0
        indices = np.arange(y_valid.shape[0])
        chunksize = 500
        for start_idx in range(0, indices.shape[0] - chunksize + 1, chunksize):
            index_slice = indices[start_idx:start_idx + chunksize]
            p = sess.run('correct_predictions:0', 
                         feed_dict={'features:0': X_valid[index_slice],
                                    'targets:0': y_valid[index_slice]})
            n_correct += np.sum(p)
            n_predictions += p.shape[0]
        valid_acc = n_correct / n_predictions
        # ---------------------------------------
                                                
        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / mbatch_cnt), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
    
    saver.save(sess, save_path='./convnet-vgg16.ckpt')


# In[5]:


##########################
### RELOAD & TEST
##########################

with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./convnet-vgg16.ckpt')
    
    # test_acc = sess.run('accuracy:0', feed_dict={'features:0': X_test,
    #                                              'targets:0': y_test})
    # ---------------------------------------
    # workaround for GPUs with <= 4 Gb memory
    n_predictions, n_correct = 0, 0
    indices = np.arange(y_test.shape[0])
    chunksize = 500
    for start_idx in range(0, indices.shape[0] - chunksize + 1, chunksize):
        index_slice = indices[start_idx:start_idx + chunksize]
        p = sess.run('correct_predictions:0', 
                     feed_dict={'features:0': X_test[index_slice],
                                'targets:0': y_test[index_slice]})
        n_correct += np.sum(p)
        n_predictions += p.shape[0]
    test_acc = n_correct / n_predictions
    # ---------------------------------------

    print('Test ACC: %.3f' % test_acc)

