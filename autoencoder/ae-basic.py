
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

##########################
### WRAPPER FUNCTIONS
##########################

def fully_connected(input_tensor, output_nodes,
                    activation=None, seed=None,
                    name='fully_connected'):

    with tf.name_scope(name):
        input_nodes = input_tensor.get_shape().as_list()[1]
        weights = tf.Variable(tf.truncated_normal(shape=(input_nodes,
                                                         output_nodes),
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[output_nodes]), name='biases')

        act = tf.matmul(input_tensor, weights) + biases
        if activation is not None:
            act = activation(act)
        return act


##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", validation_size=0) #read in all datas as train data


##########################
### SETTINGS
##########################


# Hyperparameters
learning_rate = 0.01
training_epochs = 5
batch_size = 128

# Architecture
hidden_size = 32
input_size = 784
image_width = 28

# Other
print_interval = 200
random_seed = 123


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)

    # Input data
    input_layer = tf.placeholder(tf.float32, [None, input_size],
                                 name='input')

    ###########
    # Encoder
    ###########
    
    hidden_layer = fully_connected(input_layer, hidden_size, 
                                   activation=tf.nn.relu, 
                                   name='encoding')
    
    ###########
    # Decoder
    ###########
    
    logits = fully_connected(hidden_layer, input_size, 
                             activation=None, name='logits')
    # note MNIST pixels are normalized to 0-1 range
    out_layer = tf.nn.sigmoid(logits, name='decoding') 
    
    ##################
    # Loss & Optimizer
    ##################
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=input_layer, logits=logits), name='cost')
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Saver to save session for reuse
    saver = tf.train.Saver()

import numpy as np

##########################
### TRAINING & EVALUATION
##########################
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed) # random seed for mnist iterator
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size


        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], 
                            feed_dict={'input:0': batch_x})
            avg_cost += c
            
            if not i % print_interval:
                print("Minibatch: %03d | Cost:    %.3f" % (i + 1, c))
    
        print("Epoch:     %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)))
    
    saver.save(sess, save_path='./autoencoder.ckpt')


##########################
### VISUALIZATION
##########################

n_images = 15

fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                         sharex=True, sharey=True, figsize=(20, 2.5))
test_images = mnist.test.images[:n_images]

with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./autoencoder.ckpt')
    decoded = sess.run('decoding:0', feed_dict={'input:0': test_images})

for i in range(n_images):
    for ax, img in zip(axes, [test_images, decoded]):
        ax[i].imshow(img[i].reshape((image_width, image_width)), cmap='binary')

