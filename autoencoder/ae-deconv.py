import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("../data/mnist", validation_size=0)

# Hyperparameters
learning_rate = 0.001
training_epochs = 5
batch_size = 128

# Architecture
hidden_size = 16
input_size = 784
image_width = 28

# Other
print_interval = 200
random_seed = 123

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, input_size], name='inputs')
    input_layer = tf.reshape(tf_x, shape=[-1, image_width, image_width, 1])

    ###########
    # Encoder
    ###########
    
    # 28x28x1 => 28x28x8
    conv1 = tf.layers.conv2d(input_layer, filters=8, kernel_size=(3, 3),
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    # 28x28x8 => 14x14x8
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), 
                                       strides=(2, 2), padding='same')
    
    # 14x14x8 => 14x14x4
    conv2 = tf.layers.conv2d(maxpool1, filters=4, kernel_size=(3, 3), 
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    
    # 14x14x4 => 7x7x4
    encode = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), 
                                     strides=(2, 2), padding='same', 
                                     name='encoding')

    ###########
    # Decoder
    ###########
    
    # 7x7x4 => 14x14x8
    deconv1 = tf.layers.conv2d_transpose(encode, filters=8, 
                                         kernel_size=(3, 3), strides=(2, 2), 
                                         padding='same',
                                         activation=tf.nn.relu)
    
    
    # 14x14x8 => 28x28x8
    deconv2 = tf.layers.conv2d_transpose(deconv1, filters=8, 
                                         kernel_size=(3, 3), strides=(2, 2), 
                                         padding='same',
                                         activation=tf.nn.relu)
    
    # 28x28x8 => 28x28x1
    logits = tf.layers.conv2d(deconv2, filters=1, kernel_size=(3,3), 
                              strides=(1, 1), padding='same', 
                              activation=None)
    
    decode = tf.nn.sigmoid(logits, name='decoding')

    ##################
    # Loss & Optimizer
    ##################
    
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_layer, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, name='train')   

    # Saver to save session for reuse
    saver = tf.train.Saver()


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed) # random seed for mnist iterator
    for epoch in range(training_epochs):
        total_loss = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train, loss], feed_dict={'inputs:0': batch_x})

            total_loss += c

            if not i % print_interval:
                print("Minibatch: %03d | loss:    %.3f" % (i + 1, c))

        print("Epoch:     %03d | Avg loss %.3f" % (epoch + 1, total_loss / (i + 1)))
    
    saver.save(sess, save_path='./autoencoder.ckpt')
