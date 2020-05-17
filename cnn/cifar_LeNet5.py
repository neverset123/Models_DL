# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import cifar_LoadData
import numpy as np
import LeNet5

IMAGE_SIZE = cifar_LoadData.img_size
OUTPUT_NODE = cifar_LoadData.num_classes
NUM_CHANNELS = cifar_LoadData.num_channels
NUM_LABELS =cifar_LoadData.num_classes

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
epochs=30
validation_ratio=0.2
 
MODEL_SAVE_PATH = "./"
MODEL_NAME = "model.ckpt"

###################################################
 
def LeNet5_run():

    #Dataset
    total_x, _, total_y=cifar_LoadData.load_training_data()
    test_x, _, test_y=cifar_LoadData.load_test_data()
    #total_y=total_y.astype(np.int)
    #test_y=test_y.astype(np.int)
    
    ## Shuffling & train/validation split
    shuffle_idx = np.arange(total_y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    total_x, total_y = total_x[shuffle_idx], total_y[shuffle_idx]
    train_x, train_y = total_x[:int(cifar_LoadData._num_images_train*(1-validation_ratio)), :, :, :], total_y[:int(cifar_LoadData._num_images_train*(1-validation_ratio)), :]
    valid_x, valid_y = total_x[int(cifar_LoadData._num_images_train*(1-validation_ratio)):, :, :, :], total_y[int(cifar_LoadData._num_images_train*(1-validation_ratio)):, :]

    # Graph
    x = tf.placeholder(tf.float32, [None,
                                    IMAGE_SIZE,
                                    IMAGE_SIZE,
                                    NUM_CHANNELS],
                                    name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    training_phase = tf.placeholder(tf.bool, None, name='training_phase')
 
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
 
    y = LeNet5.LeNet5(x, training_phase, regularizer, NUM_CHANNELS, NUM_LABELS)
 
    global_step = tf.Variable(0, trainable=False)
 
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
 
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               cifar_LoadData._num_images_train // BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
 
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for epoch in range(epochs):
            NumOfBatch = int(cifar_LoadData._num_images_train*(1-validation_ratio)) // BATCH_SIZE
            for i in range(NumOfBatch):
                xs = train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                ys = train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
    
                _, loss_valuue, step, accuracy_train = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: xs, y_: ys, training_phase: True})

                if (step-1)%1000==0:
                    print("training steps: %d , training loss: %g, train accuracy: %g" % (step, loss_valuue, accuracy_train))

            #validation
            accuracy_val = sess.run('accuracy:0', 
                                feed_dict={x: valid_x, 
                                y_: valid_y, 
                                training_phase: False})
            print("validation accuracy: %g" % (accuracy_val))
            if accuracy_val>0.5:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

            # test
            test_acc = sess.run('accuracy:0', 
                             feed_dict={'x-input:0': test_x,
                                       'y-input:0': test_y,
                                       'training_phase:0': False})
            print("test accuracy: %g" % (test_acc))
 
 
def main(argv=None):
    LeNet5_run()
 
 
if __name__ == '__main__':
    tf.app.run()