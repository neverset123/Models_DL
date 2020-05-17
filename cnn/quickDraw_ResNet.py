# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import ResNet
import cifar_LoadData
from pathlib import Path
import math

validation_ratio=0.01 #0.2
test_ratio=0.01
img_size=28 #cifar_LoadData.img_size
img_channels=1 #cifar_LoadData.num_channels
num_classes=10 #cifar_LoadData.num_classes
num_images=1343629 #cifar_LoadData._num_images_train
num_imges_train= num_images*(1-validation_ratio-test_ratio)#60000 #mnist.train.num_examples
num_imges_valid=num_images*validation_ratio #10000
num_imges_test=num_images*test_ratio #cifar_LoadData._num_images_test

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001 #0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
epochs=10 #30

momentum=0.9
epsilon = 1e-8
num_layers=34
MODEL_NAME = "model.ckpt"
save_dir='./'

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the int array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=np.uint8)[class_numbers]


def load_data(filename_list):
    input_x=np.empty((0,img_size*img_size*img_channels), np.uint8)
    input_y=np.empty((0,1), np.uint8)
    j=0
    for i in filename_list:
        filename='./data/quickDraw/'+str(i)+'.npy'
        x_batch=np.load(filename)
        y_batch=j*np.ones((x_batch.shape[0],1), np.uint8)
        input_x=np.concatenate((input_x, x_batch))
        input_y=np.concatenate((input_y, y_batch))
        j+=1
    #print(one_hot_encoded(input_y, num_classes).shape)    
    #print(input_x.shape)
    return input_x, input_y, one_hot_encoded(input_y, num_classes)


def DenseNet_run():

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    test_loss, test_acc = [], []
    # load data
    #Dataset
    class_name_file='class_name.txt'
    class_name_path='./data/quickDraw/'+class_name_file
    with open(class_name_path) as f:
        file_list=f.read().splitlines()
    total_x, _, total_y=load_data(file_list)
    total_x=np.reshape(total_x, (-1, img_size, img_size, img_channels))
    total_y=np.reshape(total_y, (-1, num_classes))
    
    ## Shuffling & train/validation split
    shuffle_idx = np.arange(total_y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    total_x, total_y = total_x[shuffle_idx], total_y[shuffle_idx]
    train_x, train_y = total_x[:int(num_images*(1-validation_ratio-test_ratio)), :, :, :], total_y[:int(num_images*(1-validation_ratio-test_ratio)), :]
    valid_x, valid_y = total_x[int(num_images*(1-validation_ratio-test_ratio)):int(num_images*(1-test_ratio)), :, :, :], total_y[int(num_images*(1-validation_ratio-test_ratio)):int(num_images*(1-test_ratio)),:]
    test_x, test_y = total_x[int(num_images*(1-test_ratio)):, :, :, :], total_y[int(num_images*(1-test_ratio)):,:]

    #reset graph
    tf.reset_default_graph()
    # Graph
    x = tf.placeholder(tf.float32, [None,
                                    img_size,
                                    img_size,
                                    img_channels],
                                    name='x-input')
    y_ = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

    training_phase = tf.placeholder(tf.bool, None, name='training_phase')
    keep_prob =tf.placeholder(tf.float32, None, name='keep_prob')
 
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = ResNet.ResNet(x, num_layers, training_phase, num_classes).model
 
    global_step = tf.Variable(0, trainable=False)
 
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # labels is the label index, not the values
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #loss = cross_entropy_mean+tf.losses.get_regularization_loss()
    #loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean
 
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               num_imges_train // BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    with tf.control_dependencies(update_ops):
        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(loss, global_step=global_step)
        #train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True).minimize(loss, global_step=global_step)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #minimize is an combi-operation of compute gradients and apply gradients
        #grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
        #train_step=optimizer.apply_gradients(grads, global_step=global_step)

    # Prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #load pretrained model
        ckpt = tf.train.get_checkpoint_state('/model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else: 
            sess.run(tf.global_variables_initializer())

        #tf.global_variables_initializer().run()
        
        #log training process
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', sess.graph)
        #start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(epochs):

            NumOfBatchTrain = math.ceil(num_imges_train // BATCH_SIZE)
            for i in range(NumOfBatchTrain): 
                if i< NumOfBatchTrain-1:
                    train_x_batch = train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                    train_y_batch = train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                else:
                    train_x_batch = train_x[i*BATCH_SIZE:, :, :,:]
                    train_y_batch = train_y[i*BATCH_SIZE:,:]

                #train_x_batch, train_y_batch = mnist.train.next_batch(BATCH_SIZE)
                #train_x_batch=np.reshape(train_x_batch, (-1, img_size, img_size, img_channels))

                _, loss_train_batch, step, acc_train_batch = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: train_x_batch, 
                                                                                                                    y_: train_y_batch, 
                                                                                                                    training_phase: True, 
                                                                                                                    keep_prob: 0.5})
                train_loss.append(loss_train_batch)
                train_acc.append(acc_train_batch)
                if (step-1)%100==0:
                    print("training steps: %d , training loss: %g, train accuracy: %g" % (step, loss_train_batch, acc_train_batch))

            #validation in batch
            NumOfBatchValid= math.ceil(num_imges_valid // BATCH_SIZE)
            _valid_loss, _valid_acc = [], []

            for i in range(NumOfBatchValid):
                #valid_x_batch, valid_y_batch = mnist.test.next_batch(BATCH_SIZE)
                #valid_x_batch=np.reshape(valid_x_batch, (-1, img_size, img_size, img_channels))
                if i< NumOfBatchValid-1:
                    valid_x_batch = valid_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                    valid_y_batch = valid_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                else:
                    valid_x_batch = valid_x[i*BATCH_SIZE:, :, :,:]
                    valid_y_batch = valid_y[i*BATCH_SIZE:,:]                    
                loss_val_batch, accuracy_val_batch= sess.run([loss, accuracy], 
                                    feed_dict={x: valid_x_batch, 
                                    y_: valid_y_batch, 
                                    training_phase: False,
                                    keep_prob: 1.0})
                _valid_loss.append(loss_val_batch)
                _valid_acc.append(accuracy_val_batch) 
            valid_loss.append(np.mean(_valid_loss))
            valid_acc.append(np.mean(_valid_acc))
            print("validation loss: %g, validation accuracy: %g" % (valid_loss[-1], valid_acc[-1]))
            if valid_acc[-1]>0.5:
                saver.save(sess, os.path.join(save_dir, MODEL_NAME), global_step=global_step)

            # test
            NumOfBatchTest = math.ceil(num_imges_test // BATCH_SIZE)
            _test_loss, _test_acc = [], []

            for i in range(NumOfBatchTest):
                if i<NumOfBatchTest-1:
                    test_x_batch = test_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                    test_y_batch = test_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                else:
                    test_x_batch = test_x[i*BATCH_SIZE:, :, :,:]
                    test_y_batch = test_y[i*BATCH_SIZE:,:]
                loss_val_batch, accuracy_val_batch= sess.run([loss, accuracy], 
                                    feed_dict={x: test_x_batch, 
                                    y_: test_y_batch, 
                                    training_phase: False,
                                    keep_prob: 1.0})
                _test_loss.append(loss_val_batch)
                _test_acc.append(accuracy_val_batch) 
            test_loss.append(np.mean(_test_loss))
            test_acc.append(np.mean(_test_acc))
            print("test loss: %g, test accuracy: %g" % (test_loss[-1], test_acc[-1]))

        coord.request_stop()
        coord.join(threads)

        #save loss and accuracy data 
        Path(os.path.join(save_dir, 'accuracy_loss')).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'train_loss'), train_loss)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'train_acc'), train_acc)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_loss'), valid_loss)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_acc'), valid_acc)                                    
 
 
if __name__ == '__main__':
    DenseNet_run()
