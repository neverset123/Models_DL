import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from autoencode_NN import autoencoder_NN
from DataLoader_celeba import train_data, valid_data, test_data

#input parameter
img_size=32
img_channels=3
num_classes=2
num_imges_train= 162770
num_imges_valid=19867
num_imges_test=19962

# Hyperparameters
LEARNING_RATE_BASE = 0.001 #0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
epochs = 30
BATCH_SIZE = 128

# Other
random_seed = 123

MODEL_NAME = "model.ckpt"
save_dir='./'


def run_ae_NN():

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    test_loss, test_acc = [], []
    # load data
    #Dataset 
    train_x, _, train_y=train_data(shuffle=True)   
    valid_x, _, valid_y=valid_data(shuffle=True)   
    test_x, _, test_y=test_data(shuffle=True)
    ## Shuffling & train/validation split
    #shuffle_idx = np.arange(num_imges_train)
    #shuffle_rng = np.random.RandomState(random_seed)
    #shuffle_rng.shuffle(shuffle_idx)
    #total_x, total_y = total_x[shuffle_idx], total_y[shuffle_idx]
    #train_x, train_y = total_x[:int(num_images*(1-validation_ratio)), :, :, :], total_y[:int(num_images*(1-validation_ratio)), :]
    #valid_x, valid_y = total_x[int(num_images*(1-validation_ratio)):, :, :, :], total_y[int(num_images*(1-validation_ratio)):, :]
    
    #reset graph
    tf.reset_default_graph()

    #run ae_NN
    tf_x = tf.placeholder(tf.float32, [None, img_size, img_size, img_channels], name='inputs')
    training_phase = tf.placeholder(tf.bool, None, name='training_phase')
    keep_prob =tf.placeholder(tf.float32, None, name='keep_prob')
    #autoencoder
    logits=autoencoder_NN(tf_x)

    #setting training step
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               num_imges_train // BATCH_SIZE,
                                               LEARNING_RATE_DECAY)


    #loss and accuracy
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_x, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    train_step=optimizer.apply_gradients(grads, global_step=global_step)   

    # Prediction(in Autoencoder there is no accuracy indicator, only loss should be used)
    correct_prediction = tf.equal(logits, tf_x)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    #save session for reuse
    saver = tf.train.Saver()

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        np.random.seed(random_seed) # random seed for mnist iterator
        #start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(epochs):
            NumOfBatchTrain = num_imges_train// BATCH_SIZE
            for i in range(NumOfBatchTrain):  
                train_x_batch = train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                train_y_batch = train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                #train_x_batch, train_y_batch = mnist.train.next_batch(BATCH_SIZE)
                #train_x_batch=np.reshape(train_x_batch, (-1, img_size, img_size, img_channels))

                _, loss_value_batch, step, acc_value_batch = sess.run([train_op, loss, global_step, accuracy], feed_dict={tf_x: train_x_batch, 
                                                                                                                    training_phase: True, 
                                                                                                                    keep_prob: 0.5})
                train_loss.append(loss_value_batch)
                train_acc.append(acc_value_batch)
                if (step-1)%100==0:
                    print("training steps: %d , training loss: %g, train accuracy: %g" % (step, loss_value_batch, acc_value_batch))

            #validation in batch
            NumOfBatchValid= num_imges_valid// BATCH_SIZE
            _valid_loss, _valid_acc = [], []

            for i in range(NumOfBatchValid):
                #valid_x_batch, valid_y_batch = mnist.test.next_batch(BATCH_SIZE)
                #valid_x_batch=np.reshape(valid_x_batch, (-1, img_size, img_size, img_channels))
                valid_x_batch = valid_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                valid_y_batch = valid_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                loss_val_batch, accuracy_val_batch= sess.run([loss, accuracy], 
                                    feed_dict={tf_x: valid_x_batch, 
                                    training_phase: False,
                                    keep_prob: 1.0})
                _valid_loss.append(loss_val_batch)
                _valid_acc.append(accuracy_val_batch) 
            valid_loss.append(np.mean(_valid_loss))
            valid_acc.append(np.mean(_valid_acc))
            print("validation accuracy: %g" % (valid_acc[-1]))
            if valid_acc[-1]>0.5:
                saver.save(sess, os.path.join(save_dir, MODEL_NAME), global_step=global_step)

            # test
           # test
            NumOfBatchTest = int(num_imges_test) // BATCH_SIZE
            _test_loss, _test_acc = [], []

            for i in range(NumOfBatchTest):
                test_x_batch = test_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                test_y_batch = test_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                loss_val_batch, accuracy_val_batch= sess.run([loss, accuracy], 
                                    feed_dict={tf_x: test_x_batch, 
                                    training_phase: False,
                                    keep_prob: 1.0})
                _test_loss.append(loss_val_batch)
                _test_acc.append(accuracy_val_batch) 
            test_loss.append(np.mean(_test_loss))
            test_acc.append(np.mean(_test_acc))
            print("test accuracy: %g" % (test_acc[-1]))

        coord.request_stop()
        coord.join(threads)

        #save loss and accuracy data 
        np.save(os.path.join(save_dir, 'accuracy_loss', 'train_loss'), train_loss)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'train_acc'), train_acc)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_loss'), valid_loss)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_acc'), valid_acc)      

if __name__ == "__main__":
    run_ae_NN()
