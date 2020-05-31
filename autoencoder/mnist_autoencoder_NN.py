import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from autoencode_NN import autoencoder_NN

#input parameter
img_size=28
img_channels=1
num_classes=10
num_imges_train= 60000 #mnist.train.num_examples
num_imges_valid=10000

# Hyperparameters
LEARNING_RATE_BASE = 0.001 #0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
epochs = 30
BATCH_SIZE = 128

# Architecture
image_width = 28

# Other
print_interval = 200
random_seed = 123

MODEL_NAME = "model.ckpt"
save_dir='./'


def run_ae_NN():

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    mnist = input_data.read_data_sets("../data/mnist", one_hot=False)
    
    #reset graph
    tf.reset_default_graph()

    #run ae_NN
    tf_x = tf.placeholder(tf.float32, [None, image_width, image_width, 1], name='inputs')
    input_layer = tf.reshape(tf_x, shape=[-1, image_width, image_width, 1])
    training_phase = tf.placeholder(tf.bool, None, name='training_phase')
    keep_prob =tf.placeholder(tf.float32, None, name='keep_prob')

    logits=autoencoder_NN(input_layer)

    #setting training step
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               num_imges_train // BATCH_SIZE,
                                               LEARNING_RATE_DECAY)


    #loss and accuracy
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_layer, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    train_step=optimizer.apply_gradients(grads, global_step=global_step)   

    # Prediction(in Autoencoder there is no accuracy indicator, only loss should be used)
    correct_prediction = tf.equal(logits, input_layer)
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
                #train_x_batch = train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                #train_y_batch = train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                train_x_batch, train_y_batch = mnist.train.next_batch(BATCH_SIZE)
                train_x_batch=np.reshape(train_x_batch, (-1, img_size, img_size, img_channels))

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
                valid_x_batch, valid_y_batch = mnist.test.next_batch(BATCH_SIZE)
                valid_x_batch=np.reshape(valid_x_batch, (-1, img_size, img_size, img_channels))
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
            '''test_acc = sess.run('accuracy:0', 
                             feed_dict={'x-input:0': test_x,
                                       'y-input:0': test_y,
                                       'training_phase:0': False})
            print("test accuracy: %g" % (test_acc))'''

        coord.request_stop()
        coord.join(threads)

        #save loss and accuracy data 
        np.save(os.path.join(save_dir, 'accuracy_loss', 'train_loss'), train_loss)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'train_acc'), train_acc)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_loss'), valid_loss)
        np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_acc'), valid_acc)      

if __name__ == "__main__":
    run_ae_NN()
