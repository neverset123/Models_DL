import numpy as np
import tensorflow as tf
import LeNet5


img_size=28
img_channels=1
num_classes=10
num_imges=1343629
validation_ratio=0.01
test_ratio=0.01

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
epochs=30

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



def LeNet5_run():
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
    train_x, train_y = total_x[:int(num_imges*(1-validation_ratio-test_ratio)), :, :, :], total_y[:int(num_imges*(1-validation_ratio-test_ratio)), :]
    valid_x, valid_y = total_x[int(num_imges*(1-validation_ratio-test_ratio)):int(num_imges*(1-test_ratio)), :, :, :], total_y[int(num_imges*(1-validation_ratio-test_ratio)):int(num_imges*(1-test_ratio)),:]
    test_x, test_y = total_x[int(num_imges*(1-test_ratio)):, :, :, :], total_y[int(num_imges*(1-test_ratio)):,:]

    # Graph
    x = tf.placeholder(tf.float32, [None,
                                    img_size,
                                    img_size,
                                    img_channels],
                                    name='x-input')
    y_ = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

    training_phase = tf.placeholder(tf.bool, None, name='training_phase')
 
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
 
    y = LeNet5.LeNet5(x, training_phase, regularizer, img_channels, num_classes)
 
    global_step = tf.Variable(0, trainable=False)
 
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
 
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               num_imges // BATCH_SIZE,
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
            NumOfBatch = train_y.shape[0] // BATCH_SIZE
            for i in range(NumOfBatch):  
                train_x_batch = train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :,:]
                train_y_batch = train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                _, loss_valuue, step, accuracy_train = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: train_x_batch, y_: train_y_batch, training_phase: True})

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
 
 
if __name__ == '__main__':
    LeNet5_run()