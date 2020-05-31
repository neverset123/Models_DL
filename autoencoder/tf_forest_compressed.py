import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/mnist", one_hot=False)

# Parameters
learning_rate = 0.001
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000
random_seed=123

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)
    # Input and Target data
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # For random forest, labels must be integers (the class id)
    Y = tf.placeholder(tf.int32, shape=[None])
    X_resized=tf.reshape(X, (-1, 28, 28, 1))

    #adding encoder 
    # 28x28x1 => 28x28x8
    conv1 = tf.layers.conv2d(X_resized, filters=8, kernel_size=(3, 3),
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

    encode_resized=tf.reshape(encode, (-1, 7*7*4))

    num_features=7*7*4
    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                        num_features=num_features,
                                        num_trees=num_trees,
                                        max_nodes=max_nodes).fill()

    # Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # Get training graph and loss
    train_op_rf = forest_graph.training_graph(encode_resized, Y)
    #loss_op = forest_graph.training_loss(encode_resized, Y)
    pred_logits, _, _ = forest_graph.inference_graph(encode_resized)
    loss_rf = forest_graph.training_loss(encode_resized, Y)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=pred_logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss_op=cross_entropy_mean+loss_rf
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies([train_op_rf]):
         train_op = tf.no_op(name="train")

    # Measure the accuracy
    correct_prediction = tf.equal(tf.argmax(pred_logits, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables (i.e. assign their default value) and forest resources
    init_vars = tf.group(tf.global_variables_initializer(),
        resources.initialize_resources(resources.shared_resources()))

with tf.Session(graph=g) as sess:
    # Run the initializer
    sess.run(init_vars)

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

    # Test Model
    test_x, test_y = mnist.test.images, mnist.test.labels
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))