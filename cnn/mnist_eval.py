# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
import mnist_inference
import mnist_train
 
import numpy as np

EVAL_INTERVAL_SECS = 10
 
 
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
        x = tf.placeholder(tf.float32, [mnist.validation.num_examples,
                                        cnn.mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [mnist.validation.num_examples, mnist_inference.OUTPUT_NODE],
                            name="y-input")
 
        reshaped_xs = np.reshape(mnist.validation.images,
                                 [mnist.validation.num_examples,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.NUM_CHANNELS])
 
        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}
 
        y = mnist_inference.inference(x, False, None)
 

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        #print(variables_to_restore)
        # {'layer3-conv2/bias/ExponentialMovingAverage': <tf.Variable 'layer3-conv2/bias:0' shape=(64,) dtype=float32_ref>,
        # 'layer1-conv1/bias/ExponentialMovingAverage': <tf.Variable 'layer1-conv1/bias:0' shape=(32,) dtype=float32_ref>,
        # 'layer6-fc2/bias/ExponentialMovingAverage': <tf.Variable 'layer6-fc2/bias:0' shape=(10,) dtype=float32_ref>,
        # 'layer3-conv2/weight/ExponentialMovingAverage': <tf.Variable 'layer3-conv2/weight:0' shape=(5, 5, 32, 64) dtype=float32_ref>,
        # 'layer6-fc2/weight/ExponentialMovingAverage': <tf.Variable 'layer6-fc2/weight:0' shape=(512, 10) dtype=float32_ref>,
        # 'layer1-conv1/weight/ExponentialMovingAverage': <tf.Variable 'layer1-conv1/weight:0' shape=(5, 5, 1, 32) dtype=float32_ref>,
        # 'layer5-fc1/bias/ExponentialMovingAverage': <tf.Variable 'layer5-fc1/bias:0' shape=(512,) dtype=float32_ref>,
        # 'layer5-fc1/weight/ExponentialMovingAverage': <tf.Variable 'layer5-fc1/weight:0' shape=(3136, 512) dtype=float32_ref>}
 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        while True:
            with tf.Session(config=config) as sess:

                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                # print(ckpt)
                # model_checkpoint_path: "/path/to/model/cnn/model.ckpt-4001"
                # all_model_checkpoint_paths: "/path/to/model/cnn/model.ckpt-1"
                # all_model_checkpoint_paths: "/path/to/model/cnn/model.ckpt-1001"
                # all_model_checkpoint_paths: "/path/to/model/cnn/model.ckpt-2001"
                # all_model_checkpoint_paths: "/path/to/model/cnn/model.ckpt-3001"
                # all_model_checkpoint_paths: "/path/to/model/cnn/model.ckpt-4001"
                if ckpt and ckpt.model_checkpoint_path:

                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
 

                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
 
                    print("After %s training stpe(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)
 
 
def main(argv=None):
    mnist = input_data.read_data_sets("./", one_hot=True)
    evaluate(mnist)
 
 
if __name__ == '__main__':
    tf.app.run()
