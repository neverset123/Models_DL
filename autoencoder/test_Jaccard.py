
import tensorflow as tf

def jaccard_distance(prediction, labels):
    Nominator=tf.reduce_sum(tf.minimum(prediction, labels))
    Denominator=tf.reduce_sum(tf.maximum(prediction, labels))
    return 1-Nominator/Denominator

X=tf.constant([[1.0, 2.0], [3.0, 4.0]])

Y=tf.constant([[1.0, 1.0], [0.0, 1.0]])


sess = tf.Session()
print(sess.run(jaccard_distance(X,Y)))