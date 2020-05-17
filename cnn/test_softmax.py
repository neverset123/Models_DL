import tensorflow as tf
import numpy as np

Truth = np.array([0,0,1,0])
Pred_logits = np.array([3.5,2.1,7.89,4.4])

loss = tf.nn.softmax_cross_entropy_with_logits(labels=Truth,logits=Pred_logits)
loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Truth,logits=Pred_logits)
loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(Truth),logits=Pred_logits)

with tf.Session() as sess:
    print(sess.run(loss))
    print(sess.run(loss2))
    print(sess.run(loss3))
