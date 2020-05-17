import tensorflow as tf
from tensorflow.contrib.layers import flatten

num_classes=10

def conv_layer(input, filter, kernel, strides=1, padding=None, regularizer=None, name="conv"):
    with tf.name_scope(name):
        network = tf.layers.conv2d(inputs=input, 
                                    filters=filter, 
                                    kernel_size=kernel, 
                                    strides=strides, 
                                    kernel_regularizer=regularizer, 
                                    kernel_initializer=tf.initializers.he_normal(),
                                    padding=padding)
        return network

def FRNLayer(x, eps=1e-6):
    channels=x.get_shape().as_list()[-1]

    tau_shape = (1,1,1,channels)
    tau = tf.get_variable(initializer=tf.zeros(shape=tau_shape,
                                                dtype=tf.float32),
                                                trainable=True,
                                                name='tau')
    beta_shape = (1,1,1,channels)
    beta = tf.get_variable(initializer=tf.zeros(shape=beta_shape,
                                            dtype=tf.float32),
                                            name='beta')
    gamma_shape = (1,1,1,channels)
    gamma= tf.get_variable(initializer=tf.ones(shape=gamma_shape,
                                                dtype=tf.float32),
                                                name='gamma')
                                                                                               
    nu2=tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True)
    x=x*tf.rsqrt(nu2+tf.abs(eps))
    return tf.maximum(gamma*x+beta, tau)

class NIN():
    def __init__(self, x, training, drop_rate, regularizer):
        self.training = training
        self.dropout_rate=drop_rate
        self.regularizer=regularizer
        self.model = self.Classifier(x)
        self.model_FRN =self.Classifier_FRN(x)

    def Classifier(self, input_x):
        in_channel=input_x.get_shape().as_list()[-1]
        #1st block
        print(input_x)
        x=conv_layer(input_x, filter=192, kernel=[5,5], strides=1, padding='SAME', name='conv1_0')
        x=tf.nn.relu(x)
        x=conv_layer(x, filter=160, kernel=[1,1], strides=1, padding='SAME', name='conv1_1')
        x=tf.nn.relu(x)
        x=conv_layer(x, filter=96, kernel=[1,1], strides=1, padding='SAME', name='conv1_2')
        x=tf.nn.relu(x)
        x=tf.layers.max_pooling2d(x, pool_size=[3,3], strides=2, padding='SAME', name='max_pool')
        x=tf.layers.dropout(x, rate=self.dropout_rate, training=self.training)

        #2nd block
        x=conv_layer(x, filter=192, kernel=[5,5], strides=1, padding='SAME', name='conv2_0')
        x=tf.nn.relu(x)
        x=conv_layer(x, filter=192, kernel=[1,1], strides=1, padding='SAME', name='conv2_1')
        x=tf.nn.relu(x)
        x=conv_layer(x, filter=192, kernel=[1,1], strides=1, padding='SAME', name='conv2_2')
        x=tf.nn.relu(x)
        x=tf.layers.max_pooling2d(x, pool_size=[3,3], strides=2, padding='SAME')
        x=tf.layers.dropout(x, rate=self.dropout_rate, training=self.training)

        #3rd block
        x=conv_layer(x, filter=192, kernel=[3,3], strides=1, padding='SAME', name='conv3_0')
        x=tf.nn.relu(x)
        x=conv_layer(x, filter=192, kernel=[1,1], strides=1, padding='SAME', name='conv3_1')
        x=tf.nn.relu(x)
        x=conv_layer(x, filter=num_classes, kernel=[1,1], strides=1, padding='SAME', name='conv3_2')
        x=tf.nn.relu(x)

        #GAP
        x=tf.reduce_mean(x, [1,2], name='GAP', keep_dims=True)
        x=tf.squeeze(x, [1,2], name='GAP')
        print(x)

        return x

    def Classifier_FRN(self, input_x):
        #1st block
        with tf.variable_scope('conv1_0', reuse=tf.AUTO_REUSE  ):
            x=conv_layer(input_x, filter=192, kernel=[5,5], strides=1, padding='SAME', name='conv1_0')
            x=FRNLayer(x)
            print(x)
            x=tf.nn.relu(x)
        with tf.variable_scope('conv1_1', reuse=tf.AUTO_REUSE  ):    
            x=conv_layer(x, filter=160, kernel=[1,1], strides=1, padding='SAME', name='conv1_1')
            x=FRNLayer(x)
            print(x)
            x=tf.nn.relu(x)
        with tf.variable_scope('conv1_2', reuse=tf.AUTO_REUSE  ):  
            x=conv_layer(x, filter=96, kernel=[1,1], strides=1, padding='SAME', name='conv1_2')
            x=FRNLayer(x)
            print(x)   
            x=tf.nn.relu(x)
        x=tf.layers.max_pooling2d(x, pool_size=[3,3], strides=2, padding='SAME', name='max_pool')
        x=tf.layers.dropout(x, rate=self.dropout_rate, training=self.training)

        #2nd block
        with tf.variable_scope('conv2_0', reuse=tf.AUTO_REUSE  ):  
            x=conv_layer(x, filter=192, kernel=[5,5], strides=1, padding='SAME', name='conv2_0')
            x=FRNLayer(x)
            print(x)
            x=tf.nn.relu(x)
        with tf.variable_scope('conv2_1', reuse=tf.AUTO_REUSE  ):      
            x=conv_layer(x, filter=192, kernel=[1,1], strides=1, padding='SAME', name='conv2_1')
            x=FRNLayer(x)
            print(x) 
            x=tf.nn.relu(x)
        with tf.variable_scope('conv2_2', reuse=tf.AUTO_REUSE  ):     
            x=conv_layer(x, filter=192, kernel=[1,1], strides=1, padding='SAME', name='conv2_2')
            x=FRNLayer(x)
            print(x) 
            x=tf.nn.relu(x)
        x=tf.layers.max_pooling2d(x, pool_size=[3,3], strides=2, padding='SAME')
        x=tf.layers.dropout(x, rate=self.dropout_rate, training=self.training)

        #3rd block
        with tf.variable_scope('conv3_0', reuse=tf.AUTO_REUSE  ): 
            x=conv_layer(x, filter=192, kernel=[3,3], strides=1, padding='SAME', name='conv3_0')
            x=FRNLayer(x)
            print(x) 
            x=tf.nn.relu(x)
        with tf.variable_scope('conv3_1', reuse=tf.AUTO_REUSE  ):
            x=conv_layer(x, filter=192, kernel=[1,1], strides=1, padding='SAME', name='conv3_1')
            x=FRNLayer(x)
            print(x)
            x=tf.nn.relu(x)
        with tf.variable_scope('conv3_2', reuse=tf.AUTO_REUSE  ):
            x=conv_layer(x, filter=num_classes, kernel=[1,1], strides=1, padding='SAME', name='conv3_2')
            x=FRNLayer(x)
            print(x)
            x=tf.nn.relu(x)

        #GAP
        x=tf.reduce_mean(x, [1,2], name='GAP', keep_dims=True)
        x=tf.squeeze(x, [1,2], name='GAP')
        print(x)

        return x
