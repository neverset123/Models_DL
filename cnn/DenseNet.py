import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

class_num = 10

def conv_layer(input, filter, kernel, stride=1, regularizer=None, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, 
                                    filters=filter, 
                                    kernel_size=kernel, 
                                    strides=stride, 
                                    kernel_regularizer=regularizer, 
                                    kernel_initializer=tf.initializers.he_normal(),
                                    padding='SAME')
        return network

def Global_Average_Pooling(x):
    return tf.reduce_mean(x, axis=[1,2])

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x, regularizer=None) :
    return tf.layers.dense(inputs=x, kernel_regularizer=regularizer, kernel_initializer=tf.initializers.he_normal(), units=class_num, name='linear')



class DenseNet():
    def __init__(self, x, num_layers, training, drop_rate, regularizer=None):
        self.nb_blocks = 4
        self.dense_n=num_layers
        if self.dense_n < 161 :
            self.filters = 32
        else :
            self.filters = 48
        self.training = training
        self.dropout_rate=drop_rate
        self.regularizer=regularizer
        self.model = self.Dense_net(x)

    def get_denseblock_list(self, dense_n) :
        x = []

        if dense_n == 121 :
            x = [6, 12, 24, 16]

        elif dense_n == 169 :
            x = [6, 12, 32, 32]

        elif dense_n == 201 :
            x = [6, 12, 48, 32]

        elif dense_n == 161 :
            x = [6, 12, 36, 24]

        else:
            raise ValueError('no corresponding option')

        return x


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            #x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x=tf.layers.batch_normalization(x, training=self.training)
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], regularizer=self.regularizer, layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            #x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x=tf.layers.batch_normalization(x, training=self.training)
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], regularizer=self.regularizer, layer_name=scope+'_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            #x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x=tf.layers.batch_normalization(x, training=self.training)
            x = Relu(x)
            in_channel = x.get_shape().as_list()[-1]
            x = conv_layer(x, filter=in_channel*0.5, kernel=[1,1], regularizer=self.regularizer, layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            #print(input_x)
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            #print(x)

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                #print(x)
                layers_concat.append(x)

            x = Concatenation(layers_concat) #the feature maps are concatenated

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, regularizer=self.regularizer, layer_name='conv0')
        #x = conv_layer(input_x, filter=2 * self.filters, kernel=[3,3], stride=1, regularizer=self.regularizer, layer_name='conv0')
        #x = Max_Pooling(x, pool_size=[3,3], stride=2)

        denseblock_list=self.get_denseblock_list(self.dense_n)

        for i in range(self.nb_blocks) :
            x = self.dense_block(input_x=x, nb_layers=denseblock_list[i], layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        
        '''x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        #x = self.dense_block(input_x=x, nb_layers=24, layer_name='dense_3')
        #x = self.transition_layer(x, scope='trans_3')
        

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')
        #x = self.dense_block(input_x=x, nb_layers=16, layer_name='dense_final')'''

        #x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x=tf.layers.batch_normalization(x, training=self.training)
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x, regularizer=self.regularizer)
        return x
