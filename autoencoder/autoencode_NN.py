import tensorflow as tf

img_channels=3

def autoencoder_NN(input_layer):
    
    # 28x28x1 => 28x28x8
    conv1 = tf.layers.conv2d(input_layer, filters=8, kernel_size=(3, 3),
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

    
    # 7x7x4 => 14x14x8
    deconv1 = tf.layers.conv2d_transpose(encode, filters=8, 
                                         kernel_size=(3, 3), strides=(2, 2), 
                                         padding='same',
                                         activation=tf.nn.relu)
    
    
    # 14x14x8 => 28x28x8
    deconv2 = tf.layers.conv2d_transpose(deconv1, filters=8, 
                                         kernel_size=(3, 3), strides=(2, 2), 
                                         padding='same',
                                         activation=tf.nn.relu)
    
    # 28x28x8 => 28x28x1
    logits = tf.layers.conv2d(deconv2, filters=img_channels, kernel_size=(3,3), 
                              strides=(1, 1), padding='same', 
                              activation=None)
    
    return logits