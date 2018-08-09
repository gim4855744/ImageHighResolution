import tensorflow as tf


def conv2d_transpose(features, output_shape, kernel_size, strides, name='conv2d_transpose'):

    with tf.variable_scope(name):

        kernel_size = list(kernel_size) + [output_shape[-1], features.get_shape()[-1]]
        strides = [1] + list(strides) + [1]

        kernel = tf.get_variable('kernel', kernel_size, tf.float32, tf.random_normal_initializer())
        biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, tf.constant_initializer())
        convolution = tf.nn.bias_add(tf.nn.conv2d_transpose(features, kernel, output_shape, strides), biases)

    return convolution
