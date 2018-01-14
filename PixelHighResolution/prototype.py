import tensorflow as tf
import numpy as np

batch_size = 8

source_height = 16
source_width = 16
source_channel = 3

target_height = 32
target_width = 32
target_channel = 3

with tf.Graph().as_default():

    source_images = tf.placeholder(tf.float32, [batch_size, source_height, source_width, source_channel], 'source_images')
    target_images = tf.placeholder(tf.float32, [batch_size, target_height, target_width, target_channel], 'target_images')

    convolution_weight1 = tf.get_variable('convolution_weight1', [2, 2, 3, 16], tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    convolution_bias1 = tf.get_variable('convolution_bias1', [16], tf.float32, tf.zeros_initializer())
    convolution1 = tf.nn.conv2d(source_images, convolution_weight1, [1, 1, 1, 1], 'SAME') + convolution_bias1

    convolution_weight2 = tf.get_variable('convolution_weight2', [2, 2, 16, 32], tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    convolution_bias2 = tf.get_variable('convolution_bias2', [32], tf.float32, tf.zeros_initializer())
    convolution2 = tf.nn.conv2d(convolution1, convolution_weight2, [1, 1, 1, 1], 'SAME') + convolution_bias2

    max_pool1 = tf.nn.max_pool(convolution2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    convolution_weight3 = tf.get_variable('convolution_weight3', [8, 8, 32, 64], tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    convolution_bias3 = tf.get_variable('convolution_bias3', [64], tf.float32, tf.zeros_initializer())
    convolution3 = tf.nn.conv2d(max_pool1, convolution_weight3, [1, 1, 1, 1], 'SAME') + convolution_bias3

    max_pool2 = tf.nn.max_pool(convolution3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    fc_weight1 = tf.get_variable('fw_weight1', [1024, 3072], tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    fc_bias1 = tf.get_variable('fc_bias1', [3072], tf.float32, tf.zeros_initializer())
    fc1 = tf.matmul(tf.reshape(max_pool2, [batch_size, 1024]), fc_weight1) + fc_bias1

    outputs = tf.reshape(fc1, [batch_size, 32, 32, 3])
    target = [[[[2] * target_channel] * target_width] * target_height] * batch_size

    loss = tf.reduce_mean(tf.square(tf.subtract(outputs, target)))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10001):
            output, loss_val = sess.run([train_step, loss], feed_dict={source_images: [[[[1] * source_channel] * source_width] * source_height] * batch_size})
            if step % 100 == 0:
                print(step, loss_val)
        print(sess.run(outputs, feed_dict={source_images: [[[[1] * source_channel] * source_width] * source_height] * batch_size}))
