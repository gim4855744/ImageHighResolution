import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.contrib.layers import fully_connected


def next_batch():

    dir_name = './images/'
    images = []

    for filename in os.listdir('./images'):
        file_path = dir_name + filename
        img = Image.open(file_path)
        img_arr = np.array(img)
        images.append(img_arr)


def main(argv):

    if len(argv) != 1:
        raise Exception('Problem with flags: %s' % argv)

    batch_size = 16
    learning_rate = 0.001
    num_train_steps = 10001

    input_height = 16
    input_width = 16
    input_channel = 3

    output_height = 32
    output_width = 32
    output_channel = 3

    input_size = input_height * input_width * input_channel
    output_size = output_height * output_width * output_channel

    with tf.Graph().as_default():

        # placeholder
        input_image = tf.placeholder(tf.float32, [batch_size, input_height, input_width, input_channel], 'input_image')
        output_image = tf.placeholder(tf.float32, [batch_size, output_height, output_width, output_channel], 'output_image')

        # convert images to 2D tensor
        in_flat_img = tf.reshape(input_image, [batch_size, input_size], 'in_flat_img')
        out_flat_img = tf.reshape(output_image, [batch_size, output_size], 'out_flat_img')

        # 1st fully connected
        num_hidden = input_size * 2
        value = fully_connected(in_flat_img, num_hidden)

        # 2nd fully connected
        num_hidden = num_hidden * 2
        value = fully_connected(value, num_hidden)

        # 3rd fully connected
        num_hidden = output_size
        value = fully_connected(value, num_hidden)

        diff = tf.subtract(value, out_flat_img)
        loss = tf.reduce_mean(tf.square(diff))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for step in range(num_train_steps):
                feed_dict = {input_image: [[[[1] * input_channel] * input_width] * input_height] * batch_size,
                             output_image: [[[[2] * output_channel] * output_width] * output_height] * batch_size}
                _, loss_val = sess.run([train_step, loss], feed_dict)
                if step % 100 == 0:
                    print(step, loss_val)

            out = sess.run(value, {input_image: [[[[2] * input_channel] * input_width] * input_height] * batch_size})
            print(out)


if __name__ == '__main__':
    tf.app.run()
