import os
import tensorflow as tf
import numpy as np

from PIL import Image
from utils import Batch


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', './images/input/', 'input images directory')
tf.app.flags.DEFINE_string('target_dir', './images/target/', 'target images directory')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')


INPUT_DIR = FLAGS.input_dir
TARGET_DIR = FLAGS.target_dir
BATCH_SIZE = FLAGS.batch_size

INPUT_HEIGHT = 16
INPUT_WIDTH = 16
INPUT_CHANNELS = 3

TARGET_HEIGHT = 32
TARGET_WIDTH = 32
TARGET_CHANNELS = 3


def main(argv):

    if len(argv) != 1:
        raise Exception

    input_images = []
    target_images = []

    for img_name in sorted(os.listdir(INPUT_DIR)):
        img_path = INPUT_DIR + img_name
        img = Image.open(img_path)
        img.load()
        data = np.asarray(img, 'int32')
        input_images.append(data)
    for img_name in sorted(os.listdir(TARGET_DIR)):
        img_path = TARGET_DIR + img_name
        img = Image.open(img_path)
        img.load()
        data = np.asarray(img, 'int32')
        target_images.append(data)

    batch = Batch(input_images, target_images, BATCH_SIZE)

    with tf.Graph().as_default():

        inputs = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS])
        targets = tf.placeholder(tf.float32, [None, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS])

        strides = [1, 1, 1, 1]

        filter1 = tf.Variable(tf.truncated_normal([2, 2, INPUT_CHANNELS, 12], stddev=0.1))
        bias1 = tf.Variable(tf.zeros([12]))
        convolution1 = tf.nn.leaky_relu(tf.nn.conv2d(inputs, filter1, strides, padding='SAME') + bias1, 0.2)
        max_pool1 = tf.nn.max_pool(convolution1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        filter2 = tf.Variable(tf.truncated_normal([2, 2, 12, 48], stddev=0.1))
        bias2 = tf.Variable(tf.zeros([48]))
        convolution2 = tf.nn.leaky_relu(tf.nn.conv2d(max_pool1, filter2, strides, padding='SAME') + bias2, 0.2)
        max_pool2 = tf.nn.max_pool(convolution2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        filter3 = tf.Variable(tf.truncated_normal([2, 2, 48, 192], stddev=0.1))
        bias3 = tf.Variable(tf.zeros([192]))
        convolution3 = tf.nn.leaky_relu(tf.nn.conv2d(max_pool2, filter3, strides, padding='SAME') + bias3, 0.2)
        max_pool3 = tf.nn.max_pool(convolution3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        filter4 = tf.Variable(tf.truncated_normal([2, 2, 192, 768], stddev=0.1))
        bias4 = tf.Variable(tf.zeros([768]))
        convolution4 = tf.nn.leaky_relu(tf.nn.conv2d(max_pool3, filter4, strides, padding='SAME') + bias4, 0.2)
        max_pool4 = tf.nn.max_pool(convolution4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        image = tf.reshape(max_pool4, [-1, 16, 16, 3])

        loss = tf.reduce_mean(tf.square(tf.subtract(image, inputs)))
        optimizer = tf.train.AdamOptimizer(0.0001)
        train_step = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(100001):
            batch_inputs, batch_targets = batch.next_batch()
            feed_dict = {inputs: batch_inputs, targets: batch_targets}
            _, loss_val = sess.run([train_step, loss], feed_dict)
            if step % 1000 == 0:
                print(step, loss_val)

        batch_inputs, batch_targets = batch.next_batch()
        feed_dict = {inputs: batch_inputs, targets: batch_targets}
        img = sess.run(image, feed_dict)

        data = Image.fromarray(np.asarray(np.clip(batch_inputs[0] * 255, 0, 255), 'uint8'), 'RGB')
        data.show()

        data = Image.fromarray(np.asarray(np.clip(img[0] * 255, 0, 255), 'uint8'), 'RGB')
        data.show()

        '''
        data = batch_inputs[0] * 255
        img = Image.fromarray(np.asarray(np.clip(data, 0, 255), 'uint8'), 'RGB')

        data = batch_targets[0] * 255
        img = Image.fromarray(np.asarray(np.clip(data, 0, 255), 'uint8'), 'RGB')
        '''


if __name__ == '__main__':
    tf.app.run()
