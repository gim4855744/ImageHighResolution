import os
import tensorflow as tf
import numpy as np

from collections import namedtuple
from PIL import Image
from utils import Batch


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', './images/input/', 'input images directory')
tf.app.flags.DEFINE_string('target_dir', './images/target/', 'target images directory')

HyperParams = namedtuple(
    'HyperParams',
    [
        'batch_size',
        'learning_rate'
    ]
)

INPUT_DIR = FLAGS.input_dir
TARGET_DIR = FLAGS.target_dir

INPUT_HEIGHT = 16
INPUT_WIDTH = 16
INPUT_CHANNELS = 3

TARGET_HEIGHT = 32
TARGET_WIDTH = 32
TARGET_CHANNELS = 3


def load_images(dir_path):
    images = []
    for img_name in sorted(os.listdir(dir_path)):
        img_path = dir_path + img_name
        img = Image.open(img_path)
        arr_img = np.array(img, 'int32')
        images.append(arr_img)
    return images


def save_image(save_path, arr_image):
    image = Image.fromarray(np.asarray(np.clip(arr_image * 255, 0, 255), 'uint8'), 'RGB')
    image.save(save_path)


def convolution(inputs, filters):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=5,
        strides=(1, 1),
        padding='same',
        activation=tf.nn.leaky_relu
    )


def deconvolution(inputs, filters):
    return tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=5,
        strides=(2, 2),
        padding='same',
        activation=tf.nn.leaky_relu
    )


def dense(inputs, units):
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=tf.nn.leaky_relu
    )


def main(argv):

    if len(argv) != 1:
        raise Exception

    hps = HyperParams(
        batch_size=64,
        learning_rate=0.0001
    )

    input_images = load_images(INPUT_DIR)
    target_images = load_images(TARGET_DIR)

    batch = Batch(input_images, target_images, hps.batch_size)

    inputs = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS])
    targets = tf.placeholder(tf.float32, [None, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS])

    '''
    convolution1 = convolution(inputs, 12)
    convolution2 = convolution(convolution1, 48)
    convolution3 = convolution(convolution2, 192)
    convolution4 = convolution(convolution3, 768)

    convolution4_flat = tf.reshape(convolution4, [-1, 1 * 1 * 768])
    dense1 = dense(convolution4_flat, 1 * 1 * 768)

    deconvolution1 = deconvolution(tf.reshape(dense1, [-1, 1, 1, 768]), 192)
    deconvolution2 = deconvolution(deconvolution1, 48)
    deconvolution3 = deconvolution(deconvolution2, 12)
    deconvolution4 = deconvolution(deconvolution3, 3)
    deconvolution5 = deconvolution(deconvolution4, 3)

    deconvolution5_flat = tf.reshape(deconvolution5, [-1, 32 * 32 * 3])
    dense2 = dense(deconvolution5_flat, 32 * 32 * 3)

    predict = tf.reshape(dense2, [-1, 32, 32, 3])
    '''

    deconvolution1 = deconvolution(inputs, 81)
    convolution1 = convolution(deconvolution1, 27)
    convolution2 = convolution(convolution1, 9)
    predict = convolution(convolution2, 3)

    loss = tf.losses.mean_squared_error(labels=targets, predictions=predict)
    optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # saver.restore(sess, './tmp/model.ckpt')

    for step in range(50001):
        batch_inputs, batch_targets = batch.next_batch()
        feed_dict = {inputs: batch_inputs, targets: batch_targets}
        _, loss_val = sess.run([train_step, loss], feed_dict)
        if step % 10000 == 0:
            save_path = saver.save(sess, './tmp/model.ckpt')
            print('model saved in %s' % save_path)
        if step % 1000 == 0:
            print(step, loss_val)

    test_input = load_images('./images/test_inputs/')
    test_target = load_images('./images/test_targets/')

    batch = Batch(test_input, test_target, 1)
    batch_inputs, batch_targets = batch.next_batch()

    feed_dict = {inputs: batch_inputs, targets: batch_targets}
    img = sess.run(predict, feed_dict)

    save_image('./target.jpg', batch_targets[0])
    save_image('./predict.jpg', img[0])


if __name__ == '__main__':
    tf.app.run()
