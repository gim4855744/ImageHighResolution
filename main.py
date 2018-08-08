import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', '', 'train/eval/convert')
tf.flags.DEFINE_string('data_path', '', 'train or test data path')
tf.flags.DEFINE_integer('num_threads', 4, 'num threads for queue runner')
tf.flags.DEFINE_integer('input_height', 32, 'input image height')
tf.flags.DEFINE_integer('input_width', 32, 'input image width')
tf.flags.DEFINE_integer('input_channel', 3, 'input image channel')
tf.flags.DEFINE_integer('output_height', 64, 'output image height')
tf.flags.DEFINE_integer('output_width', 64, 'output image width')
tf.flags.DEFINE_integer('output_channel', 3, 'output image channel')


def main(argv):
    pass


if __name__ == '__main__':
    tf.app.run()
