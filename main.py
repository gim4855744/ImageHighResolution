import tensorflow as tf
from utils import get_channels
from model import ImageHighResolution


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', '', 'train/eval/convert')
tf.flags.DEFINE_string('data_path', '', 'train or test data path')
tf.flags.DEFINE_integer('num_threads', 4, 'num threads for queue runner')
tf.flags.DEFINE_integer('num_steps', 100, 'num train steps')
tf.flags.DEFINE_integer('batch_size', 16, 'num mini batch size')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
tf.flags.DEFINE_integer('input_height', 32, 'input image height')
tf.flags.DEFINE_integer('input_width', 32, 'input image width')
tf.flags.DEFINE_integer('output_height', 64, 'output image height')
tf.flags.DEFINE_integer('output_width', 64, 'output image width')


def main(argv):

    channels = get_channels(FLAGS.data_path)
    model = ImageHighResolution(
        FLAGS.data_path, FLAGS.num_threads, FLAGS.batch_size, FLAGS.learning_rate,
        FLAGS.input_height, FLAGS.input_width,
        FLAGS.output_height, FLAGS.output_width,
        channels
    )

    if FLAGS.mode == 'train':
        model.train(FLAGS.num_steps)
    elif FLAGS.mode == 'eval':
        pass
    elif FLAGS.mode == 'convert':
        pass
    else:
        pass


if __name__ == '__main__':
    tf.app.run()
