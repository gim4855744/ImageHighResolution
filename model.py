import tensorflow as tf
from batch_queue import BatchQueue
from layers import conv2d_transpose
from utils import normalization


class ImageHighResolution:

    def __init__(
            self,
            path, num_threads, batch_size, learning_rate, input_height,
            input_width, output_height, output_width, channels
    ):

        input_shape = (batch_size, input_height, input_width, channels)
        output_shape = (batch_size, output_height, output_width, channels)

        self.input_images = tf.placeholder(tf.float32, input_shape, 'input_images')
        self.output_images = tf.placeholder(tf.float32, output_shape, 'output_images')

        kernel_size = (3, 3)
        strides = (2, 2)

        self.gen_images = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                conv2d_transpose(
                    self.input_images, (batch_size, output_height, output_width, channels),
                    kernel_size, strides, 'gen_images'
                )
            )
        )

        flatten_output_images = tf.reshape(self.output_images, [batch_size, -1])
        flatten_gen_images = tf.reshape(self.gen_images, [batch_size, -1])

        self.loss = tf.losses.mean_squared_error(self.output_images, self.gen_images)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.coord = tf.train.Coordinator()
        self.queue = BatchQueue(
            path, batch_size, num_threads, (input_height, input_width), (output_height, output_width),
            self.sess, self.coord
        )

    def train(self, num_steps):

        self.sess.run(tf.global_variables_initializer())
        for step in range(num_steps):
            input_images, output_images = self.sess.run(self.queue.dequeue())
            input_images = normalization(input_images)
            output_images = normalization(output_images)
            feed_dict = {self.input_images: input_images, self.output_images: output_images}
            _, loss = self.sess.run([self.train_step, self.loss], feed_dict)
            print(step + 1, loss)

    def eval(self):
        pass

    def convert(self):
        pass
