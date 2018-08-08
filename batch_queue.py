import tensorflow as tf
import numpy as np
import os
from PIL import Image


class BatchQueue:

    def __init__(self, path, batch_size, num_threads, input_size, output_size, sess=None, coord=None):

        file_path = [os.path.join(path, filename) for filename in os.listdir(path)]
        filename_queue = tf.train.string_input_producer(file_path)

        reader = tf.WholeFileReader()
        decoder = tf.image.decode_image
        _, data = reader.read(filename_queue)

        shape = np.shape(Image.open(file_path[0]))
        image = decoder(data, shape[2])
        image.set_shape(shape)

        min_after_dequeue = batch_size * 100
        capacity = min_after_dequeue + (num_threads * batch_size)
        queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, tf.uint8, shape)
        enqueue_ops = queue.enqueue(image)
        queue_runner = tf.train.QueueRunner(queue, [enqueue_ops] * num_threads)
        tf.train.add_queue_runner(queue_runner)

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.sess = sess
        self.coord = coord
        self.queue = queue
        self.threads = tf.train.start_queue_runners(sess, coord)

    def dequeue(self):
        images = self.queue.dequeue_many(self.batch_size)
        input_images = tf.image.resize_nearest_neighbor(images, self.input_size)
        output_images = tf.image.resize_nearest_neighbor(images, self.output_size)
        return input_images, output_images

    def stop_threads(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
