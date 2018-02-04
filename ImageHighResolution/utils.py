import numpy as np


class Batch:

    def __init__(self, input_images, output_images, batch_size):

        if len(input_images) != len(output_images):
            raise Exception

        self._total_images = []
        self._num_images = len(input_images)
        self._batch_size = batch_size
        self.__batch_idx = 0

        for idx in range(self._num_images):
            self._total_images.append([input_images[idx] / 255, output_images[idx] / 255])

    def next_batch(self):

        batch_images = []
        input_batch = []
        output_batch = []

        for _ in range(self._batch_size):
            batch_images.append(self._total_images[self.__batch_idx])
            self.__batch_idx = (self.__batch_idx + 1) % self._num_images

        np.random.shuffle(batch_images)

        for input_img, output_img in batch_images:
            input_batch.append(input_img)
            output_batch.append(output_img)

        return input_batch, output_batch
