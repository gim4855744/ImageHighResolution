import numpy as np
from PIL import Image


class Batch:

    def __init__(self, input_data, target_data, batch_size):

        self._total_data = []
        self._batch_size = batch_size
        self.__batch_idx = 0

        for idx in range(len(input_data)):
            self._total_data.append([input_data[idx] / 255, target_data[idx] / 255])
        self._total_size = len(self._total_data)

    def next_batch(self):

        batch_data = []
        batch_input = []
        batch_target = []

        total_data = self._total_data
        total_size = self._total_size
        batch_size = self._batch_size

        for _ in range(batch_size):
            batch_data.append(total_data[self.__batch_idx])
            self.__batch_idx = (self.__batch_idx + 1) % total_size

        np.random.shuffle(batch_data)

        for data in batch_data:
            batch_input.append(data[0])
            batch_target.append(data[1])

        return batch_input, batch_target
