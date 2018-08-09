import numpy as np
import os
from PIL import Image


def normalization(images):
    return images / 127.5 - 1


def get_channels(path):
    filename = os.listdir(path)
    file_path = os.path.join(path, filename[0])
    image = Image.open(file_path)
    channels = np.shape(image)[2]
    return channels
