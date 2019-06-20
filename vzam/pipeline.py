import numpy as np
from vzam.image_manipulation import equalize


def preprocess_image_load(image):
    image = equalize(image)
    return np.array(image)


def restore_image(image):
    return preprocess_image_load(image)
