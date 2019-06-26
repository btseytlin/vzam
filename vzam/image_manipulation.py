import cv2
import numpy as np
from PIL import ImageEnhance, ImageFilter
from PIL import Image
import skimage
from random import uniform
from skimage import exposure
import av


def exponentially_weighted_average(arrays, gamma):
    """
    :param arrays: list of grayscale images, intensities expected to be in range [0, 1]
    :param gamma:
    :return:
    """
    arr = np.zeros(arrays[0].shape, np.float)
    weights = [np.power(gamma, k) for k, array in enumerate(arrays)]
    for k, array in enumerate(arrays):
        arr += weights[k] * array
    arr = arr / sum(weights)
    return arr


# def extract_tiri(video_fpath,
#                  buffer_size=20,
#                  gamma=1.65):
#     """
#     :param video_fpath: str, path to video file
#     :param buffer_size: int, amount of images to average
#     :param gamma: float, exponential weighting parameter
#     :return:
#     """
#     container = av.open(video_fpath)

#     stream = container.streams.video[0]
#     frame_idx = 0
#     tiris = []
#     timestamps = []

#     buffer_images = []
#     for frame in container.decode(stream):
#         img = np.array(frame.to_image().convert('L')) / 256
#         buffer_images.append(img)
#         if (frame_idx + 1) % buffer_size == 0:
#             tiri = exponentially_weighted_average(buffer_images, gamma)
#             tiris.append(tiri)
#             timestamps.append(round(frame.time,3))
#             buffer_images = []
#     return tiris, timestamps


def rgb_to_bgr(image):
    return image[:, :, ::-1].copy()


def equalize(img):
    img = np.array(img)
    img = skimage.img_as_float(img)
    for channel in range(img.shape[2]):  # equalizing each channel
        img[:, :, channel] = exposure.equalize_hist(img[:, :, channel])
    img = skimage.img_as_ubyte(img)
    return img


def histogram(image, mask, bins, channels=[0, 1, 2],
              ranges=[0, 256, 0, 256, 0, 256]):
    hist = cv2.calcHist([image], channels, mask, bins, ranges)
    return hist.flatten()


def add_gaussian_noise(image, mean=0, var=0.1):
    image = np.array(image)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (image.shape))
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    return noisy.astype(image.dtype)


def corrupt(img, enh_clr, enh_brt, enh_shrp, enh_contr, nvar, blur_rad, rotation):
    img = ImageEnhance.Color(img).enhance(enh_clr)
    img = ImageEnhance.Brightness(img).enhance(enh_brt)
    img = ImageEnhance.Sharpness(img).enhance(enh_shrp)
    img = ImageEnhance.Contrast(img).enhance(enh_contr)

    img = Image.fromarray(add_gaussian_noise(img, var=nvar))

    img = img.filter(ImageFilter.GaussianBlur(radius=blur_rad))

    img = img.rotate(rotation)
    return img


def random_corrupt_params(rotation_range=[-10, 10],
                          noise_variance_range=[0.01, 0.10],
                          blur_radius_range=[0.1, 1.5],
                          enhance_factor_range=[0.8, 1]):
    return [
        uniform(*enhance_factor_range),
        uniform(*enhance_factor_range),
        uniform(*enhance_factor_range),
        uniform(*enhance_factor_range),
        uniform(*noise_variance_range),
        uniform(*blur_radius_range),
        uniform(*rotation_range),
    ]


def random_corrupt(img):
    return corrupt(img, *random_corrupt_params())
