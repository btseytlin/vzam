import cv2
import numpy as np
from PIL import ImageEnhance, ImageFilter
from PIL import Image
import skimage
from random import uniform
from skimage import exposure


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
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
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
