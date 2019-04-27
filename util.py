import cv2
import numpy as np
import pandas as pd
from scipy.misc import imread
from PIL import ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from PIL import Image
import skimage
from random import uniform
from sklearn.preprocessing import normalize
from skimage import exposure
import os
import av
import shutil

def rgb_to_bgr(image):
    return image[:, :, ::-1].copy() 

def plot_img(img):
    plt.imshow(img)

def plot_img_file(path):
    img = imread(path, mode="RGB")
    show_img(img)


def equalize(img):
    img = np.array(img)
    for channel in range(img.shape[2]):  # equalizing each channel
        img[:, :, channel] = exposure.equalize_hist(img[:, :, channel])
    return img

def preprocess_image_load(image):
    image = image.resize((600, 600))
    image = skimage.img_as_ubyte(equalize(image))
    return np.array(image)

def preprocess_image(image):
    image = image.resize((600, 600))
    image = skimage.img_as_ubyte(equalize(image))
    return np.array(image)

def draw_video_frame(path, index):
    vid = cv2.VideoCapture(path)    
    vid.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = vid.read()
    if ret:
        frame = frame.astype('uint8')
        plot_img(frame)
        plt.show()
    vid.release()
    if not ret:
        raise Exception("Failed to retrieve frame")

def normalize_l1(arr):
    return normalize(arr.reshape(1 ,-1), 'l1').flatten()

def histogram(image, mask, bins, channels=[0, 1, 2], ranges=[0, 256, 0, 256, 0, 256]):
    # extract a 3D color histogram from the masked region of the
    # image, using the supplied number of bins per channel
    
    hist = cv2.calcHist([image], channels, mask, bins, ranges)
    return hist.flatten()
    
def gray_histogram_features(image, bins=[150]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = histogram(image, None, bins, channels=[0], ranges=[0,256])
    return normalize_l1(np.array(hist))

def color_histogram_features(image, bins=[9, 9, 9]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = histogram(image, None, bins=bins)
    return normalize_l1(np.array(hist))

def local_histogram_features(image, gray=False, bins=[5, 6, 6], norm=True):
    channels = [0, 1, 2]
    ranges = [0, 256, 0, 256, 0, 256]
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = [0]
        ranges = [0, 256]
    else: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hists = []

    # grab the dimensions and compute the center of the image
    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    # divide the image into four rectangles/segments (top-left,
    # top-right, bottom-right, bottom-left)
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
        (0, cX, cY, h)]

    # construct an elliptical mask representing the center of the
    # image
    (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
    ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

    # loop over the segments
    for (startX, endX, startY, endY) in segments:
        # construct a mask for each corner of the image, subtracting
        # the elliptical center from it
        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)

        # extract a color histogram from the image, then update the
        # feature vector
        hist = histogram(image, cornerMask, bins=bins, channels=channels, ranges=ranges)
        hists.append(hist)
    # extract a color histogram from the elliptical region and
    # update the feature vector
    hist = histogram(image, ellipMask, bins=bins, channels=channels, ranges=ranges)
    hists.append(hist)
    
    features = np.concatenate(hists)
    if norm:
        return normalize_l1(features)
    else:
        return features

def gray_local_histogram_features(image, bins=[125]):
    return local_histogram_features(image, gray=True, bins=bins)

def extract_descriptors(image, points=32, extractor=cv2.ORB_create, norm=True):
    vector_size = points * 32
    # Using KAZE, cause SIFT, ORB and other was moved to additional module
    # which is adding addtional pain during install
    alg = extractor(points)
    # Dinding image keypoints
    kps = alg.detect(image)
    # Getting first 32 of them. 
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:points]
    # computing descriptors vector
    kps, dsc = alg.compute(image, kps)
    # Making descriptor of same size
    # Descriptor vector size is 64
    #needed_size = (vector_size * 64)
    # Flatten all of them in one big vector - our feature vector
    dsc = dsc.flatten() if dsc is not None else np.zeros(vector_size)
    if dsc.size < vector_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        dsc = np.concatenate([dsc, np.zeros(vector_size - dsc.size)])
    #except cv2.error as e:
    #    print 'Error: ', e
    #    return None 
    if norm:
        return normalize_l1(dsc)
    else:
        return dsc

def joint_feature_extractor(image, k_points=2, bins=[4, 5, 5]):
    # color hist + orb
    hist_features = local_histogram_features(image, bins=bins, norm=False)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ds =  extract_descriptors(img, k_points, norm=False)
    features = np.concatenate([hist_features, ds]) 
    return normalize_l1(features)

def add_gaussian_noise(image, mean=0, var=0.1):
    image = np.array(image)
    row,col,ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy.astype(image.dtype)

def joint_corrupt(img, factor):
    # Noise, rotate, blur, change brightness, contrast, sharpness
    
    max_abs_rotation = 25
    min_rotation = -factor*max_abs_rotation
    max_rotation = factor*max_abs_rotation
    
    min_noise = 0.1
    max_noise = 0.5
    noise_variance = factor*max_noise
    
    min_blur_radius = 1
    max_blur_radius = 2
    blur_radius = factor*max_blur_radius
    
    color_enhance_factor = 1 - factor/2
    brightness_enhance_factor = 1 - factor/2
    sharpness_enhance_factor = 1 - factor/2
    contrast_enhance_factor = 1 - factor/2
    
    img = ImageEnhance.Color(img).enhance(color_enhance_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_enhance_factor)
    img = ImageEnhance.Sharpness(img).enhance(sharpness_enhance_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_enhance_factor)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    img = Image.fromarray(add_gaussian_noise(img, var=noise_variance))
    
    img = img.rotate(max_rotation)
    return img

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

def restore_image(img):
    cut = 0.05
    width, height = img.size
    crop_rectangle = (cut*width, cut*height, width-cut*width, height-cut*height)
    img = img.crop(crop_rectangle)
    if img.size > (width, height):
        img = img.thumbnail((width, height))

    img = np.array(img)
    #img = denoise_tv_chambolle(img, multichannel=True)
    img = equalize(img)
    return skimage.img_as_ubyte(img)


def get_video_rows(path, feature_extractor, preprocessor, write_frames_dir=None):
    frames_path = None
    if write_frames_dir:
        frames_path = os.path.join(write_frames_dir,
                                   os.path.basename(path) + '_frames')
        if os.path.exists(frames_path):
            shutil.rmtree(frames_path)
        os.makedirs(frames_path)

    rows = []
    video_name = os.path.basename(path)

    container = av.open(path)

    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'
    for frame in container.decode(stream):
        img = frame.to_image()
        img = preprocessor(img)
        features = feature_extractor(rgb_to_bgr(np.array(img)))
        row = [path, frame.time]  # metadata, features
        for i in range(len(features)):
            row.append(features[i])
        rows.append(tuple(row))
        if frames_path:
            fpath = os.path.join(frames_path, '{}.jpg'.format(str(frame.pts)))
            Image.fromarray(img).save(fpath)
    return rows


def get_dataframe(fpaths, feature_extractor, preprocessor=preprocess_image_load,
                  write_frames_dir=None):
    rows = []
    i = 0
    for fpath in fpaths:
        video_rows = get_video_rows(fpath, feature_extractor=feature_extractor,
                                    preprocessor=preprocessor,
                                    write_frames_dir=write_frames_dir)
        rows += video_rows
        i += 1
        print('Done ' + str(round(float(i) / len(fpaths), 2)))

    cols = ['video_path', 'frame_time']
    for i in range(len(rows[0]) - 2):
        cols.append('x_' + str(i))
    df = pd.DataFrame(rows, columns=cols, index=range(len(rows)))
    return df


def make_subclips(in_path, out_dir, number_subclips=1, subclip_duration=10):
    actual_video_name = os.path.basename(in_path)

    subclips_dir = os.path.join(out_dir, actual_video_name + '_subclips')
    if os.path.exists(subclips_dir):
        shutil.rmtree(subclips_dir)
    os.makedirs(subclips_dir)

    container = VideoFileClip(in_path)

    duration = container.duration

    fnames = []
    for i in range(number_subclips):
        subclip_start_time = np.random.randint(0, duration - subclip_duration - 1)
        subclip_range = (subclip_start_time, subclip_start_time + subclip_duration)
        subclip_fname = os.path.join(subclips_dir, str(subclip_range) + '.' +
                                     actual_video_name.split('.')[-1])
        ffmpeg_extract_subclip(in_path, subclip_range[0], subclip_range[1],
                               targetname=subclip_fname)
        fnames.append(subclip_fname)
    return fnames


def preprocess_corrupt(image, params):
    image = image.resize((600, 600))
    image = restore_image(corrupt(image, *params))
    return image


def get_corrupted_dataframe(fpaths, feature_extractor,
                            preprocessor=preprocess_image, write_frames_dir=None):
    rows = []
    i = 0
    for fpath in fpaths:
        cparams = random_corrupt_params()

        def video_frame_corrupt(img):
            return preprocess_corrupt(img, cparams)

        video_rows = get_video_rows(fpath, feature_extractor=feature_extractor,
                                    preprocessor=video_frame_corrupt,
                                    write_frames_dir=write_frames_dir)
        rows += video_rows
        i += 1
        print('Done ' + str(round(float(i) / len(fpaths), 2)))

    cols = ['video_path', 'frame_time']
    for i in range(len(rows[0]) - 2):
        cols.append('x_' + str(i))
    df = pd.DataFrame(rows, columns=cols, index=range(len(rows)))
    return df


def get_subclip_dataframe(fpaths, feature_extractor, subclip_dir,
                          remove_subclips=True, number_subclips=10,
                          subclip_duration=10):
    dfs = []
    for fpath in fpaths:
        subclip_corruption = random_corrupt_params()
        subclip_fnames = make_subclips(fpath, subclip_dir,
                                       number_subclips=number_subclips,
                                       subclip_duration=subclip_duration)

        subclip_df = get_corrupted_dataframe(subclip_fnames,
                                             feature_extractor=feature_extractor)
        subclip_df['source_fpath'] = fpath
        dfs.append(subclip_df)
        if remove_subclips:
            for fname in subclip_fnames:
                if os.path.exists(fname):
                    os.remove(fname)
            os.rmdir(os.path.dirname(fname))

    df = pd.concat(dfs, axis=0)
    return df