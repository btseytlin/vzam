import cv2
import numpy as np
from skimage.feature import hog

from vzam.image_manipulation import histogram
from vzam.util import normalize_l1


def gray_histogram_features(image, bins=[150]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = histogram(image, None, bins, channels=[0], ranges=[0, 256])
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
    ellipMask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

    # loop over the segments
    for (startX, endX, startY, endY) in segments:
        # construct a mask for each corner of the image, subtracting
        # the elliptical center from it
        cornerMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)

        # extract a color histogram from the image, then update the
        # feature vector
        hist = histogram(image, cornerMask, bins=bins, channels=channels,
                         ranges=ranges)
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


def hog_features(image):
    features = hog(image, orientations=9, pixels_per_cell=(64, 64),
                    cells_per_block=(1, 1), multichannel=True)
    return normalize_l1(features)


def hog_localhist_features(image, weights=(0.2, 0.8)):
    fd1 = local_histogram_features(image, bins=[4, 4, 4])
    fd2 = hog_features(image)
    return np.hstack([weights[0]*fd1,
                      weights[1]*fd2])


def gray_local_histogram_features(image, bins=[256]):
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
    # needed_size = (vector_size * 64)
    # Flatten all of them in one big vector - our feature vector
    dsc = dsc.flatten() if dsc is not None else np.zeros(vector_size)
    if dsc.size < vector_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        dsc = np.concatenate([dsc, np.zeros(vector_size - dsc.size)])
    # except cv2.error as e:
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
    ds = extract_descriptors(img, k_points, norm=False)
    features = np.concatenate([hist_features, ds])
    return normalize_l1(features)

def rHash(image, hash_size=64):
    image = np.asarray(image)
    
    n_blocks_w = np.sqrt(hash_size)
    n_blocks_h = np.sqrt(hash_size)
    
    block_width = int(len(image)//n_blocks_w)
    block_height = int(len(image)//n_blocks_h)
    
    assert image.shape[0] % n_blocks_w == 0
    assert image.shape[1] % n_blocks_h == 0
    
    block_means = []
    for i in range(0, len(image), block_height):
        for j in range(0, image.shape[1], block_width):
            mean = np.mean(image[i:i+block_height,j:j+block_width])
            block_means.append(mean)
    fingerprint = np.array(block_means) >= np.median(block_means)
    return fingerprint.astype(int)


def quandrant_rHash(image, hash_size=64):
    image = np.asarray(image)
    mid_x = len(image) // 2
    mid_y = image.shape[1] // 2

    hsize = hash_size // 4
    fingerprints = []
    fingerprints.append(rHash(image[0:mid_x,0:mid_y],  hsize))
    fingerprints.append(rHash(image[mid_x:,0:mid_y],  hsize))
    fingerprints.append(rHash(image[0:mid_x,mid_y:],  hsize))
    fingerprints.append(rHash(image[mid_x:,mid_y:],  hsize))
    return np.hstack(fingerprints)
