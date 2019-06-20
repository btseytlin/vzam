import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import os
import shutil


def draw_video_frame(path, index):
    vid = cv2.VideoCapture(path)
    vid.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = vid.read()
    if ret:
        frame = frame.astype('uint8')
        plt.imshow(frame)
        plt.show()
    vid.release()
    if not ret:
        raise Exception("Failed to retrieve frame")


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