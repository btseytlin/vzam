import numpy as np
import pandas as pd
from PIL import Image
import os
import av
import shutil

from vzam.image_manipulation import random_corrupt_params, corrupt, rgb_to_bgr
from vzam.pipeline import preprocess_image_load
from vzam.video_manipulation import make_subclips


def get_video_rows(path,
                   feature_extractor,
                   preprocessor,
                   keyframe_only=True,
                   write_frames_dir=None,
                   mod_filter=None):
    frames_path = None
    if write_frames_dir:
        frames_path = os.path.join(write_frames_dir,
                                   os.path.basename(path) + '_frames')
        if os.path.exists(frames_path):
            shutil.rmtree(frames_path)
        os.makedirs(frames_path)

    rows = []

    container = av.open(path)

    stream = container.streams.video[0]
    if keyframe_only:
        stream.codec_context.skip_frame = 'NONKEY'

    frame_idx = 0

    for frame in container.decode(stream):
        frame_idx += 1
        if mod_filter is not None and frame_idx % mod_filter != 0:
            continue
        img = frame.to_image()
        img = preprocessor(img)
        features = feature_extractor(rgb_to_bgr(np.array(img)))
        row = [path, round(frame.time, 1)]  # metadata
        for i in range(len(features)):
            row.append(features[i]) # add features
        rows.append(tuple(row))
        if frames_path:
            fpath = os.path.join(frames_path, '{}.jpg'.format(str(round(frame.time,1))))
            Image.fromarray(img).save(fpath)
    return rows


def rows_to_df(rows):
    cols = ['video_path', 'frame_time']
    for i in range(len(rows[0]) - 2):
        cols.append('x_' + str(i))
    df = pd.DataFrame(rows, columns=cols)
    return df


def get_dataframe(fpaths,
                  feature_extractor,
                  preprocessor,
                  keyframe_only=True,
                  write_frames_dir=None,
                  mod_filter=None):
    rows = []
    i = 0
    for fpath in fpaths:
        print('Processing', fpath)
        video_rows = get_video_rows(fpath,
                                    feature_extractor=feature_extractor,
                                    preprocessor=preprocessor,
                                    keyframe_only=keyframe_only,
                                    write_frames_dir=write_frames_dir,
                                    mod_filter=mod_filter)
        rows += video_rows
        i += 1
        print('Done ' + str(round(float(i) / len(fpaths), 2)))

    return rows_to_df(rows)


def get_corrupted_dataframe(fpaths,
                            feature_extractor,
                            keyframe_only=True,
                            write_frames_dir=None,
                            mod_filter=None):
    rows = []
    i = 0
    for fpath in fpaths:
        print('Processing', fpath)
        cparams = random_corrupt_params()

        def video_frame_corrupt(img):
            return preprocess_image_load(corrupt(img, *cparams))

        video_rows = get_video_rows(fpath, feature_extractor=feature_extractor,
                                    preprocessor=video_frame_corrupt,
                                    write_frames_dir=write_frames_dir,
                                    keyframe_only=keyframe_only,
                                    mod_filter=mod_filter)
        rows += video_rows
        i += 1
        print('Done ' + str(round(float(i) / len(fpaths), 2)))

    return rows_to_df(rows)


def get_subclip_dataframe(fpaths,
                          feature_extractor,
                          subclip_dir,
                          number_subclips=10,
                          subclip_duration=20,
                          mod_filter=None,
                          write_frames_dir=None,
                          keyframe_only=False,
                          remove_subclips=True):
    dfs = []
    for fpath in fpaths:
        subclip_fnames = make_subclips(fpath,
                                       subclip_dir,
                                       number_subclips=number_subclips,
                                       subclip_duration=subclip_duration)

        subclip_df = get_corrupted_dataframe(subclip_fnames,
                                             write_frames_dir=write_frames_dir,
                                             mod_filter=mod_filter,
                                             feature_extractor=feature_extractor,
                                             keyframe_only=keyframe_only)
        subclip_df['clip_fpath'] = subclip_df['video_path']
        subclip_df['video_path'] = fpath
        dfs.append(subclip_df)

        if remove_subclips:
            for fname in subclip_fnames:
                if os.path.exists(fname):
                    os.remove(fname)
            os.rmdir(os.path.dirname(fname))

    df = pd.concat(dfs, axis=0)
    return df