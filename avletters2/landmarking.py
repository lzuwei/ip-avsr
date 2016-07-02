import warnings
import numpy as np
import os, errno
import csv

import menpo.io as mio
from menpo.visualize import print_progress
from menpo.feature import igo, fast_dsift
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68
from menpodetect.dlib import load_dlib_frontal_face_detector
from menpofit.dlib import DlibWrapper
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter, ModifiedAlternatingInverseCompositional
from menpowidgets import visualize_images

# constants, change according to system
AVLETTERS2_DIR = '/Volumes/New Volume/Thesis/avletters2'
FACE_MODEL_PATH = '/Volumes/New Volume/Thesis/avletters2/Configs/shape_predictor_68_face_landmarks.dat'


def find_all_videos(dir, ext=['.mp4', '.mov'], relpath=False):
    videofiles = []
    find_all_videos_impl(dir, videofiles, ext)
    if relpath:
        for i, f in enumerate(videofiles):
            videofiles[i] = f[len(dir) + 1:]
    return videofiles


def find_all_videos_impl(dir, videofiles, ext):
    files = os.listdir(dir)
    for f in files:
        path = os.path.join(dir, f)
        if os.path.isdir(path):
            find_all_videos_impl(path, videofiles, ext)
        elif os.path.splitext(f)[1] in ext:
            videofiles.append(path)


def is_video(file, ext=['.mp4', '.mov']):
    return os.path.splitext(file)[1] in ext


def fit_image(image):
    # Face detection
    bboxes = fit_image.detect(image, image_diagonal=1000)

    # Check if at least one face was detected, otherwise throw a warning
    if len(bboxes) > 0:
        # Use the first bounding box (the most probable to represent a face) to initialise
        fitting_result = fit_image.fitter.fit_from_bb(image, bboxes[0])

        # Assign shape on the image
        image.landmarks['final_shape'] = fitting_result.final_shape
    else:
        # Throw warning if no face was detected
        warnings.warn("No face detected")

    # Return the image
    return image


def create_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def process_video(file, dest):
    if is_video(file):
        create_dir(os.path.dirname(dest))
        frames = mio.import_video(file, normalise=False)
        print('{} contains {} frames'.format(file, len(frames)))
        print('writing landmarks to {}...'.format(dest))
        frames = frames.map(fit_image)
        with open(dest, 'w') as outputfile:
            outwriter = csv.writer(outputfile)
            for i, frame in enumerate(print_progress(frames)):
                if 'final_shape' not in frame.landmarks:
                    initial_shape = frames[i - 1].landmarks['final_shape'].lms
                    fitting_result = fit_image.fitter.fit_from_shape(frame, initial_shape)
                    frame.landmarks['final_shape'] = fitting_result.final_shape
                    landmarks = frame.landmarks['all'].points.reshape((136,)).tolist()
                    row = [i] + landmarks
                else:
                    lmg = frame.landmarks['final_shape']
                    landmarks = lmg['all'].points.reshape((136,)).tolist()  # reshape to 136 points
                    row = [i] + landmarks
                outwriter.writerow(row)


if __name__ == '__main__':
    print('Generating Landmarks for AV Letters 2 Dataset...')
    fit_image.detect = load_dlib_frontal_face_detector()
    fit_image.fitter = DlibWrapper(FACE_MODEL_PATH)
    videofiles = find_all_videos(os.path.join(AVLETTERS2_DIR, 'Videos'), relpath=True)
    for video in videofiles:
        landmarkfile = os.path.splitext(video)[0] + '.csv'
        process_video(os.path.join(AVLETTERS2_DIR, 'Videos', video),
                      os.path.join(AVLETTERS2_DIR, 'Landmarks', landmarkfile))
    print('All Done!')