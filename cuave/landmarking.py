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
CUAVE_DIR = '/Volumes/New Volume/Thesis/cuave/individuals'
FACE_MODEL_PATH = '../config/shape_predictor_68_face_landmarks.dat'
EXT = ['.mp4', '.mov', '.mpg']


def find_all_videos(dir, ext=EXT, relpath=False):
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


def is_video(file, ext=EXT):
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
        warnings.warn('No face detected')

    # Return the image
    return image


def create_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def fill_row(outwriter, frame_no, row):
    outwriter.writerow([frame_no] + row)


def process_video(file, dest):
    if is_video(file):
        create_dir(os.path.dirname(dest))
        frames = mio.import_video(file, normalise=False)
        print('{} contains {} frames'.format(file, len(frames)))
        print('writing landmarks to {}...'.format(dest))
        frames = frames.map(fit_image)
        with open(dest, 'w') as outputfile:
            outwriter = csv.writer(outputfile)
            try:
                for i, frame in enumerate(print_progress(frames)):
                    if 'final_shape' not in frame.landmarks:
                        warnings.warn('no faces detected in the frame {}, '
                                      'initializing landmarks to -1s...'.format(i))
                        # dlib does not fitting from previous initial shape so
                        # leave entire row as -1s
                        # initial_shape = frames[i - 1].landmarks['final_shape'].lms
                        # fitting_result = fit_image.fitter.fit_from_shape(frame, initial_shape)
                        # frame.landmarks['final_shape'] = fitting_result.final_shape
                        landmarks = [-1] * 136
                    else:
                        lmg = frame.landmarks['final_shape']
                        landmarks = lmg['all'].points.reshape((136,)).tolist()  # reshape to 136 points
                    fill_row(outwriter, i, landmarks)
            except Exception as e:
                warnings.warn('Runtime Error at frame {}'.format(i))
                print('initializing landmarks to -1s...')
                fill_row(outwriter, i, [-1] * 136)


if __name__ == '__main__':
    print('Generating Landmarks for CUAVE Dataset...')
    fit_image.detect = load_dlib_frontal_face_detector()
    fit_image.fitter = DlibWrapper(FACE_MODEL_PATH)
    videofiles = find_all_videos(os.path.join(CUAVE_DIR, 'videos'), relpath=True)
    print(videofiles)
    exit()
    for video in videofiles[7:]:
        landmarkfile = os.path.splitext(video)[0] + '.csv'
        process_video(os.path.join(CUAVE_DIR, 'videos', video),
                      os.path.join(CUAVE_DIR, 'landmarks', landmarkfile))
    print('All Done!')
