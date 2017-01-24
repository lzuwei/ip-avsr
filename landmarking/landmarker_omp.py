import warnings
import os
import errno
import csv
import argparse

import menpo.io as mio
from menpo.visualize import print_progress
from menpodetect.dlib import load_dlib_frontal_face_detector
from menpofit.dlib import DlibWrapper

import multiprocessing as mp

# constants, change according to system
FACE_MODEL_PATH = '../config/shape_predictor_68_face_landmarks.dat'
EXT = ['.mp4', '.mov', '.mpg']
NO_LANDMARKS = 68


def find_all_videos(dir, ext=EXT, relpath=False):
    # get the absolute path of the file
    abspath = os.path.abspath(dir)
    videofiles = []
    find_all_videos_impl(abspath, videofiles, ext)
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


class ImageFitter(object):
    def __init__(self, model):
        self.detector = load_dlib_frontal_face_detector()
        self.fitter = DlibWrapper(model)

    def fit_image(self, image):
        # Face detection
        bboxes = self.detector(image, image_diagonal=1000)

        # Check if at least one face was detected, otherwise throw a warning
        if len(bboxes) > 0:
            # Use the first bounding box (the most probable to represent a face) to initialise
            fitting_result = self.fitter.fit_from_bb(image, bboxes[0])

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


def process_video_worker(video_queue, face_model, input_dir, output_dir):
    image_fitter = ImageFitter(face_model)
    while True:
        video = video_queue.get()
        if video is None:
            print('[{}] Done...'.format(os.getpid()))
            video_queue.task_done()
            break
        relative_path = video[len(input_dir) + 1:]
        landmarkfile = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.csv')
        process_video(video, landmarkfile, image_fitter)
        video_queue.task_done()


def process_video(file, dest, fitter):
    if is_video(file):
        try:
            frames = mio.import_video(file, normalise=False)
        except IOError:
            warnings.warn('IO error reading video file {}, '.format(file) +
                          'the file may be corrupted or the video format is unsupported, skipping...')
        except ValueError as e:
            warnings.warn('Value Error reading video file {}, '.format(file) +
                          e.message)
            return
        # check if directory is non empty
        if os.path.dirname(dest):
            create_dir(os.path.dirname(dest))
        print('[{}] {} contains {} frames'.format(os.getpid(), file, len(frames)))
        print('[{}] writing landmarks to {}...'.format(os.getpid(), dest))
        frames = frames.map(fitter.fit_image)
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
                        landmarks = [-1] * NO_LANDMARKS*2
                    else:
                        lmg = frame.landmarks['final_shape']
                        landmarks = lmg['all'].points.reshape((NO_LANDMARKS*2,)).tolist()  # reshape to 136 points
                    fill_row(outwriter, i, landmarks)
            except Exception as _:
                warnings.warn('Runtime Error at frame {}'.format(i))
                print('initializing landmarks to -1s...')
                fill_row(outwriter, i, [-1] * NO_LANDMARKS*2)


def parse_options():
    options = dict()
    parser = argparse.ArgumentParser()
    options['model'] = '../config/shape_predictor_68_face_landmarks.dat'
    parser.add_argument('--input_dir', help='directory to search for videos, supported formats [.mov, .mpg, .mp4]')
    parser.add_argument('--output_dir', help='output directory to store the landmarks')
    parser.add_argument('--model', help='location of landmark model file. '
                                        'Default: ../config/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--file', help='perform landmarking on a single file')
    parser.add_argument('--output', help='output landmark file name, if not specified '
                                         'creates landmark file in current directory')
    parser.add_argument('--workers', help='number of workers to spawn. Default: number of CPUs available')
    args = parser.parse_args()
    if args.input_dir:
        options['input_dir'] = args.input_dir
    if args.output_dir:
        options['output_dir'] = args.output_dir
    if args.model:
        options['model'] = args.model
    if args.file:
        options['file'] = args.file
    if args.output:
        options['output'] = args.output
    if args.workers:
        options['workers'] = int(args.workers)
    return options


def add_tasks(queue, task_list):
    for t in task_list:
        queue.put(t)
    return queue


def add_poison_pills(queue, no_pills):
    for _ in range(no_pills):
        queue.put(None)
    return queue


if __name__ == '__main__':
    options = parse_options()

    if 'file' in options:
        video_file = options['file']
        video_file_basename = os.path.basename(video_file)
        print('Generating Landmarks from {}'.format(video_file))
        output = options['output'] if 'output' in options else os.path.splitext(video_file_basename)[0] + '.csv'
        process_video(video_file, output)
        exit()

    print('Generating Landmarks from {}'.format(options['input_dir']))
    videofiles = find_all_videos(options['input_dir'], relpath=False)
    videofiles.sort()
    print('Found {} video(s)...'.format(len(videofiles)))
    if 'workers' in options:
        no_workers = options['workers']
    else:
        no_workers = mp.cpu_count()
    queue = mp.JoinableQueue()
    queue = add_tasks(queue, videofiles)
    queue = add_poison_pills(queue, no_workers)
    input_dir = os.path.abspath(options['input_dir'])
    output_dir = os.path.abspath(options['output_dir'])
    workers = []
    print('Using {} workers...'.format(no_workers))
    for i in range(no_workers):
        workers.append(mp.Process(target=process_video_worker, args=(queue, options['model'], input_dir, output_dir)))
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    queue.join()
    print('All Done!')
