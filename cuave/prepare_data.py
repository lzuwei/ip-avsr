from __future__ import print_function
import sys
sys.path.insert(0, '../')
import argparse
import utils.ffmpeg
from utils.preprocessing import *
from utils.io import *
from utils.plotting_utils import *


def parse_htk_labels(filename):
    """
    #Normal in 100ns
    7800000 14480000 zero
    17510000 22920000 one
    26580000 32630000 two
    36290000 40590000 three
    46240000 49900000 four
    55310000 59370000 five
    63590000 69800000 six
    ...
    #Moving

    :param filename:
    :return:
    """
    labels = []
    with open(filename, 'r') as f:
        line = f.readline()[:-1]
        if 'Normal' in line:
            while True:
                # iterate until #Moving
                line = f.readline()
                if '#Moving' in line:
                    break
                else:
                    start, end, number = line[:-2].split(' ')  # remove \n\r
                    labels.append((start, end, number))
    return labels


def to_100ns(time_in_sec):
    return int(time_in_sec * 10000000)


def digit_to_int(digit):
    digit_map = {'zero': 0,
                 'one': 1,
                 'two': 2,
                 'three': 3,
                 'four': 4,
                 'five': 5,
                 'six': 6,
                 'seven': 7,
                 'eight': 8,
                 'nine': 9}
    return digit_map[digit]


def segment_video(video_file, label_file):
    _, video_frames = utils.ffmpeg.ffprobe_video(video_file)
    htk_labels = parse_htk_labels(label_file)
    print('number of video frames: {}'.format(len(video_frames)))
    print('number of labels: {}'.format(len(htk_labels)))
    current_frame = 0
    idxes = []
    seq_lens = []
    labels = []
    for start, end, label in htk_labels:
        start = int(start)
        end = int(end)
        number = digit_to_int(label)
        # print(start, end, number)
        seq_len = 0
        while True:
            f = video_frames[current_frame]
            pts_time = to_100ns(f.pkt_pts_time)
            # check if frame is withing utterance window
            if pts_time > start and pts_time <= end:
                idxes.append(current_frame)
                labels.append(number)
                seq_len += 1
                current_frame += 1
                # TODO: extract/select mouth ROI of frame
            else:
                if pts_time > end:
                    break
                current_frame += 1  # keep moving to the start of the next sequence
        seq_lens.append(seq_len)
    print(len(idxes))
    print(len(labels))
    print(seq_lens)


def reorder_images(train, val, test):
    train = reshape_images_order(train, (30, 50))
    val = reshape_images_order(val, (30, 50))
    test = reshape_images_order(test, (30, 50))
    return train, val, test


def samplewise_normalize(train, val, test):
    train = normalize_input(train)
    val = normalize_input(val)
    test = normalize_input(test)
    return train, val, test


def remove_mean(train, val, test, train_vid_lens, val_vid_lens, test_vid_lens):
    train = sequencewise_mean_image_subtraction(train, train_vid_lens)
    val = sequencewise_mean_image_subtraction(val, val_vid_lens)
    test = sequencewise_mean_image_subtraction(test, test_vid_lens)
    return train, val, test


def diff_image(train, val, test, train_vid_lens, val_vid_lens, test_vid_lens):
    train = compute_diff_images(train, train_vid_lens)
    val = compute_diff_images(val, val_vid_lens)
    test = compute_diff_images(test, test_vid_lens)
    return train, val, test


def parse_options():
    options = dict()
    options['remove_mean'] = False
    options['diff_image'] = False
    options['samplewise_norm'] = False
    options['no_reorder'] = False
    options['output'] = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_mean', action='store_true', help='remove mean image')
    parser.add_argument('--diff_image', action='store_true', help='compute difference of image')
    parser.add_argument('--samplewise_norm', action='store_true', help='samplewise normalize')
    parser.add_argument('--no_reorder', action='store_true', help='disable data reordering from f to c')
    parser.add_argument('--output', help='write output to .mat file')
    parser.add_argument('input', nargs='+', help='input cuave .mat file to preprocess')
    args = parser.parse_args()
    if args.remove_mean:
        options['remove_mean'] = args.remove_mean
    if args.diff_image:
        options['diff_image'] = args.diff_image
    if args.samplewise_norm:
        options['samplewise_norm'] = args.samplewise_norm
    if args.no_reorder:
        options['no_reorder'] = args.no_reorder
    if args.output:
        options['output'] = args.output
    if args.input:
        options['input'] = args.input[0]
    return options


def main():
    options = parse_options()
    # data = load_mat_file('data/allData_mouthROIs_basedOnMouthCenter_trValTestSets.mat')
    data = load_mat_file(options['input'])
    X_train = data['trData'].astype('float32')
    X_val = data['valData'].astype('float32')
    X_test = data['testData'].astype('float32')
    train_vid_lens = data['trVideoLengthVec'].reshape((-1,))
    val_vid_lens = data['valVideoLengthVec'].reshape((-1,))
    test_vid_lens = data['testVideoLengthVec'].reshape((-1,))

    if not options['no_reorder']:
        X_train, X_val, X_test = reorder_images(X_train, X_val, X_test)
    if options['samplewise_norm']:
        X_train, X_val, X_test = samplewise_normalize(X_train, X_val, X_test)
    if options['remove_mean']:
        X_train, X_val, X_test = remove_mean(X_train, X_val, X_test, train_vid_lens, val_vid_lens, test_vid_lens)
    if options['diff_image']:
        X_train, X_val, X_test = diff_image(X_train, X_val, X_test, train_vid_lens, val_vid_lens, test_vid_lens)

    visualize_images(X_test[500:536], (30, 50))

    data['trData'] = X_train
    data['valData'] = X_val
    data['testData'] = X_test
    if options['output']:
        save_mat(data, options['output'])
    # print(data.keys())
    print('data prepared!')


if __name__ == '__main__':
    main()