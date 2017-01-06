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


def test_mergesamples():
    s = np.array([[1],[2],[3],[4],[1],[2],[3],[4],[1],[2],[3],[4],[1],[2],[3],[4],[5]])
    # s = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,5])
    l = [4,4,4,5]
    r = downsample(s, l, 3, 0)
    print(r)


def test_embed_temporal_info():
    s = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[1,1,1],[2,2,2],[3,3,3],[4,4,4],
                  [1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
    # s = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,5])
    l = np.array([4,4,4,5])
    r, l = downsample(s, l, 3, 0)
    r, l = embed_temporal_info(r, l, 3, 3)
    print(r)


def parse_options():
    options = dict()
    options['remove_mean'] = False
    options['diff_image'] = False
    options['samplewise_norm'] = False
    options['no_reorder'] = False
    options['merge_samples'] = False
    options['output'] = None
    options['delta_win'] = 9
    options['mergesize'] = 3
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_mean', action='store_true', help='remove mean image')
    parser.add_argument('--diff_image', action='store_true', help='compute difference of image')
    parser.add_argument('--samplewise_norm', action='store_true', help='samplewise normalize')
    parser.add_argument('--no_reorder', action='store_true', help='disable data reordering from f to c')
    parser.add_argument('--concat_deltas', action='store_true', help='concat 1st and 2nd deltas')
    parser.add_argument('--embed_temporal_info', help='embed temporal info to features [window],[step]. ie: 3,1')
    parser.add_argument('--output', help='write output to .mat file')
    parser.add_argument('--delta_win', help='size of delta window')
    parser.add_argument('input', nargs='+', help='input cuave .mat file to preprocess')
    args = parser.parse_args()
    if args.remove_mean:
        options['remove_mean'] = args.remove_mean
    if args.diff_image:
        options['diff_image'] = args.diff_image
    if args.samplewise_norm:
        options['samplewise_norm'] = args.samplewise_norm
    if args.embed_temporal_info:
        options['embed_temporal_info'] = args.embed_temporal_info
    if args.no_reorder:
        options['no_reorder'] = args.no_reorder
    if args.output:
        options['output'] = args.output
    if args.input:
        options['input'] = args.input[0]
    return options


def main():
    options = parse_options()
    data = load_mat_file(options['input'])
    data_matrix = data['dataMatrix'].astype('float32')
    vid_len_vec = data['videoLengthVec'].astype('int').reshape((-1,))
    targets_vec = data['targetsVec'].reshape((-1,))

    if not options['no_reorder']:
        data_matrix = reorder_data(data_matrix, (30, 50))
    if options['samplewise_norm']:
        data_matrix = normalize_input(data_matrix)
    if options['remove_mean']:
        data_matrix = sequencewise_mean_image_subtraction(data_matrix, vid_len_vec)
    if options['diff_image']:
        data_matrix = compute_diff_images(data_matrix, vid_len_vec)
    if 'embed_temporal_info' in options:
        window, step = tuple([int(d) for d in options['embed_temporal_info'].split(',')])
        data_matrix, targets_vec, vid_len_vec = downsample(data_matrix, targets_vec, vid_len_vec, window, 0)
        data_matrix, vid_len_vec = embed_temporal_info(data_matrix, vid_len_vec, window, step)

    data['dataMatrix'] = data_matrix

    if 'embed_temporal_info' in options:
        data['videoLengthVec'] = vid_len_vec
        data['targetsVec'] = targets_vec

    if 'output' in options:
        save_mat(data, options['output'])
    # print(data.keys())
    print('data prepared!')


if __name__ == '__main__':
    main()