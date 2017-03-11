from __future__ import print_function
import sys
sys.path.insert(0, '../')
import argparse
from utils.preprocessing import *
from utils.io import *
from utils.plotting_utils import *


def parse_options():
    options = dict()
    options['remove_mean'] = False
    options['diff_image'] = False
    options['samplewise_norm'] = False
    options['merge_samples'] = False
    options['output'] = None
    options['mergesize'] = 3
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_mean', action='store_true', help='remove mean image')
    parser.add_argument('--diff_image', action='store_true', help='compute difference of image')
    parser.add_argument('--samplewise_norm', action='store_true', help='samplewise normalize')
    parser.add_argument('--reorder_data', help='redorder data from f to c convention. eg: 30,50')
    parser.add_argument('--concat_deltas', help='concat 1st and 2nd deltas, default delta window: 2')
    parser.add_argument('--embed_temporal_info', help='embed temporal info to features [window],[step]. ie: 3,1')
    parser.add_argument('--output', help='write output to .mat file')
    parser.add_argument('input', nargs='+', help='input data .mat file to preprocess')
    args = parser.parse_args()
    if args.remove_mean:
        options['remove_mean'] = args.remove_mean
    if args.diff_image:
        options['diff_image'] = args.diff_image
    if args.samplewise_norm:
        options['samplewise_norm'] = args.samplewise_norm
    if args.embed_temporal_info:
        options['embed_temporal_info'] = args.embed_temporal_info
    if args.reorder_data:
        options['reorder_data'] = args.reorder_data
    if args.output:
        options['output'] = args.output
    if args.input:
        options['input'] = args.input[0]
    if args.concat_deltas:
        options['concat_deltas'] = int(args.concat_deltas)
    return options


def main():
    options = parse_options()
    data = load_mat_file(options['input'])
    data_matrix = data['dataMatrix'].astype('float32')
    vid_len_vec = data['videoLengthVec'].astype('int').reshape((-1,))
    targets_vec = data['targetsVec'].reshape((-1,))

    if 'reorder_data' in options:
        imagesize = tuple([int(d) for d in options['reorder_data'].split(',')])
        data_matrix = reorder_data(data_matrix, imagesize)
    if options['samplewise_norm']:
        data_matrix = normalize_input(data_matrix)
    if options['remove_mean']:
        data_matrix = sequencewise_mean_image_subtraction(data_matrix, vid_len_vec)
    if options['diff_image']:
        data_matrix = compute_diff_images(data_matrix, vid_len_vec)
    if 'embed_temporal_info' in options:
        window, step = tuple([int(d) for d in options['embed_temporal_info'].split(',')])
        data_matrix, targets_vec, vid_len_vec = factorize(data_matrix, targets_vec, vid_len_vec, step, 0)
        data_matrix, targets_vec, vid_len_vec = embed_temporal_info(data_matrix, targets_vec, vid_len_vec, window, step)
    if 'concat_deltas' in options:
        data_matrix = concat_first_second_deltas(data_matrix, vid_len_vec, options['concat_deltas'])

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