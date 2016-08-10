"""
preprocess the images
"""
import sys
sys.path.append('../')
import numpy as np
import argparse
from utils.io import load_mat_file, save_mat
from utils.preprocessing import normalize_input, reorder_data, concat_first_second_deltas
from utils.preprocessing import sequencewise_mean_image_subtraction, compute_dct_features, compute_diff_images
from utils.plotting_utils import visualize_images


def remove_mean(data):
    """
    remove mean image from data, sequence-wise mean removal,
    perform dct on images followed by normalization of input
    :param data: image data
    :return:
    """
    X = data['dataMatrix'].astype('float32')
    vidlens = data['videoLengthVec'].reshape((-1,))
    # realign to 'C' Format
    X = reorder_data(X, (26, 44))
    X = sequencewise_mean_image_subtraction(X, vidlens)
    X_fortran = reorder_data(X, (26, 44), 'c', 'f')
    visualize_images(X[700:764], shape=(26, 44))
    dct_feats = compute_dct_features(X, (26, 44), 30, method='zigzag')
    dct_data = dict()
    dct_data['dctFeatures'] = concat_first_second_deltas(dct_feats, vidlens)
    # samplewise normalize
    X = normalize_input(X, centralize=True)
    data['dataMatrix'] = X
    data['dataMatrixF'] = X_fortran
    save_mat(data, 'data/allMouthROIsMeanRemoved_frontal.mat')
    save_mat(dct_data, 'data/dctFeatMeanRemoved_OuluVs2.mat')


def diff_image(data):
    X = data['dataMatrix'].astype('float32')
    vidlens = data['videoLengthVec'].reshape((-1,))
    # realign to 'C' Format
    X = reorder_data(X, (26, 44))
    X = compute_diff_images(X, vidlens)
    data['dataMatrix'] = X
    save_mat(data, 'data/allMouthROIsDiffImage_frontal.mat')


def parse_options():
    options = dict()
    options['operation'] = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', help='remove_mean, diff_image')
    args = parser.parse_args()
    if args.operation:
        options['operation'] = args.operation
    return options


def main():
    options = parse_options()
    data = load_mat_file('data/allMouthROIsResized_frontal.mat')
    if options['operation'] == 'remove_mean':
        remove_mean(data)
    elif options['operation'] == 'diff_image':
        diff_image(data)


if __name__ == '__main__':
    main()
