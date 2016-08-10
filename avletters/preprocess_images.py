"""
realign images to c format from f format
"""
import sys
sys.path.append('../')
import argparse
from utils.io import load_mat_file, save_mat
from utils.preprocessing import resize_images, normalize_input, sequencewise_mean_image_subtraction, reorder_data
from utils.preprocessing import compute_dct_features, concat_first_second_deltas
from utils.preprocessing import compute_diff_images, apply_zca_whitening
from utils.plotting_utils import visualize_images


def resize(data):
    X = data['dataMatrix']
    vidlens = data['videoLengthVec'].reshape((-1,))
    X = resize_images(X)
    # X = apply_zca_whitening(X)
    visualize_images(X[800:864])
    dct_feats = compute_dct_features(X, (30, 40), 30, method='zigzag')
    dct_feats = concat_first_second_deltas(dct_feats, vidlens)
    X = normalize_input(X)
    data['dataMatrix'] = X
    save_mat(data, 'data/resized.mat')
    d = dict()
    d['dctFeatures'] = dct_feats
    save_mat(d, 'data/dctFeat_AVLetters.mat')


def remove_mean(data):
    X = data['dataMatrix'].astype('float32')
    vidlens = data['videoLengthVec'].reshape((-1,))
    X = resize_images(X)
    X = sequencewise_mean_image_subtraction(X, vidlens)
    # X = apply_zca_whitening(X)
    X_fortran = reorder_data(X, (30, 40), 'c', 'f')
    dct_feats = compute_dct_features(X, (30, 40), 30, method='zigzag')
    dct_feats = concat_first_second_deltas(dct_feats, vidlens)
    d = dict()
    d['dctFeatures'] = dct_feats
    save_mat(d, 'data/dctFeat_mean_removed_AVLetters.mat')
    visualize_images(X[800:864])
    # samplewise normalize
    X = normalize_input(X, centralize=True)
    data['dataMatrix'] = X
    data['dataMatrixF'] = X_fortran
    save_mat(data, 'data/resized_mean_removed.mat')


def diff_image(data):
    X = data['dataMatrix'].astype('float32')
    vidlens = data['videoLengthVec'].reshape((-1,))
    X = resize_images(X)
    X = apply_zca_whitening(X)
    # X = normalize_input(X)
    visualize_images(X[2000:2081])
    X = compute_diff_images(X, vidlens)
    X = apply_zca_whitening(X)
    X = normalize_input(X)
    visualize_images(X[2000:2081])
    data['dataMatrix'] = X
    save_mat(data, 'data/resized_diff_image_AVLetters.mat')


def parse_options():
    options = dict()
    options['operation'] = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', help='remove_mean, diff_image, resize')
    args = parser.parse_args()
    if args.operation:
        options['operation'] = args.operation
    return options


def main():
    options = parse_options()
    data = load_mat_file('data/allData_mouthROIs.mat')
    if options['operation'] == 'remove_mean':
        remove_mean(data)
    elif options['operation'] == 'diff_image':
        diff_image(data)
    elif options['operation'] == 'resize':
        resize(data)
    else:
        print('unknown operation')


if __name__ == '__main__':
    main()
