"""
preprocess the images
"""
import sys
import numpy as np
sys.path.append('../')
from utils.io import load_mat_file, save_mat
from utils.preprocessing import normalize_input, reorder_data, concat_first_second_deltas
from utils.preprocessing import sequencewise_mean_image_subtraction, compute_dct_features
from utils.plotting_utils import visualize_images


def main():

    data = load_mat_file('data/allMouthROIsResized_frontal.mat')
    X = data['dataMatrix'].astype('float32')
    vidlens = data['videoLengthVec'].reshape((-1,))
    # realign to 'C' Format
    X = reorder_data(X, (26, 44))
    X = sequencewise_mean_image_subtraction(X, vidlens)
    # samplewise normalize
    X = normalize_input(X, centralize=True)
    visualize_images(X[700:764], shape=(26, 44))
    data['dataMatrix'] = X
    save_mat(data, 'data/allMouthROIsMeanRemoved_frontal.mat')
    dct_feats = compute_dct_features(X, (26, 44), 30, method='zigzag')
    dct_data = dict()
    dct_data['dctFeatures'] = concat_first_second_deltas(dct_feats, vidlens)
    save_mat(dct_data, 'data/dctFeatMeanRemoved_OuluVs2.mat')

if __name__ == '__main__':
    main()
