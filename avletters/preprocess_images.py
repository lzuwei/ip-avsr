"""
realign images to c format from f format
"""
import sys
import numpy as np
sys.path.append('../')
from utils.io import load_mat_file, save_mat
from utils.preprocessing import resize_images, normalize_input, sequencewise_mean_image_subtraction
from utils.plotting_utils import visualize_images
from scipy.fftpack import dct


def remove_mean_image(X, seqlens, savefilename=None):
    mean_removed = sequencewise_mean_image_subtraction(X, seqlens)
    return mean_removed


def main():
    data = load_mat_file('data/allData_mouthROIs.mat')
    X = data['dataMatrix'].astype('float32')
    vidlens = data['videoLengthVec'].reshape((-1,))
    # realign to 'C' Format
    X = resize_images(X)
    # remove mean
    X = remove_mean_image(X, vidlens)
    # samplewise normalize
    X = normalize_input(X, centralize=True)
    visualize_images(X[300:336], shape=(30, 40))
    data['dataMatrix'] = X
    save_mat(data, 'data/resized.mat')

if __name__ == '__main__':
    main()
