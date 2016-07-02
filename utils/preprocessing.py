# Preprocessing scripts for AV Letters Dataset

import numpy as np
import numpy.matlib as matlab
import scipy.io as sio
import scipy.signal as signal
from scipy.misc import imresize


def deltas(x, w=9):
    """
    Calculate the deltas (derivatives) of a sequence
    Use a W-point window (W odd, default 9) to calculate deltas using a
    simple linear slope.  This mirrors the delta calculation performed
    in feacalc etc.  Each row of X is filtered separately.

    Notes:
    x is your data matrix where each feature corresponds to a row (so you may have to transpose the data you
    pass as an argument) and then transpose the output of the function).

    :param x: x is your data matrix where each feature corresponds to a row.
        (so you may have to transpose the data you pass as an argument)
        and then transpose the output of the function)
    :param w: window size, defaults to 9
    :return: derivatives of a sequence
    """
    # compute the shape of the input
    num_row, num_cols = x.shape

    # define window shape
    hlen = w // 2  # floor integer divide
    w = 2 * hlen + 1  # odd number
    win = np.arange(hlen, -hlen - 1, -1, dtype=np.float32)

    # pad the data by repeating the first and last columns
    a = matlab.repmat(x[:, 1], 1, hlen).reshape((num_row, hlen), order='F')
    b = matlab.repmat(x[:, -1], 1, hlen).reshape((num_row, hlen), order='F')
    xx = np.concatenate((a, x, b), axis=1)

    # apply the delta filter, see matlab 1D filter
    d = signal.lfilter(win, 1, xx, 1)

    # trim the edges
    return d[:, hlen*2: hlen*2 + num_cols]


def load_mat_file(path):
    """
    Loads .mat file
    :param path: path to .mat file
    :return: dictionary containing .mat data
    """
    return sio.loadmat(path)


def create_split_index(data_len, vid_len_vec, iter_vec):
    """
    Creates a test split index for the data matrix based on
    the iteration vector and video length vector
    :param data_len: len of the data matrix
    :param vid_len_vec: Video Length Vector
    :param iter_vec: Iteration Vector
    :return: an index vector containing the test and training split indexes
    """
    indexes = np.zeros((data_len,), dtype=bool)
    # populate the iterVec to iteration matrix of all the frames
    start = 0
    for vid in range(len(iter_vec)):
        end = start + vid_len_vec[vid][0]
        if iter_vec[vid] == 1 or iter_vec[vid] == 2:
            indexes[start:end] = True
        else:
            indexes[start:end] = False
        start = end
    return indexes


def split_videolen(videolen_vec, iter_vec):
    train_vidlen = []
    test_vidlen = []
    for idx, iter in enumerate(iter_vec):
        if iter == 1 or iter == 2:
            train_vidlen.append(videolen_vec[idx][0])
        else:
            test_vidlen.append(videolen_vec[idx][0])
    return train_vidlen, test_vidlen


def resize_img(img, orig_dim=(60, 80), dim=(30, 40), reshape=True, order='F'):
    """
    Resizes the image to new dimensions
    :param img: image to resize
    :param orig_dim: original dimension of image
    :param dim: new dimension of image
    :param reshape: 1-D to 2-D required
    :param order: 'C' order or 'F' order
    :return: resized image
    """
    if reshape:
        img = img.reshape(orig_dim, order=order)
    return imresize(img, dim)


def resize_images(images, orig_dim=(60, 80), dim=(30, 40), reshape=True, order='F'):
    """
    resize a data matrix consisting of multiple images
    :param images: images to resize
    :param orig_dim: original dimension of images
    :param dim: new dimension of images
    :param reshape: 1-D to 2-D reshaping required
    :param order: 'C' order or 'F' order
    :return: resized images back in original shape
    """
    if reshape:
        resized = np.zeros((images.shape[0], dim[0] * dim[1]))
    else:
        resized = np.zeros((images.shape[0], dim[0], dim[1]))
    for i, img in enumerate(images):
        if reshape:
            # Note reshaped into C format, instead of F format
            resized[i] = resize_img(img, orig_dim, dim, reshape, order).reshape((dim[0]*dim[1],))
        else:
            resized[i] = resize_img(img, orig_dim, dim, reshape, order)
    return resized


def normalize_input(input, centralize=True, quantize=False):

    def center(item):
        item = item - item.mean()
        item = item / np.std(item)
        return item

    def rescale(item):
        min = np.min(item)
        max = np.max(item)
        item = (item - min) / (max - min)
        return item

    for i, item in enumerate(input):
        if centralize:
            input[i] = center(item)
        if quantize:
            input[i] = rescale(item)
    return input


def featurewise_normalize_sequence(input):
    """
    feature-wise z-normalize an input
    :param input: an input matrix of shape (input no, feature)
    :return: featurewise z-normalized input, feature mean, feature std
    """
    feature_means = np.mean(input, axis=0)
    assert feature_means.shape == (input.shape[1],)
    input = input - feature_means
    feature_std = np.std(input, axis=0)
    assert feature_std.shape == (input.shape[1],)
    input = input / feature_std
    return input, feature_means, feature_std


def save_mat(train, test, path):
    dic = {}
    dic['trainDataResized'] = train
    dic['testDataResized'] = test

    print('save matlab file...')
    sio.savemat(path, dic)