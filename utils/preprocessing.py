# Preprocessing scripts for AV Letters Dataset

import numpy as np
import numpy.matlib as matlab
import scipy.signal as signal
import scipy.fftpack as fft
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


def sequencewise_mean_image_subtraction(input, seqlens, axis=0):
    mean_subtracted = np.zeros(input.shape, input.dtype)
    start = 0
    end = 0
    for len in seqlens:
        end += len
        seq = input[start:end, :]
        mean_image = np.sum(seq, axis, input.dtype) / len
        mean_subtracted[start:end, :] = seq - mean_image
        start += len
    return mean_subtracted


def zigzag(X):
    """
    traverses a 2D array in a zigzag fashion in the following sequence:

    [[1, 2, 6, 7],
     [3, 5, 8, 11],
     [4, 9, 10, 12]]

    :param X: 2D array
    :return: 1D array containing elements of X arranged in the traversal sequence
    """
    rows, cols = X.shape
    size = rows * cols
    out = np.zeros((rows * cols, ), dtype=X.dtype)
    cur_row = 0
    cur_col = 0
    direction = 0
    DOWN = 1
    UP = 0

    for i in range(size):
        out[i] = X[cur_row][cur_col]
        if cur_row == 0:
            if cur_col % 2:  # odd, move diagonal down
                direction = DOWN
                cur_row += 1
                cur_col -= 1
            else:
                if cur_col == cols - 1:  # no more cols to move, go down
                    direction = DOWN
                    cur_row += 1
                else:
                    cur_col += 1  # even, move right
        elif cur_col == 0:
            if cur_row % 2:
                if cur_row == rows - 1:  # no more rows to move down, move right
                    direction = UP
                    cur_col += 1
                else:
                    cur_row += 1
            else:
                direction = UP
                cur_row -= 1
                cur_col += 1
        elif direction == UP:
            if cur_col == cols - 1:  # no more cols to move up
                direction = DOWN
                cur_row += 1
            else:
                cur_row -= 1
                cur_col += 1
        else:
            if cur_row == rows - 1:  # no more rows to move down
                direction = UP
                cur_col += 1
            else:
                cur_row += 1
                cur_col -= 1
    return out


def test_zigzag():
    X = np.array([[1, 2, 6, 7],
                  [3, 5, 8, 11],
                  [4, 9, 10, 12]])

    Y = np.array([[1, 2, 5, 6, 9, 10],
                  [3, 4, 7, 8, 11, 12]])

    res = zigzag(X)
    # check if results are sorted, compare each element with itself after
    assert all(res[i] < res[i + 1] for i in range(len(res) - 1))
    res = zigzag(Y)
    assert all(res[i] < res[i + 1] for i in range(len(res) - 1))


def compute_dct_features(X, image_shape, no_coeff=30, method='zigzag'):
    """
    compute 2D-dct features of a given image.
    Type 2 DCT and finds the DCT coefficents with the largest mean normalized variance
    :param X: 1 dimensional input image in 'c' format
    :param image_shape: image shape
    :param no_coeff: number of coefficients to extract
    :param method: method to extract coefficents, zigzag, variance
    :return: dct features
    """
    # strip highest freq as it is the mean intensity
    X_dct = fft.dct(X.reshape((-1,) + image_shape).reshape((-1,) + (image_shape[0] * image_shape[1],)))

    if method == 'zigzag':
        out = np.zeros((len(X_dct), no_coeff), dtype=X_dct.dtype)
        for i in xrange(len(X_dct)):
            image = X_dct[i].reshape(image_shape)
            out[i] = zigzag(image)[1:no_coeff + 1]
        return out
    elif method == 'variance':
        X_dct = X_dct[:, 1:]
        # mean coefficient per frequency
        mean_dct = np.mean(X_dct, 0)
        # mean normalize
        mean_norm_dct = X_dct - mean_dct
        # find standard deviation for each frequency component
        std_dct = np.std(mean_norm_dct, 0)
        # sort by largest variance
        idxs = np.argsort(std_dct)[::-1][:no_coeff]
        # return DCT coefficients with the largest variance
        return mean_norm_dct[:, idxs]
    else:
        raise NotImplementedError("method not implemented, use only 'zigzag', 'variance'")


def concat_first_second_deltas(X, vidlenvec):
    """
    Compute and concatenate 1st and 2nd order derivatives of input X given a sequence list
    :param X: input feature vector X
    :param vidlenvec: temporal sequence of X
    :return: A matrix of shape(num rows of intput X, X + 1st order X + 2nd order X)
    """
    # construct a new feature matrix
    feature_len = X.shape[1]
    Y = np.zeros((X.shape[0], feature_len * 3))  # new feature vector with 1st, 2nd delta
    start = 0
    for vidlen in vidlenvec:
        end = start + vidlen
        seq = X[start: end]  # (vidlen, feature_len)
        first_order = deltas(seq.T)
        second_order = deltas(first_order)
        assert first_order.shape == (feature_len, vidlen)
        assert second_order.shape == (feature_len, vidlen)
        assert len(seq) == vidlen
        seq = np.concatenate((seq, first_order.T, second_order.T), axis=1)
        for idx, j in enumerate(range(start, end)):
            Y[j] = seq[idx]
        start += vidlen
    return Y


def reorder_data(X, shape, orig_order='f', desired_order='c'):
    """
    reorder data arranged from 2D from one order format to another
    :param X: 1D input aligned in orig_order
    :param shape: 2D shape of input collapsed from
    :param orig_order: original order of packing
    :param desired_order: desired order of packing
    :return: realigned 1D input
    """
    d1, d2 = shape
    X = X.reshape((-1, d1, d2), order=orig_order).reshape((-1, d1 * d2), order=desired_order)
    return X