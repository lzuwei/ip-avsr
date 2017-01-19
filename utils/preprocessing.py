# Preprocessing scripts for AV Letters Dataset

import math
import numpy as np
import numpy.matlib as matlab
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.misc import imresize


def test_delta():
    a = np.array([[1,1,1,1,1,1,1,1,10], [2,2,2,2,2,2,2,2,20], [3,3,3,3,3,3,3,3,30], [4,4,4,4,4,4,4,4,40]])
    aa = deltas(a, 9)
    print(aa)


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


def split_data(data_matrix, split_idx, len_vec=None):
    """
    split the data according to the split index provided
    :param data_matrix: numpy array of data to split
    :param split_idx: an integer list of indexes to extract
    :param len_vec: for sequences, split the data according to the sequence length provided
    :return: split data
    """
    assert len(split_idx) == len(data_matrix)
    if len_vec is None:
        return data_matrix[split_idx]
    else:
        # compute the cumulative total sum of the lec_vec
        size_split = np.sum(len_vec[split_idx])
        split = np.zeros(data_matrix.shape[:-1] + (size_split,), dtype=data_matrix.dtype)
        offset = np.zeros((len(len_vec)))
        for i in range(1, len(len_vec)):
            offset[i] = offset[i - 1] + len_vec[i]
        for i, idx in enumerate(split_idx):
            split[i] = data_matrix[idx:len_vec[idx]]
        return split


def split_seq_data(X, y, subjects, video_lens, train_ids, val_ids, test_ids):
    """
    Splits the data into training and testing sets
    :param X: input X
    :param y: target y
    :param subjects: array of video -> subject mapping
    :param video_lens: array of video lengths for each video
    :param train_ids: list of subject ids used for training
    :param val_ids: list of subject ids used for validation
    :param test_ids: list of subject ids used for testing
    :return: split data
    """
    # construct a subjects data matrix offset
    X_feature_dim = X.shape[1]
    train_X = np.empty((0, X_feature_dim), dtype='float32')
    val_X = np.empty((0, X_feature_dim), dtype='float32')
    test_X = np.empty((0, X_feature_dim), dtype='float32')
    train_y = np.empty((0,), dtype='int')
    val_y = np.empty((0,), dtype='int')
    test_y = np.empty((0,), dtype='int')
    train_vidlens = np.empty((0,), dtype='int')
    val_vidlens = np.empty((0,), dtype='int')
    test_vidlens = np.empty((0,), dtype='int')
    train_subjects = np.empty((0,), dtype='int')
    val_subjects = np.empty((0,), dtype='int')
    test_subjects = np.empty((0,), dtype='int')
    previous_subject = 1
    subject_video_count = 0
    current_video_idx = 0
    current_data_idx = 0
    populate = False
    for idx, subject in enumerate(subjects):
        if previous_subject == subject:  # accumulate
            subject_video_count += 1
        else:  # populate the previous subject
            populate = True
        if idx == len(subjects) - 1:  # check if it is the last entry, if so populate
            populate = True
            previous_subject = subject
        if populate:
            # slice the data into the respective splits
            end_video_idx = current_video_idx + subject_video_count
            subject_data_len = int(np.sum(video_lens[current_video_idx:end_video_idx]))
            end_data_idx = current_data_idx + subject_data_len
            if previous_subject in train_ids:
                train_X = np.concatenate((train_X, X[current_data_idx:end_data_idx]))
                train_y = np.concatenate((train_y, y[current_data_idx:end_data_idx]))
                train_vidlens = np.concatenate((train_vidlens, video_lens[current_video_idx:end_video_idx]))
                train_subjects = np.concatenate((train_subjects, subjects[current_video_idx:end_video_idx]))
            elif previous_subject in val_ids:
                val_X = np.concatenate((val_X, X[current_data_idx:end_data_idx]))
                val_y = np.concatenate((val_y, y[current_data_idx:end_data_idx]))
                val_vidlens = np.concatenate((val_vidlens, video_lens[current_video_idx:end_video_idx]))
                val_subjects = np.concatenate((val_subjects, subjects[current_video_idx:end_video_idx]))
            else:
                test_X = np.concatenate((test_X, X[current_data_idx:end_data_idx]))
                test_y = np.concatenate((test_y, y[current_data_idx:end_data_idx]))
                test_vidlens = np.concatenate((test_vidlens, video_lens[current_video_idx:end_video_idx]))
                test_subjects = np.concatenate((test_subjects, subjects[current_video_idx:end_video_idx]))
            previous_subject = subject
            current_video_idx = end_video_idx
            current_data_idx = end_data_idx
            subject_video_count = 1
            populate = False
    return train_X, train_y, train_vidlens, train_subjects, \
           val_X, val_y, val_vidlens, val_subjects, \
           test_X, test_y, test_vidlens, test_subjects


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
    """
    samplewise normalize input
    :param input: input features
    :param centralize: apply 0 mean, std 1
    :param quantize: rescale values to fall between 0 and 1
    :return: normalized input
    """
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
    """
    sequence-wise mean image removal
    :param input: input sequences
    :param seqlens: sequence lengths
    :param axis: axis to apply mean image removal
    :return: mean removed input sequences
    """
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


def fill_zigzag(shape):
    """
        fills a 2D array in a zigzag fashion in the following sequence:

        [[1, 2, 6, 7],
         [3, 5, 8, 11],
         [4, 9, 10, 12]]

        :param shape: shape of 2D array to return
        :return: 2D array containing elements of X arranged in the traversal sequence
        """
    rows, cols = shape
    size = rows * cols
    out = np.zeros((rows, cols,), dtype=int)
    cur_row = 0
    cur_col = 0
    direction = 0
    DOWN = 1
    UP = 0

    for i in range(size):
        out[cur_row][cur_col] = i + 1
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
    X_dct = fft.dct(X, norm='ortho')

    if method == 'zigzag':
        out = np.zeros((len(X_dct), no_coeff), dtype=X_dct.dtype)
        for i in xrange(len(X_dct)):
            image = X_dct[i].reshape(image_shape)
            out[i] = zigzag(image)[1:no_coeff + 1]
        return out
    elif method == 'rel_variance':
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
        return X_dct[:, idxs]
    elif method == 'variance':
        X_dct = X_dct[:, 1:]
        # find standard deviation for each frequency component
        std_dct = np.std(X_dct, 0)
        # sort by largest variance
        idxs = np.argsort(std_dct)[::-1][:no_coeff]
        # return DCT coefficients with the largest variance
        return X_dct[:, idxs]
    elif method == 'energy':
        X_dct = X_dct[:, 1:]
        X_sum = np.abs(X_dct)
        X_sum = np.sum(X_sum, 0)
        idxs = np.argsort(X_sum)[::-1][:no_coeff]
        return X_dct[:, idxs]
    else:
        raise NotImplementedError("method not implemented, use only 'zigzag', 'variance', 'rel_variance")


def concat_first_second_deltas(X, vidlenvec, w=9):
    """
    Compute and concatenate 1st and 2nd order derivatives of input X given a sequence list
    :param X: input feature vector X
    :param vidlenvec: temporal sequence of X
    :param w: window size, defaults to 9
    :return: A matrix of shape(num rows of intput X, X + 1st order X + 2nd order X)
    """
    # construct a new feature matrix
    feature_len = X.shape[1]
    Y = np.zeros((X.shape[0], feature_len * 3))  # new feature vector with 1st, 2nd delta
    start = 0
    for vidlen in vidlenvec:
        end = start + vidlen
        seq = X[start: end]  # (vidlen, feature_len)
        first_order = deltas(seq.T, w)
        second_order = deltas(first_order, w)
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


def compute_diff_images(X, vidlenvec):

    diff_X = np.zeros(X.shape, dtype=X.dtype)
    start = 0
    for l in vidlenvec:
        end = start + l
        seq = X[start:end]
        diff_seq = np.diff(seq, 1, 0)
        diff_X[start] = diff_seq[0]  # copy the 1st diff image
        diff_X[start+1:end] = diff_seq
        start = end
    return diff_X


def zca_whiten(inputs):
    sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # Correlation matrix
    U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
    epsilon = 0.1  # Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)  # Data whitening


def apply_zca_whitening(X):
    for i, img in enumerate(X):
        X[i] = zca_whiten(img.reshape((1, -1)))
    return X


def factorize(inputs, targets, input_len, multipleof, axis_to_delete=None):
    """
    factorize inputs to a factor of a given multiple
    :param inputs: input data vector arranged as (input size, feature len)
    :param targets: targets data vector
    :param input_len: input data vector original length of shape (1,)
    :param multipleof: given multiple to factorize input
    :return: merged samples of shape (input size / merge size, merged features)
    """
    # if 1 dimension, reshape to 2 dim array
    if len(inputs.shape) < 2:
        inputs = inputs.reshape((-1, 1))
    idx_to_remove = []
    curr_idx = 0
    for l in input_len:
        end_idx = curr_idx + l
        remainder = l % multipleof
        # randomly remove items if it is not divisible by merge size
        idx_to_remove += np.random.permutation(range(curr_idx, end_idx))[:remainder].tolist()
        curr_idx += l
    input_len = input_len - (input_len % multipleof)
    return np.delete(inputs, idx_to_remove, axis=axis_to_delete),\
           np.delete(targets, idx_to_remove, axis=axis_to_delete), input_len


def embed_temporal_info(X, targets, X_len, window, step):
    """
    first downsample input to multiple of step
    repeat head and tail according to window
    embed features based on window
    [][.][]|[][.][]|[][.][]|[][.][]
    {}{}[][.][][][.][][][.][][][.][]{}{}
    {}{}{}{}{}[][.][][][.][][][.][][][.][]{}{}{}{}{}
    win = 1, step = 3, repeats = win - step + ceil(step/2) = 1 - 3 + ceil(3/2) = 0
    win = 3, step = 3, repeats = 3 - 3 + ceil(step/2) = 3 - 3 + 2 = 2
    win = 6, step = 3, repeats = 6 - 3 + ceil(step/2) = 6 - 3 + 2 = 5
    startpos = floor(step/2) = floor(3/2) = 1
    :param X: input matrix in the shape (feature no, feature size)
    :param targets: target matrix
    :param X_len: lengths of each sequence
    :param window: temporal window to embed features
    :param step: step size to move per temporal feature
    :return:
    """
    embedsize = X.shape[-1] * (window*2 + 1)
    res = np.zeros((np.sum(X_len)/step, embedsize), dtype=X.dtype)
    res_targets = np.zeros((np.sum(X_len)/step,), dtype=targets.dtype)
    # select the sequence
    curr_idx = 0
    res_iter = 0
    for l in X_len:
        end_idx = curr_idx + l
        seq = X[curr_idx:end_idx]
        seq_target = targets[curr_idx:end_idx]
        repeats = int(window - step + math.ceil(step/2.0))  # compute the number of elements to repeat
        # extend the sequence in the head and tail based on the number of repeats
        seq = np.concatenate((np.repeat(seq[:1, :], repeats, axis=0),
                              seq,
                              np.repeat(seq[-1:, :], repeats, axis=0)), axis=0)
        # compute the starting position of the sequence to embed temporal info
        startpos = repeats + (step/2)
        # iterate through the sequence and embed temporal information, stop at the end of the original sequence len
        while startpos - repeats < l:
            temporal_feature = seq[startpos - window: startpos + window + 1].reshape((-1,))
            res[res_iter] = temporal_feature
            res_targets[res_iter] = seq_target[0]
            startpos += step
            res_iter += 1
        curr_idx += l  # move to next sequence
    res_len = X_len / step
    return res, res_targets, res_len


def force_align(x1, x2, mode='fill'):
    """
    Force Align 2 streams of data to equal lengths
    :param x1: stream 1 tuple consisting (X, targets, sequence lengths)
    :param x2: stream 2 tuple consisting (X, targets, sequence lengths)
    :param mode: 'fill', 'discard'
    :return: x1, x2 streams forced aligned
    """
    x1, x1_targets, x1_lens = x1
    x2, x2_targets, x2_lens = x2
    x1_new = []
    x1_targets_new = []
    x2_new = []
    x2_targets_new = []
    x1_curr_idx = 0
    x2_curr_idx = 0
    for i, l1 in enumerate(x1_lens):
        l2 = x2_lens[i]
        difference = l1 - l2
        if mode == 'fill':
            if difference < 0:
                '''fill x1 with difference'''
                difference = abs(difference)
                for j in range(l1):
                    x1_new.append(x1[x1_curr_idx + j])
                    x1_targets_new.append(x1_targets[x1_curr_idx + j])
                '''[0][1][2][3]'''
                last_element = x1[x1_curr_idx + l1 - 1]
                last_element_target = x1_targets[x1_curr_idx + l1 - 1]
                for j in range(difference):
                    x1_new.append(np.copy(last_element))
                    x1_targets_new.append(np.copy(last_element_target))
                x1_lens[i] = l1 + difference  # update the lens
                for j in range(l2):
                    x2_new.append(x2[x2_curr_idx + j])
                    x2_targets_new.append(x2_targets[x2_curr_idx + j])
            else:
                '''fill x2 with difference'''
                for j in range(l2):
                    x2_new.append(x2[x2_curr_idx + j])
                    x2_targets_new.append(x2_targets[x2_curr_idx + j])
                '''[0][1][2][3]'''
                last_element = x2[x2_curr_idx + l1 - 1]
                last_element_target = x2_targets[x2_curr_idx + l2 - 1]
                for j in range(difference):
                    x2_new.append(np.copy(last_element))
                    x2_targets_new.append(np.copy(last_element_target))
                x2_lens[i] = l2 + difference  # update the lens
                for j in range(l1):
                    x1_new.append(x1[x1_curr_idx + j])
                    x1_targets_new.append(x1_targets[x1_curr_idx + j])
            x1_curr_idx += l1
            x2_curr_idx += l2
        # TODO: discard mode
    return (np.array(x1_new), np.array(x1_targets_new), x1_lens), (np.array(x2_new), np.array(x2_targets_new), x2_lens)
