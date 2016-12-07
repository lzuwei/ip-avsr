from __future__ import print_function
import sys
sys.path.insert(0, '../')
import time
import ConfigParser
import argparse

import matplotlib
matplotlib.use('Agg')  # Change matplotlib backend, in case we have no X server running..

from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from utils.io import *
from utils.regularization import early_stop2

import theano.tensor as T
import theano

import lasagne as las
import numpy as np
from lasagne.updates import adam

from modelzoo import lstm_classifier_baseline
from utils.plotting_utils import print_network


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def split_data(X, y, dct, subjects, video_lens, train_ids, val_ids, test_ids):
    """
    Splits the data into training and testing sets
    :param X: input X
    :param y: target y
    :param dct: dct features
    :param subjects: array of video -> subject mapping
    :param video_lens: array of video lengths for each video
    :param train_ids: list of subject ids used for training
    :param val_ids: list of subject ids used for validation
    :param test_ids: list of subject ids used for testing
    :return: split data
    """
    # construct a subjects data matrix offset
    X_feature_dim = X.shape[1]
    dct_dim = dct.shape[1]
    train_X = np.empty((0, X_feature_dim), dtype='float32')
    val_X = np.empty((0, X_feature_dim), dtype='float32')
    test_X = np.empty((0, X_feature_dim), dtype='float32')
    train_y = np.empty((0,), dtype='int')
    val_y = np.empty((0,), dtype='int')
    test_y = np.empty((0,), dtype='int')
    train_dct = np.empty((0, dct_dim), dtype='float32')
    val_dct = np.empty((0, dct_dim), dtype='float32')
    test_dct = np.empty((0, dct_dim), dtype='float32')
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
            previous_subject = subject
            populate = True
        if populate:
            # slice the data into the respective splits
            end_video_idx = current_video_idx + subject_video_count
            subject_data_len = int(np.sum(video_lens[current_video_idx:end_video_idx]))
            end_data_idx = current_data_idx + subject_data_len
            if previous_subject in train_ids:
                train_X = np.concatenate((train_X, X[current_data_idx:end_data_idx]))
                train_y = np.concatenate((train_y, y[current_data_idx:end_data_idx]))
                train_dct = np.concatenate((train_dct, dct[current_data_idx:end_data_idx]))
                train_vidlens = np.concatenate((train_vidlens, video_lens[current_video_idx:end_video_idx]))
                train_subjects = np.concatenate((train_subjects, subjects[current_video_idx:end_video_idx]))
            elif previous_subject in val_ids:
                val_X = np.concatenate((val_X, X[current_data_idx:end_data_idx]))
                val_y = np.concatenate((val_y, y[current_data_idx:end_data_idx]))
                val_dct = np.concatenate((val_dct, dct[current_data_idx:end_data_idx]))
                val_vidlens = np.concatenate((val_vidlens, video_lens[current_video_idx:end_video_idx]))
                val_subjects = np.concatenate((val_subjects, subjects[current_video_idx:end_video_idx]))
            else:
                test_X = np.concatenate((test_X, X[current_data_idx:end_data_idx]))
                test_y = np.concatenate((test_y, y[current_data_idx:end_data_idx]))
                test_dct = np.concatenate((test_dct, dct[current_data_idx:end_data_idx]))
                test_vidlens = np.concatenate((test_vidlens, video_lens[current_video_idx:end_video_idx]))
                test_subjects = np.concatenate((test_subjects, subjects[current_video_idx:end_video_idx]))
            previous_subject = subject
            current_video_idx = end_video_idx
            current_data_idx = end_data_idx
            subject_video_count = 1
            populate = False
    return train_X, train_y, train_dct, train_vidlens, train_subjects, \
           val_X, val_y, val_dct, val_vidlens, val_subjects, \
           test_X, test_y, test_dct, test_vidlens, test_subjects


def read_data_split_file(path, sep=','):
    with open(path) as f:
        subjects = f.readline().split(sep)
        subjects = [int(s) for s in subjects]
    return subjects


def evaluate_model(X_val, y_val, mask_val, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val)
    no_gps = output.shape[1]
    confusion_matrix = np.zeros((no_gps, no_gps), dtype='int')

    ix = np.argmax(output, axis=1)
    c = ix == y_val
    classification_rate = np.sum(c == True) / float(len(c))

    # construct the confusion matrix
    for i, target in enumerate(y_val):
        confusion_matrix[target, ix[i]] += 1

    return classification_rate, confusion_matrix


def parse_options():
    options = dict()
    options['config'] = 'config/unimodal_dct.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file to use, default=config/unimodal_dct.ini')
    parser.add_argument('--write_results', help='write results to file')
    parser.add_argument('--dct_data', help='DCT Features Data File')
    parser.add_argument('--no_coeff', help='Number of DCT Coefficients')
    parser.add_argument('--no_epochs', help='Max epochs to run')
    parser.add_argument('--epochsize', help='Number of mini batches to run for each epoch')
    parser.add_argument('--batchsize', help='Mini batch size')
    parser.add_argument('--validation_window', help='validation window size')
    args = parser.parse_args()
    if args.config:
        options['config'] = args.config
    if args.write_results:
        options['write_results'] = args.write_results
    if args.dct_data:
        options['dct_data'] = args.dct_data
    if args.no_coeff:
        options['no_coeff'] = int(args.no_coeff)
    if args.no_epochs:
        options['no_epochs'] = int(args.no_epochs)
    if args.validation_window:
        options['validation_window'] = int(args.validation_window)
    if args.epochsize:
        options['epochsize'] = int(args.epochsize)
    if args.batchsize:
        options['batchsize'] = int(args.batchsize)
    return options


def main():
    configure_theano()
    options = parse_options()
    config_file = options['config']
    config = ConfigParser.ConfigParser()
    config.read(config_file)

    print('CLI options: {}'.format(options.items()))

    print(config.items('data'))
    print(config.items('models'))
    print(config.items('training'))

    print('preprocessing dataset...')
    data = load_mat_file(config.get('data', 'images'))
    dct_data = load_mat_file(options['dct_data'] if 'dct_data' in options else config.get('data', 'dct'))
    no_coeff = options['no_coeff'] if 'no_coeff' in options else config.getint('models', 'no_coeff')
    no_epochs = options['no_epochs'] if 'no_epochs' in options else config.getint('training', 'no_epochs')
    validation_window = options['validation_window'] if 'validation_window' in options \
        else config.getint('training', 'validation_window')
    epochsize = options['epochsize'] if 'epochsize' in options else config.getint('training', 'epochsize')
    batchsize = options['batchsize'] if 'batchsize' in options else config.getint('training', 'batchsize')

    # 53 subjects, 70 utterances, 5 view angles
    # s[x]_v[y]_u[z].mp4
    # resized, height, width = (26, 44)
    # ['dataMatrix', 'targetH', 'targetsPerVideoVec', 'videoLengthVec', '__header__', 'targetsVec',
    # '__globals__', 'iterVec', 'filenamesVec', 'dataMatrixCells', 'subjectsVec', 'targetW', '__version__']

    print(data.keys())
    X = data['dataMatrix'].astype('float32')
    y = data['targetsVec'].astype('int32')
    y = y.reshape((len(y),))
    dct_feats = dct_data['dctFeatures'].astype('float32')
    uniques = np.unique(y)
    print('number of classifications: {}'.format(len(uniques)))
    subjects = data['subjectsVec'].astype('int')
    subjects = subjects.reshape((len(subjects),))
    video_lens = data['videoLengthVec'].astype('int')
    video_lens = video_lens.reshape((len(video_lens,)))

    # X = reorder_data(X, (26, 44), 'f', 'c')
    # print('performing sequencewise mean image removal...')
    # X = sequencewise_mean_image_subtraction(X, video_lens)
    # visualize_images(X[550:650], (26, 44))

    # mean remove dct features
    # dct_feats = sequencewise_mean_image_subtraction(dct_feats, video_lens)

    train_subject_ids = read_data_split_file('data/train.txt')
    val_subject_ids = read_data_split_file('data/val.txt')
    test_subject_ids = read_data_split_file('data/test.txt')
    print('Train: {}'.format(train_subject_ids))
    print('Validation: {}'.format(val_subject_ids))
    print('Test: {}'.format(test_subject_ids))
    train_X, train_y, train_dct, train_vidlens, train_subjects, \
    val_X, val_y, val_dct, val_vidlens, val_subjects, \
    test_X, test_y, test_dct, test_vidlens, test_subjects = \
        split_data(X, y, dct_feats, subjects, video_lens, train_subject_ids, val_subject_ids, test_subject_ids)

    assert train_X.shape[0] + val_X.shape[0] + test_X.shape[0] == len(X)
    assert train_y.shape[0] + val_y.shape[0] + test_y.shape[0] == len(y)
    assert train_vidlens.shape[0] + val_vidlens.shape[0] + test_vidlens.shape[0] == len(video_lens)
    assert train_subjects.shape[0] + val_subjects.shape[0] + test_subjects.shape[0] == len(subjects)

    train_X = normalize_input(train_X, centralize=True)
    val_X = normalize_input(val_X, centralize=True)
    test_X = normalize_input(test_X, centralize=True)

    # featurewise normalize dct features
    train_dct, dct_mean, dct_std = featurewise_normalize_sequence(train_dct)
    val_dct = (val_dct - dct_mean) / dct_std
    test_dct = (test_dct - dct_mean) / dct_std

    # IMPT: the encoder was trained with fortan ordered images, so to visualize
    # convert all the images to C order using reshape_images_order()
    # output = dbn.predict(test_X)
    # test_X = reshape_images_order(test_X, (26, 44))
    # output = reshape_images_order(output, (26, 44))
    # visualize_reconstruction(test_X[:36, :], output[:36, :], shape=(26, 44))

    inputs = T.tensor3('inputs', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.ivector('targets')

    print('constructing end to end model...')

    network = lstm_classifier_baseline.create_model((None, None, no_coeff*3), inputs,
                                                    (None, None), mask,
                                                    250, 10)

    print_network(network)
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.categorical_crossentropy(predictions, targets))
    updates = adam(cost, all_params)

    train = theano.function(
        [inputs, targets, mask],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs, targets, mask], cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_predictions, targets))
    compute_test_cost = theano.function(
        [inputs, targets, mask], test_cost, allow_input_downcast=True)

    val_fn = theano.function([inputs, mask], test_predictions, allow_input_downcast=True)

    # We'll train the network with 10 epochs of 30 minibatches each
    print('begin training...')
    cost_train = []
    cost_val = []
    class_rate = []
    STRIP_SIZE = 3
    val_window = circular_list(validation_window)
    train_strip = np.zeros((STRIP_SIZE,))
    best_val = float('inf')
    best_cr = 0.0

    datagen = gen_lstm_batch_random(train_X, train_y, train_vidlens, batchsize=batchsize)
    val_datagen = gen_lstm_batch_random(val_X, val_y, val_vidlens, batchsize=len(val_vidlens))
    test_datagen = gen_lstm_batch_random(test_X, test_y, test_vidlens, batchsize=len(test_vidlens))
    integral_lens = compute_integral_len(train_vidlens)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, idxs_val = next(val_datagen)
    integral_lens_val = compute_integral_len(val_vidlens)
    dct_val = gen_seq_batch_from_idx(val_dct, idxs_val, val_vidlens, integral_lens_val, np.max(val_vidlens))

    X_test, y_test, mask_test, idxs_test = next(test_datagen)
    integral_lens_test = compute_integral_len(test_vidlens)
    dct_test = gen_seq_batch_from_idx(test_dct, idxs_test, test_vidlens, integral_lens_test, np.max(test_vidlens))

    for epoch in range(no_epochs):
        time_start = time.time()
        for i in range(epochsize):
            X, y, m, batch_idxs = next(datagen)
            d = gen_seq_batch_from_idx(train_dct, batch_idxs,
                                       train_vidlens, integral_lens, np.max(train_vidlens))
            print_str = 'Epoch {} batch {}/{}: {} examples using adam'.format(
                epoch + 1, i + 1, epochsize, len(X))
            print(print_str, end='')
            sys.stdout.flush()
            train(d, y, m)
            print('\r', end='')
        cost = compute_train_cost(d, y, m)
        val_cost = compute_test_cost(dct_val, y_val, mask_val)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model(dct_val, y_val, mask_val, val_fn)
        class_rate.append(cr)

        if val_cost < best_val:
            best_val = val_cost
            best_cr = cr
            test_cr, test_conf = evaluate_model(dct_test, y_test, mask_test, val_fn)
            print("Epoch {} train cost = {}, val cost = {}, "
                  "GL loss = {:.3f}, GQ = {:.3f}, CR = {:.3f}, Test CR= {:.3f} ({:.1f}sec)"
                  .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, test_cr, time.time() - time_start))
        else:
            print("Epoch {} train cost = {}, val cost = {}, "
                  "GL loss = {:.3f}, GQ = {:.3f}, CR = {:.3f} ({:.1f}sec)"
                  .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if epoch >= validation_window and early_stop2(val_window, best_val, validation_window):
            break

    phrases = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']

    print('Final Model')
    print('CR: {}, val loss: {}, Test CR: {}'.format(best_cr, best_val, test_cr))
    print('confusion matrix: ')
    plot_confusion_matrix(test_conf, phrases, fmt='latex')
    plot_validation_cost(cost_train, cost_val, savefilename='valid_cost')

    if 'write_results' in options:
        results_file = options['write_results']
        with open(results_file, mode='a') as f:
            f.write('{},{}\n'.format(test_cr, best_val))


if __name__ == '__main__':
    main()
