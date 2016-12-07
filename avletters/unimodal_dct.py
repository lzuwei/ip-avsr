from __future__ import print_function
import sys
sys.path.insert(0, '../')
import time
import ConfigParser
import argparse

import theano.tensor as T
import theano

import matplotlib
matplotlib.use('Agg')  # Change matplotlib backend, in case we have no X server running..

from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from utils.io import *
from utils.draw_net import draw_to_file
from modelzoo import lstm_classifier_baseline
from utils.regularization import early_stop2

import numpy as np


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


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

    print('Reading Config File: {}...'.format(config_file))
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

    # create the necessary variable mappings
    data_matrix = data['dataMatrix'].astype('float32')
    data_matrix_len = data_matrix.shape[0]
    targets_vec = data['targetsVec']
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']
    dct_feats = dct_data['dctFeatures'].astype('float32')

    indexes = create_split_index(data_matrix_len, vid_len_vec, iter_vec)
    train_vidlen_vec, test_vidlen_vec = split_videolen(vid_len_vec, iter_vec)
    assert len(train_vidlen_vec) == 520
    assert len(test_vidlen_vec) == 260
    assert np.sum(vid_len_vec) == data_matrix_len

    # split the data
    train_data = data_matrix[indexes == True]
    train_targets = targets_vec[indexes == True]
    train_targets = train_targets.reshape((len(train_targets),))
    test_data = data_matrix[indexes == False]
    test_targets = targets_vec[indexes == False]
    test_targets = test_targets.reshape((len(test_targets),))

    # split the dct features
    train_dct = dct_feats[indexes == True].astype(np.float32)
    test_dct = dct_feats[indexes == False].astype(np.float32)
    train_dct, dct_mean, dct_std = featurewise_normalize_sequence(train_dct)
    test_dct = (test_dct - dct_mean) / dct_std

    inputs = T.tensor3('inputs', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.ivector('targets')

    print('constructing end to end model...')
    network = lstm_classifier_baseline.create_model((None, None, no_coeff*3), inputs,
                                                    (None, None), mask,
                                                    250, 26)

    print_network(network)
    draw_to_file(las.layers.get_all_layers(network), 'network.png', verbose=True)
    # exit()
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.categorical_crossentropy(predictions, targets))
    updates = las.updates.adam(cost, all_params)

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
    best_conf = None
    best_cr = 0.0

    datagen = gen_lstm_batch_random(train_data, train_targets, train_vidlen_vec, batchsize=batchsize)
    val_datagen = gen_lstm_batch_random(test_data, test_targets, test_vidlen_vec,
                                        batchsize=len(test_vidlen_vec))
    integral_lens = compute_integral_len(train_vidlen_vec)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, idxs_val = next(val_datagen)
    integral_lens_val = compute_integral_len(test_vidlen_vec)
    dct_val = gen_seq_batch_from_idx(test_dct, idxs_val, test_vidlen_vec, integral_lens_val, np.max(test_vidlen_vec))

    for epoch in range(no_epochs):
        time_start = time.time()
        for i in range(epochsize):
            X, y, m, batch_idxs = next(datagen)
            d = gen_seq_batch_from_idx(train_dct, batch_idxs,
                                       train_vidlen_vec, integral_lens, np.max(train_vidlen_vec))
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

        print("Epoch {} train cost = {}, validation cost = {}, "
              "generalization loss = {:.3f}, GQ = {:.3f}, classification rate = {:.3f} ({:.1f}sec)"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if val_cost < best_val:
            best_val = val_cost
            best_conf = val_conf
            best_cr = cr

        if epoch >= validation_window and early_stop2(val_window, best_val, validation_window):
            break

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

    print('Best Model')
    print('classification rate: {}, validation loss: {}'.format(best_cr, best_val))
    print('confusion matrix: ')
    plot_confusion_matrix(best_conf, letters, fmt='latex')
    plot_validation_cost(cost_train, cost_val, class_rate)

if __name__ == '__main__':
    main()
