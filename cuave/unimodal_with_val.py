from __future__ import print_function
import sys
sys.path.insert(0, '../')
import time
import ConfigParser
import argparse

import matplotlib
matplotlib.use('Agg')

from lasagne.nonlinearities import sigmoid, rectify

from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from utils.io import *
from utils.regularization import early_stop2
from custom.objectives import temporal_softmax_loss

import theano.tensor as T
import theano

import lasagne as las
import numpy as np

from modelzoo import deltanet_majority_vote


def load_dbn(path='models/cuave_ae.mat'):
    """
    load a pretrained dbn from path
    :param path: path to the .mat dbn
    :return: pretrained deep belief network
    """
    # create the network using weights from pretrain_nn.mat
    nn = sio.loadmat(path)
    w1 = nn['w1']
    w2 = nn['w2']
    w3 = nn['w3']
    w4 = nn['w4']
    b1 = nn['b1'][0]
    b2 = nn['b2'][0]
    b3 = nn['b3'][0]
    b4 = nn['b4'][0]

    weights = [w1, w2, w3, w4]
    biases = [b1, b2, b3, b4]
    return weights, biases


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def evaluate_model(X_val, y_val, mask_val, window_size, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param window_size: size of window for computing delta coefficients
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, window_size)
    no_gps = output.shape[1]
    confusion_matrix = np.zeros((no_gps, no_gps), dtype='int')

    ix = np.argmax(output, axis=1)
    c = ix == y_val
    classification_rate = np.sum(c == True) / float(len(c))

    # construct the confusion matrix
    for i, target in enumerate(y_val):
        confusion_matrix[target, ix[i]] += 1

    return classification_rate, confusion_matrix


def evaluate_model2(X_val, y_val, mask_val, window_size, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param window_size: window size (typically 9)
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, window_size)
    num_classes = output.shape[-1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype='int')
    ix = np.zeros((X_val.shape[0],), dtype='int')
    seq_lens = np.sum(mask_val, axis=-1)

    # for each example, we only consider argmax of the seq len
    votes = np.zeros((num_classes,), dtype='int')
    for i, eg in enumerate(output):
        predictions = np.argmax(eg[:seq_lens[i]], axis=-1)
        for cls in range(num_classes):
            count = (predictions == cls).sum(axis=-1)
            votes[cls] = count
        ix[i] = np.argmax(votes)

    c = ix == y_val
    classification_rate = np.sum(c == True) / float(len(c))

    # construct the confusion matrix
    for i, target in enumerate(y_val):
        confusion_matrix[target, ix[i]] += 1

    return classification_rate, confusion_matrix


def parse_options():
    options = dict()
    options['config'] = 'config/unimodal_meanrmraw.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file to use, default=config/unimodal_meanrmraw.ini')
    parser.add_argument('--write_results', help='write results to file')
    parser.add_argument('--dct_data', help='DCT Features Data File')
    parser.add_argument('--no_coeff', help='Number of DCT Coefficients')
    parser.add_argument('--no_epochs', help='Max epochs to run')
    parser.add_argument('--epochsize', help='Number of mini batches to run for each epoch')
    parser.add_argument('--batchsize', help='Mini batch size')
    parser.add_argument('--validation_window', help='validation window size')
    parser.add_argument('--nonlinearity', help='nonlinearity for encoder')
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
    if args.nonlinearity:
        options['nonlinearity'] = int(args.nonlinearity)
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
    ae_pretrained = config.get('models', 'pretrained')
    input_dimension = config.getint('models', 'input_dimension')
    output_classes = config.getint('models', 'output_classes')
    lstm_size = config.getint('models', 'lstm_size')
    nonlinearity = options['nonlinearity'] if 'nonlinearity' in options else config.get('models', 'nonlinearity')

    if nonlinearity == 'sigmoid':
        nonlinearity = sigmoid
    if nonlinearity == 'rectify':
        nonlinearity = rectify

    # capture training parameters
    validation_window = int(options['validation_window']) \
        if 'validation_window' in options else config.getint('training', 'validation_window')
    no_epochs = int(options['no_epochs']) if 'no_epochs' in options else config.getint('training', 'no_epochs')
    weight_init = options['weight_init'] if 'weight_init' in options else config.get('training', 'weight_init')
    learning_rate = options['learning_rate'] if 'learning_rate' in options \
        else config.getfloat('training', 'learning_rate')
    epochsize = options['epochsize'] if 'epochsize' in options else config.getint('training', 'epochsize')
    batchsize = options['batchsize'] if 'batchsize' in options else config.getint('training', 'batchsize')
    use_peepholes = options['use_peepholes'] if 'use_peepholes' in options else config.getboolean('training',
                                                                                                  'use_peepholes')

    weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'glorot':
        weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'norm':
        weight_init_fn = las.init.Normal(0.1)
    if weight_init == 'uniform':
        weight_init_fn = las.init.Uniform()
    if weight_init == 'ortho':
        weight_init_fn = las.init.Orthogonal()

    train_subject_ids = read_data_split_file('data/train.txt')
    val_subject_ids = read_data_split_file('data/val.txt')
    test_subject_ids = read_data_split_file('data/test.txt')

    data_matrix = data['dataMatrix']
    targets_vec = data['targetsVec'].reshape((-1,))
    subjects_vec = data['subjectsVec'].reshape((-1,))
    vidlen_vec = data['videoLengthVec'].reshape((-1,))

    data_matrix = sequencewise_mean_image_subtraction(data_matrix, vidlen_vec)

    train_X, train_y, train_vidlens, train_subjects, \
    val_X, val_y, val_vidlens, val_subjects, \
    test_X, test_y, test_vidlens, test_subjects = split_seq_data(data_matrix, targets_vec, subjects_vec, vidlen_vec,
                   train_subject_ids, val_subject_ids, test_subject_ids)
    train_y += 1
    val_y += 1
    test_y += 1

    '''
    train_vidlens = data['trVideoLengthVec'].astype('int').reshape((-1,))
    val_vidlens = data['valVideoLengthVec'].astype('int').reshape((-1,))
    test_vidlens = data['testVideoLengthVec'].astype('int').reshape((-1,))
    train_X = data['trData'].astype('float32')
    val_X = data['valData'].astype('float32')
    test_X = data['testData'].astype('float32')
    train_y = data['trTargetsVec'].astype('int').reshape((-1,)) + 1  # +1 to handle the -1 introduced in lstm_gendata
    val_y = data['valTargetsVec'].astype('int').reshape((-1,)) + 1
    test_y = data['testTargetsVec'].astype('int').reshape((-1,)) + 1
    '''

    train_X = reorder_data(train_X, (30, 50))
    val_X = reorder_data(val_X, (30, 50))
    test_X = reorder_data(test_X, (30, 50))

    train_X = sequencewise_mean_image_subtraction(train_X, train_vidlens)
    val_X = sequencewise_mean_image_subtraction(val_X, val_vidlens)
    test_X = sequencewise_mean_image_subtraction(test_X, test_vidlens)

    weights, biases = load_dbn(ae_pretrained)

    train_X = normalize_input(train_X, centralize=True)
    val_X = normalize_input(val_X, centralize=True)
    test_X = normalize_input(test_X, centralize=True)

    # IMPT: the encoder was trained with fortan ordered images, so to visualize
    # convert all the images to C order using reshape_images_order()
    # output = dbn.predict(test_X)
    # test_X = reshape_images_order(test_X, (26, 44))
    # output = reshape_images_order(output, (26, 44))
    # visualize_reconstruction(test_X[:36, :], output[:36, :], shape=(26, 44))

    window = T.iscalar('theta')
    inputs = T.tensor3('inputs', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.imatrix('targets')

    print('constructing end to end model...')
    network = deltanet_majority_vote.create_model_using_pretrained_encoder(weights, biases,
                                                                           (None, None, input_dimension), inputs,
                                                                           (None, None), mask, lstm_size, window,
                                                                           output_classes, weight_init_fn,
                                                                           use_peepholes, nonlinearity)

    print_network(network)
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = temporal_softmax_loss(predictions, targets, mask)
    updates = las.updates.adam(cost, all_params, learning_rate=learning_rate)

    train = theano.function(
        [inputs, targets, mask, window],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs, targets, mask, window], cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = temporal_softmax_loss(test_predictions, targets, mask)
    compute_test_cost = theano.function(
        [inputs, targets, mask, window], test_cost, allow_input_downcast=True)

    val_fn = theano.function([inputs, mask, window], test_predictions, allow_input_downcast=True)

    # We'll train the network with 10 epochs of 30 minibatches each
    print('begin training...')
    cost_train = []
    cost_val = []
    class_rate = []
    WINDOW_SIZE = 9
    STRIP_SIZE = 3
    val_window = circular_list(validation_window)
    train_strip = np.zeros((STRIP_SIZE,))
    best_val = float('inf')

    datagen = gen_lstm_batch_random(train_X, train_y, train_vidlens, batchsize=batchsize)
    val_datagen = gen_lstm_batch_random(val_X, val_y, val_vidlens, batchsize=len(val_vidlens))
    test_datagen = gen_lstm_batch_random(test_X, test_y, test_vidlens, batchsize=len(test_vidlens))

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, _ = next(val_datagen)

    # Use this test set to check final classification performance
    X_test, y_test, mask_test, _ = next(test_datagen)

    # reshape the targets for validation
    y_val_evaluate = y_val
    y_val = y_val.reshape((-1, 1)).repeat(mask_val.shape[-1], axis=-1)

    for epoch in range(no_epochs):
        time_start = time.time()
        for i in range(epochsize):
            X, y, m, _ = next(datagen)
            # repeat targets based on max sequence len
            y = y.reshape((-1, 1))
            y = y.repeat(m.shape[-1], axis=-1)
            print_str = 'Epoch {} batch {}/{}: {} examples using adam at learning rate = {:.4f}'.format(
                epoch + 1, i + 1, epochsize, len(X), learning_rate)
            print(print_str, end='')
            sys.stdout.flush()
            train(X, y, m, WINDOW_SIZE)
            print('\r', end='')
        cost = compute_train_cost(X, y, m, WINDOW_SIZE)
        val_cost = compute_test_cost(X_val, y_val, mask_val, WINDOW_SIZE)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model2(X_val, y_val_evaluate, mask_val, WINDOW_SIZE, val_fn)
        class_rate.append(cr)

        if val_cost < best_val:
            best_val = val_cost
            val_cr = cr
            test_cr, test_conf = evaluate_model2(X_test, y_test, mask_test, WINDOW_SIZE, val_fn)
            print("Epoch {} train cost = {}, val cost = {}, "
                  "GL loss = {:.3f}, GQ = {:.3f}, CR = {:.3f}, Test CR= {:.3f} ({:.1f}sec)"
                  .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, test_cr, time.time() - time_start))
            best_params = las.layers.get_all_param_values(network)
        else:
            print("Epoch {} train cost = {}, val cost = {}, "
                  "GL loss = {:.3f}, GQ = {:.3f}, CR = {:.3f} ({:.1f}sec)"
                  .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if epoch >= validation_window and early_stop2(val_window, best_val, validation_window):
            break

    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    print('Final Model')
    print('test CR: {}, val CR: {}, val loss: {}'.format(test_cr, val_cr, best_val))
    print('confusion matrix: ')
    plot_confusion_matrix(test_conf, numbers, fmt='grid')
    plot_validation_cost(cost_train, cost_val, class_rate)

    if 'write_results' in options:
        with open(options['write_results'], mode='a') as f:
            f.write('{},{},{}\n'.format(test_cr, val_cr, best_val))

    if 'save_best' in options:
        print('Saving the best model so far...')
        las.layers.set_all_param_values(network, best_params)
        save_model_params(network, options['save_best'])
        print('Model Saved!')


if __name__ == '__main__':
    main()
