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
from custom.objectives import temporal_softmax_loss

import theano.tensor as T
import theano

import lasagne as las
import numpy as np
from lasagne.updates import adam

from modelzoo import adenet_v2
from utils.plotting_utils import print_network


def load_dbn(path='models/cuave_ae.mat'):
    """
    load a pretrained dbn from path
    :param path: path to the .mat dbn
    :return: pretrained deep belief network
    """
    # create the network using weights from pretrain_nn.mat
    nn = sio.loadmat(path)
    w1 = nn['w1'].astype('float32')
    w2 = nn['w2'].astype('float32')
    w3 = nn['w3'].astype('float32')
    w4 = nn['w4'].astype('float32')
    b1 = nn['b1'][0].astype('float32')
    b2 = nn['b2'][0].astype('float32')
    b3 = nn['b3'][0].astype('float32')
    b4 = nn['b4'][0].astype('float32')

    weights = [w1, w2, w3, w4]
    biases = [b1, b2, b3, b4]
    return weights, biases


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def read_data_split_file(path, sep=','):
    with open(path) as f:
        subjects = f.readline().split(sep)
        subjects = [int(s) for s in subjects]
    return subjects


def evaluate_model(X_val, y_val, mask_val, dct_val, window_size, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param dct_val: validation dct features
    :param window_size: size of window for computing delta coefficients
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, dct_val, window_size)
    no_gps = output.shape[1]
    confusion_matrix = np.zeros((no_gps, no_gps), dtype='int')

    ix = np.argmax(output, axis=1)
    c = ix == y_val
    classification_rate = np.sum(c == True) / float(len(c))

    # construct the confusion matrix
    for i, target in enumerate(y_val):
        confusion_matrix[target, ix[i]] += 1

    return classification_rate, confusion_matrix


def evaluate_model2(X_val, y_val, mask_val, dct_val, window_size, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param dct_val: validation dct features
    :param window_size: size of window for computing delta coefficients
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, dct_val, window_size)
    num_classes = output.shape[-1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype='int')
    ix = np.zeros((X_val.shape[0],), dtype='int')
    seq_lens = np.sum(mask_val, axis=-1)

    # for each example, we only consider argmax of the seq len
    votes = np.zeros((10,), dtype='int')
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


def map_confusion(X_val, y_val, mask_val, dct_val, window_size, eval_fn):
    """
        Evaluate a lstm model
        :param X_val: validation inputs
        :param y_val: validation targets
        :param mask_val: input masks for variable sequences
        :param dct_val: dct features
        :param window_size: size of window for computing delta coefficients
        :param eval_fn: evaluation function
        :return: classification rate, confusion matrix
        """
    output = eval_fn(X_val, mask_val, dct_val, window_size)
    ix = np.argmax(output, axis=1)
    confusions = []
    for i, target in enumerate(y_val):
        if ix[i] != target:
            confusions.append((i, target, ix[i]))
    return confusions


def parse_options():
    options = dict()
    options['config'] = 'config/bimodal_meanrm_raw_dct.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file to use, default=config/bimodal_meanrm_raw_dct.ini')
    parser.add_argument('--write_results', help='write results to file')
    args = parser.parse_args()
    if args.config:
        options['config'] = args.config
    if args.write_results:
        options['write_results'] = args.write_results
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
    dct_data = load_mat_file(config.get('data', 'dct'))
    pretrained_ae = config.get('models', 'pretrained')
    fusiontype = config.get('models', 'fusiontype')
    input_dimension = config.getint('models', 'input_dimension')
    no_coeff = config.getint('models', 'no_coeff')
    output_classes = config.getint('models', 'output_classes')
    lstm_size = config.getint('models', 'lstm_size')

    # capture training parameters
    validation_window = int(options['validation_window']) \
        if 'validation_window' in options else config.getint('training', 'validation_window')
    num_epoch = int(options['num_epoch']) if 'num_epoch' in options else config.getint('training', 'num_epoch')
    weight_init = options['weight_init'] if 'weight_init' in options else config.get('training', 'weight_init')
    learning_rate = options['learning_rate'] if 'learning_rate' in options \
        else config.getfloat('training', 'learning_rate')
    epochsize = options['epochsize'] if 'epochsize' in options else config.getint('training', 'epochsize')
    batchsize = options['batchsize'] if 'batchsize' in options else config.getint('training', 'batchsize')
    use_peepholes = options['use_peepholes'] if 'use_peepholes' in options else config.getboolean('training',
                                                                                                  'use_peepholes')
    use_blstm = config.getboolean('training', 'use_blstm')
    use_finetuning = config.getboolean('training', 'use_finetuning')

    weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'glorot':
        weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'norm':
        weight_init_fn = las.init.Normal(0.1)
    if weight_init == 'uniform':
        weight_init_fn = las.init.Uniform()
    if weight_init == 'ortho':
        weight_init_fn = las.init.Orthogonal()

    train_vidlens = data['trVideoLengthVec'].astype('int').reshape((-1,))
    val_vidlens = data['valVideoLengthVec'].astype('int').reshape((-1,))
    test_vidlens = data['testVideoLengthVec'].astype('int').reshape((-1,))
    train_X = data['trData'].astype('float32')
    val_X = data['valData'].astype('float32')
    test_X = data['testData'].astype('float32')
    train_dct = dct_data['trDctFeatures'].astype('float32')
    val_dct = dct_data['valDctFeatures'].astype('float32')
    test_dct = dct_data['testDctFeatures'].astype('float32')
    train_y = data['trTargetsVec'].astype('int').reshape((-1,)) + 1  # +1 to handle the -1 introduced in lstm_gendata
    val_y = data['valTargetsVec'].astype('int').reshape((-1,)) + 1
    test_y = data['testTargetsVec'].astype('int').reshape((-1,)) + 1

    # featurewise normalize dct features
    train_dct, dct_mean, dct_std = featurewise_normalize_sequence(train_dct)
    val_dct = (val_dct - dct_mean) / dct_std
    test_dct = (test_dct - dct_mean) / dct_std

    print('loading pretrained encoder: {}...'.format(pretrained_ae))
    weights, biases = load_dbn(pretrained_ae)

    # IMPT: the encoder was trained with fortan ordered images, so to visualize
    # convert all the images to C order using reshape_images_order()
    # output = dbn.predict(test_X)
    # test_X = reshape_images_order(test_X, (26, 44))
    # output = reshape_images_order(output, (26, 44))
    # visualize_reconstruction(test_X[:36, :], output[:36, :], shape=(26, 44))

    window = T.iscalar('theta')
    dct = T.tensor3('dct', dtype='float32')
    inputs = T.tensor3('inputs', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.imatrix('targets')

    print('constructing end to end model...')
    network, l_fuse = adenet_v2.create_model_from_pretrained_encoder(weights, biases, (None, None, input_dimension),
                                                                     inputs, (None, None), mask,
                                                                     (None, None, no_coeff*3), dct,
                                                                     lstm_size, window, output_classes, fusiontype,
                                                                     weight_init_fn, use_peepholes)

    print_network(network)
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = temporal_softmax_loss(predictions, targets, mask)
    updates = adam(cost, all_params, learning_rate=learning_rate)

    train = theano.function(
        [inputs, targets, mask, dct, window],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs, targets, mask, dct, window], cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = temporal_softmax_loss(test_predictions, targets, mask)
    compute_test_cost = theano.function(
        [inputs, targets, mask, dct, window], test_cost, allow_input_downcast=True)

    val_fn = theano.function([inputs, mask, dct, window], test_predictions, allow_input_downcast=True)

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
    best_tr = float('inf')
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

    # reshape the targets for validation
    y_val_evaluate = y_val
    y_val = y_val.reshape((-1, 1)).repeat(mask_val.shape[-1], axis=-1)

    for epoch in range(num_epoch):
        time_start = time.time()
        for i in range(epochsize):
            X, y, m, batch_idxs = next(datagen)
            # repeat targets based on max sequence len
            y = y.reshape((-1, 1))
            y = y.repeat(m.shape[-1], axis=-1)
            d = gen_seq_batch_from_idx(train_dct, batch_idxs,
                                       train_vidlens, integral_lens, np.max(train_vidlens))
            print_str = 'Epoch {} batch {}/{}: {} examples at learning rate = {:.4f}'.format(
                epoch + 1, i + 1, epochsize, len(X), learning_rate)
            print(print_str, end='')
            sys.stdout.flush()
            train(X, y, m, d, WINDOW_SIZE)
            print('\r', end='')
        cost = compute_train_cost(X, y, m, d, WINDOW_SIZE)
        val_cost = compute_test_cost(X_val, y_val, mask_val, dct_val, WINDOW_SIZE)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model2(X_val, y_val_evaluate, mask_val, dct_val, WINDOW_SIZE, val_fn)
        class_rate.append(cr)

        if val_cost < best_val:
            best_val = val_cost
            best_tr = cost
            best_conf = val_conf
            best_cr = cr
            if fusiontype == 'adasum':
                adascale_param = las.layers.get_all_param_values(l_fuse, scaling_param=True)
            test_cr, test_conf = evaluate_model2(X_test, y_test, mask_test, dct_test, WINDOW_SIZE, val_fn)
            print("Epoch {} train cost = {}, val cost = {}, "
                  "GL loss = {:.3f}, GQ = {:.3f}, CR = {:.3f}, Test CR= {:.3f} ({:.1f}sec)"
                  .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, test_cr, time.time() - time_start))
        else:
            print("Epoch {} train cost = {}, val cost = {}, "
                  "GL loss = {:.3f}, GQ = {:.3f}, CR = {:.3f} ({:.1f}sec)"
                  .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if epoch >= validation_window and early_stop2(val_window, best_val, validation_window):
            break

    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    print('Final Model')
    print('CR: {}, val loss: {}, Test CR: {}'.format(best_cr, best_val, test_cr))
    if fusiontype == 'adasum':
        print("final scaling params: {}".format(adascale_param))
    print('confusion matrix: ')
    plot_confusion_matrix(test_conf, numbers, fmt='latex')
    plot_validation_cost(cost_train, cost_val, savefilename='valid_cost')

    # confusions = map_confusion(X_test, y_test, mask_test, dct_test, WINDOW_SIZE, val_fn)
    # print(confusions)

    if 'write_results' in options:
        results_file = options['write_results']
        with open(results_file, mode='a') as f:
            f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(use_finetuning, 'yes', use_peepholes,
                                                                   'adam', weight_init, 'RELU',
                                                                   use_blstm, learning_rate, best_tr,
                                                                   best_val, best_cr*100, test_cr*100))

            s = ','.join([str(v) for v in cost_train])
            f.write('{}\n'.format(s))

            s = ','.join([str(v) for v in cost_val])
            f.write('{}\n'.format(s))

            s = ','.join([str(v) for v in class_rate])
            f.write('{}\n'.format(s))


if __name__ == '__main__':
    main()
