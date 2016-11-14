from __future__ import print_function
import sys
sys.path.insert(0, '../')
import os
import time
import pickle
import logging
import ConfigParser
import argparse

import theano.tensor as T
import theano

import matplotlib
# matplotlib.use('Agg')  # Change matplotlib backend, in case we have no X server running..

import lasagne as las
from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from utils.io import *
from utils.draw_net import *
from custom_layers.custom import DeltaLayer
from modelzoo import adenet_v1, deltanet, adenet_v2, adenet_v3, adenet_v4, adenet_v2_1, adenet_v6

import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, LSTMLayer, Gate, ElemwiseSumLayer, SliceLayer
from lasagne.layers import ReshapeLayer, DimshuffleLayer, ConcatLayer, BatchNormLayer, batch_norm
from lasagne.nonlinearities import tanh, linear, sigmoid, rectify, leaky_rectify
from lasagne.updates import nesterov_momentum, adadelta, sgd, norm_constraint
from lasagne.objectives import squared_error
from nolearn.lasagne import NeuralNet


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def load_dbn(path='models/avletters_ae.mat'):
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
    w5 = nn['w5']
    w6 = nn['w6']
    w7 = nn['w7']
    w8 = nn['w8']
    b1 = nn['b1'][0]
    b2 = nn['b2'][0]
    b3 = nn['b3'][0]
    b4 = nn['b4'][0]
    b5 = nn['b5'][0]
    b6 = nn['b6'][0]
    b7 = nn['b7'][0]
    b8 = nn['b8'][0]

    layers = [
        (InputLayer, {'name': 'input', 'shape': (None, 1200)}),
        (DenseLayer, {'name': 'l1', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w1, 'b': b1}),
        (DenseLayer, {'name': 'l2', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w2, 'b': b2}),
        (DenseLayer, {'name': 'l3', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w3, 'b': b3}),
        (DenseLayer, {'name': 'l4', 'num_units': 50, 'nonlinearity': linear, 'W': w4, 'b': b4}),
        (DenseLayer, {'name': 'l5', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w5, 'b': b5}),
        (DenseLayer, {'name': 'l6', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w6, 'b': b6}),
        (DenseLayer, {'name': 'l7', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w7, 'b': b7}),
        (DenseLayer, {'name': 'output', 'num_units': 1200, 'nonlinearity': linear, 'W': w8, 'b': b8}),
    ]

    dbn = NeuralNet(
        layers=layers,
        max_epochs=30,
        objective_loss_function=squared_error,
        update=nesterov_momentum,
        regression=True,
        verbose=1,
        update_learning_rate=0.001,
        update_momentum=0.05,
        objective_l2=0.005,
    )
    return dbn


def load_finetuned_dbn(path):
    """
    Load a fine tuned Deep Belief Net from file
    :param path: path to deep belief net parameters
    :return: deep belief net
    """
    dbn = NeuralNet(
        layers=[
            ('input', las.layers.InputLayer),
            ('l1', las.layers.DenseLayer),
            ('l2', las.layers.DenseLayer),
            ('l3', las.layers.DenseLayer),
            ('l4', las.layers.DenseLayer),
            ('l5', las.layers.DenseLayer),
            ('l6', las.layers.DenseLayer),
            ('l7', las.layers.DenseLayer),
            ('output', las.layers.DenseLayer)
        ],
        input_shape=(None, 1200),
        l1_num_units=2000, l1_nonlinearity=sigmoid,
        l2_num_units=1000, l2_nonlinearity=sigmoid,
        l3_num_units=500, l3_nonlinearity=sigmoid,
        l4_num_units=50, l4_nonlinearity=linear,
        l5_num_units=500, l5_nonlinearity=sigmoid,
        l6_num_units=1000, l6_nonlinearity=sigmoid,
        l7_num_units=2000, l7_nonlinearity=sigmoid,
        output_num_units=1200, output_nonlinearity=linear,
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.5,
        objective_l2=0.005,
        verbose=1,
        regression=True
    )
    with open(path, 'rb') as f:
        pretrained_nn = pickle.load(f)
    if pretrained_nn is not None:
        dbn.load_params_from(path)
    return dbn


def create_pretrained_encoder(weights, biases, incoming):
    l_1 = DenseLayer(incoming, 2000, W=weights[0], b=biases[0], nonlinearity=sigmoid, name='fc1')
    l_2 = DenseLayer(l_1, 1000, W=weights[1], b=biases[1], nonlinearity=sigmoid, name='fc2')
    l_3 = DenseLayer(l_2, 500, W=weights[2], b=biases[2], nonlinearity=sigmoid, name='fc3')
    l_4 = DenseLayer(l_3, 50, W=weights[3], b=biases[3], nonlinearity=linear, name='bottleneck')
    return l_4


def evaluate_model(X_val, y_val, mask_val, dct_val, window_size, eval_fn):
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
    no_gps = output.shape[1]
    confusion_matrix = np.zeros((no_gps, no_gps), dtype='int')

    ix = np.argmax(output, axis=1)
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


def visualize_confusion(X_val, utterance_no, target, actual):
    confused_with = abs(target - actual)
    visualize_sequence(X_val[utterance_no])
    visualize_sequence(X_val[utterance_no + confused_with], title='confused sequence')


def parse_options():
    options = dict()
    options['config'] = 'config/bimodal.ini'
    options['write_results'] = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file to use, default=config/bimodal.ini')
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
    ae_pretrained = config.get('models', 'pretrained')
    ae_finetuned = config.get('models', 'finetuned')
    fusiontype = config.get('models', 'fusiontype')
    learning_rate = float(config.get('training', 'learning_rate'))
    decay_rate = float(config.get('training', 'decay_rate'))
    decay_start = int(config.get('training', 'decay_start'))

    # create the necessary variable mappings
    data_matrix = data['dataMatrix'].astype('float32')
    data_matrix_len = data_matrix.shape[0]
    targets_vec = data['targetsVec']
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']
    dct_feats = dct_data['dctFeatures'].astype('float32')

    print('samplewise normalize images...')
    data_matrix = normalize_input(data_matrix, True)
    # mean remove
    # dct_feats = dct_feats[:, 0:30]
    # dct_feats = sequencewise_mean_image_subtraction(dct_feats, vid_len_vec.reshape((-1,)))

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

    finetune = False
    if finetune:
        print('fine-tuning...')
        dbn = load_dbn(ae_pretrained)
        dbn.initialize()
        dbn.fit(train_data, train_data)
        res = dbn.predict(test_data)
        # print(res.shape)
        visualize_reconstruction(test_data[300:336], res[300:336])

    save = False
    if save:
        pickle.dump(dbn, open(ae_finetuned, 'wb'))

    load = True
    if load:
        print('loading pre-trained encoding layers...')
        dbn = pickle.load(open(ae_finetuned, 'rb'))
        dbn.initialize()
        # recon = dbn.predict(test_data)
        # visualize_reconstruction(test_data[300:364], recon[300:364])
        # exit()

    load_convae = False
    if load_convae:
        print('loading pre-trained convolutional autoencoder...')
        encoder = load_model('models/conv_encoder_norm.dat')
        inputs = las.layers.get_all_layers(encoder)[0].input_var
    else:
        inputs = T.tensor3('inputs', dtype='float32')
    window = T.iscalar('theta')
    dct = T.tensor3('dct', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.ivector('targets')
    lr = theano.shared(np.array(learning_rate, dtype=theano.config.floatX), name='learning_rate')
    lr_decay = np.array(decay_rate, dtype=theano.config.floatX)

    print('constructing end to end model...')
    '''
    network, l_fuse = adenet_v1.create_model(dbn, (None, None, 1200), inputs,
                                             (None, None), mask,
                                             (None, None, 90), dct,
                                             250, window)

    network = deltanet.create_model(dbn, (None, None, 1200), inputs,
                                    (None, None), mask,
                                    250, window)

    '''
    network, l_fuse = adenet_v2.create_model(dbn, (None, None, 1200), inputs,
                                             (None, None), mask,
                                             (None, None, 90), dct,
                                             250, window, 26, fusiontype)

    '''
    network = adenet_v2_1.create_model(dbn, (None, None, 1200), inputs,
                                       (None, None), mask,
                                       (None, None, 90), dct,
                                       250, window)

    network, adascale = adenet_v4.create_model(dbn, (None, None, 1200), inputs,
                                               (None, None), mask,
                                               (None, None, 90), dct,
                                               250, window)
    '''
    print_network(network)
    draw_to_file(las.layers.get_all_layers(network), 'network.png')
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.categorical_crossentropy(predictions, targets))
    # updates = las.updates.adadelta(cost, all_params, learning_rate=lr)
    updates = las.updates.sgd(cost, all_params, learning_rate=lr)
    updates = las.updates.apply_momentum(updates, all_params, momentum=0.9)
    # updates = las.updates.adam(cost, all_params, learning_rate=lr)

    use_max_constraint = False
    if use_max_constraint:
        MAX_NORM = 4
        for param in las.layers.get_all_params(network, regularizable=True):
            if param.ndim > 1:  # only apply to dimensions larger than 1, exclude biases
                updates[param] = norm_constraint(param, MAX_NORM * las.utils.compute_norms(param.get_value()).mean())

    train = theano.function(
        [inputs, targets, mask, dct, window],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs, targets, mask, dct, window], cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_predictions, targets))
    compute_test_cost = theano.function(
        [inputs, targets, mask, dct, window], test_cost, allow_input_downcast=True)

    val_fn = theano.function([inputs, mask, dct, window], test_predictions, allow_input_downcast=True)

    # We'll train the network with 10 epochs of 30 minibatches each
    print('begin training...')
    cost_train = []
    cost_val = []
    class_rate = []
    NUM_EPOCHS = 100
    EPOCH_SIZE = 20
    BATCH_SIZE = 26
    WINDOW_SIZE = 9
    STRIP_SIZE = 3
    MAX_LOSS = 0.2
    VALIDATION_WINDOW = 4
    val_window = circular_list(VALIDATION_WINDOW)
    train_strip = np.zeros((STRIP_SIZE,))
    best_val = float('inf')
    best_conf = None
    best_cr = 0.0

    datagen = gen_lstm_batch_random(train_data, train_targets, train_vidlen_vec, batchsize=BATCH_SIZE)
    val_datagen = gen_lstm_batch_random(test_data, test_targets, test_vidlen_vec,
                                        batchsize=len(test_vidlen_vec))
    integral_lens = compute_integral_len(train_vidlen_vec)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, idxs_val = next(val_datagen)
    integral_lens_val = compute_integral_len(test_vidlen_vec)
    dct_val = gen_seq_batch_from_idx(test_dct, idxs_val, test_vidlen_vec, integral_lens_val, np.max(test_vidlen_vec))

    # confusions = map_confusion(X_val, y_val, mask_val, dct_val, WINDOW_SIZE, val_fn)

    def early_stop(cost_window):
        if len(cost_window) < 2:
            return False
        else:
            curr = cost_window[0]
            for idx, cost in enumerate(cost_window):
                if curr < cost or idx == 0:
                    curr = cost
                else:
                    return False
            return True

    for epoch in range(NUM_EPOCHS):
        time_start = time.time()
        for i in range(EPOCH_SIZE):
            X, y, m, batch_idxs = next(datagen)
            d = gen_seq_batch_from_idx(train_dct, batch_idxs,
                                       train_vidlen_vec, integral_lens, np.max(train_vidlen_vec))
            print_str = 'Epoch {} batch {}/{}: {} examples at learning rate = {:.4f}'.format(
                epoch + 1, i + 1, EPOCH_SIZE, len(X), float(lr.get_value()))
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

        cr, val_conf = evaluate_model(X_val, y_val, mask_val, dct_val, WINDOW_SIZE, val_fn)
        class_rate.append(cr)

        print("Epoch {} train cost = {}, validation cost = {}, "
              "generalization loss = {:.3f}, GQ = {:.3f}, classification rate = {:.3f} ({:.1f}sec)"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if val_cost < best_val:
            best_val = val_cost
            best_conf = val_conf
            best_cr = cr

        if epoch >= VALIDATION_WINDOW and early_stop(val_window):
            break

        # learning rate decay
        if epoch + 1 >= decay_start:
            lr.set_value(lr.get_value() * lr_decay)

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

    print('Best Model')
    print('classification rate: {}, validation loss: {}'.format(best_cr, best_val))
    if fusiontype == 'adasum':
        adascale_param = las.layers.get_all_param_values(l_fuse, scaling_param=True)
        print("final scaling params: {}".format(adascale_param))
    print('confusion matrix: ')
    plot_confusion_matrix(best_conf, letters, fmt='latex')
    plot_validation_cost(cost_train, cost_val, class_rate, 'e2e_valid_cost')

    '''
    datagen2 = gen_lstm_batch_seq(test_data, test_targets, test_vidlen_vec,
                                  batchsize=len(test_vidlen_vec))
    X_val, y_val, mask_val = next(datagen2)
    confusions = map_confusion(X_val, y_val, mask_val, dct_val, WINDOW_SIZE, val_fn)
    print(confusions)
    '''

    if options['write_results']:
        results_file = options['write_results']
        with open(results_file, mode='a') as f:
            f.write('{},{},{}\n'.format(fusiontype, best_cr, best_val))

if __name__ == '__main__':
    main()
