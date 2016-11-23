from __future__ import print_function
import sys
sys.path.insert(0, '../')
import os
import time
import cPickle as pickle
import logging
import ConfigParser
import argparse

import theano.tensor as T
import theano

import matplotlib
matplotlib.use('Agg')  # Change matplotlib backend, in case we have no X server running..

import lasagne as las
from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from utils.io import *
from custom_layers.custom import DeltaLayer
from modelzoo import adenet_v6, adenet_v2_1

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
    l_1 = DenseLayer(incoming, 1000, W=weights[0], b=biases[0], nonlinearity=sigmoid, name='fc1')
    l_2 = DenseLayer(l_1, 1000, W=weights[1], b=biases[1], nonlinearity=sigmoid, name='fc2')
    l_3 = DenseLayer(l_2, 1000, W=weights[2], b=biases[2], nonlinearity=sigmoid, name='fc3')
    l_4 = DenseLayer(l_3, 50, W=weights[3], b=biases[3], nonlinearity=linear, name='bottleneck')
    return l_4


def evaluate_model(X_val, y_val, mask_val, diff_val, window_size, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param diff_val: diff image features
    :param window_size: size of window for computing delta coefficients
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, diff_val, window_size)
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
    options['config'] = 'config/bimodal_diff_image.ini'
    options['no_plot'] = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file to use, default=config/bimodal_diff_image.ini')
    parser.add_argument('--write_results', help='write results to file')
    parser.add_argument('--update_rule', help='adadelta, sgdm, sgdnm')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--decay_rate', help='learning rate decay')
    parser.add_argument('--momentum', help='momentum')
    parser.add_argument('--momentum_schedule', help='eg: 0.9,0.9,0.95,0.99')
    parser.add_argument('--validation_window', help='validation window length, eg: 6')
    parser.add_argument('--t1', help='epoch to start learning rate decay, eg: 10')
    parser.add_argument('--weight_init', help='norm,glorot,ortho,uniform')
    parser.add_argument('--num_epoch', help='number of epochs to run')
    parser.add_argument('--use_peepholes', help='use peephole connections in LSTM')
    parser.add_argument('--no_plot', dest='no_plot', action='store_true', help='disable plots')
    parser.set_defaults(no_plot=False)
    parser.set_defaults(use_peepholes=True)
    args = parser.parse_args()
    if args.config:
        options['config'] = args.config
    if args.write_results:
        options['write_results'] = args.write_results
    if args.update_rule:
        options['update_rule'] = args.update_rule
    if args.learning_rate:
        options['learning_rate'] = args.learning_rate
    if args.decay_rate:
        options['decay_rate'] = args.decay_rate
    if args.momentum:
        options['momentum'] = args.momentum
    if args.momentum_schedule:
        options['momentum_schedule'] = args.momentum_schedule
    if args.validation_window:
        options['validation_window'] = args.validation_window
    if args.t1:
        options['t1'] = args.t1
    if args.weight_init:
        options['weight_init'] = args.weight_init
    if args.num_epoch:
        options['num_epoch'] = args.num_epoch
    if args.no_plot:
        options['no_plot'] = True
    if args.use_peepholes:
        options['use_peepholes'] = args.use_peepholes
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
    diff_data = load_mat_file(config.get('data', 'diff'))
    ae_pretrained = config.get('models', 'pretrained')
    ae_finetuned = config.get('models', 'finetuned')
    ae_finetuned_diff = config.get('models', 'finetuned_diff')
    fusiontype = config.get('models', 'fusiontype')
    do_finetune = config.getboolean('training', 'do_finetune')
    save_finetune = config.getboolean('training', 'save_finetune')
    load_finetune = config.getboolean('training', 'load_finetune')
    load_finetune_diff = config.getboolean('training', 'load_finetune_diff')
    model = config.get('models', 'model')

    # capture training parameters
    update_rule = options['update_rule'] if 'update_rule' in options else config.get('training', 'update_rule')
    learning_rate = float(options['learning_rate']) \
        if 'learning_rate' in options else config.getfloat('training', 'learning_rate')
    decay_rate = float(options['decay_rate']) if 'decay_rate' in options else config.getfloat('training', 'decay_rate')
    decay_start = int(options['decay_start']) if 'decay_start' in options else config.getint('training', 'decay_start')
    validation_window = int(options['validation_window']) \
        if 'validation_window' in options else config.getint('training', 'validation_window')
    t1 = int(options['t1']) if 't1' in options else config.getint('training', 't1')
    num_epoch = int(options['num_epoch']) if 'num_epoch' in options else config.getint('training', 'num_epoch')
    weight_init = options['weight_init'] if 'weight_init' in options else config.get('training', 'weight_init')
    use_peepholes = options['use_peepholes'] if 'use_peepholes' in options else config.getboolean('training',
                                                                                                  'use_peepholes')

    if update_rule == 'sgdm' or update_rule == 'sgdnm':
        momentum = float(options['momentum']) if 'momentum' in options else config.getfloat('training', 'momentum')
        momentum_schedule = options['momentum_schedule'] \
            if 'momentum_schedule' in options else config.get('training', 'momentum_schedule')
        mm_schedule = [float(m) for m in momentum_schedule.split(',')]

    weight_init_fn = las.init.Orthogonal()
    if weight_init == 'glorot':
        weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'norm':
        weight_init_fn = las.init.Normal(0.1)
    if weight_init == 'uniform':
        weight_init_fn = las.init.Uniform()
    if weight_init == 'ortho':
        weight_init_fn = las.init.Orthogonal()

    # create the necessary variable mappings
    data_matrix = data['dataMatrix']
    data_matrix_len = data_matrix.shape[0]
    targets_vec = data['targetsVec']
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']
    diff_data_matrix = diff_data['dataMatrix']

    # samplewise normalize
    # print('sameplewise mean normalize...')
    # data_matrix = normalize_input(data_matrix)
    # diff_data_matrix = normalize_input(diff_data_matrix)
    # diff_data_matrix = compute_diff_images(data_matrix, vid_len_vec.reshape((-1,))).astype('float32')

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
    train_diff_data = diff_data_matrix[indexes == True]
    test_diff_data = diff_data_matrix[indexes == False]

    if do_finetune:
        print('fine-tuning...')
        ae = load_dbn(ae_pretrained)
        ae.initialize()
        ae.fit(train_data, train_data)
        res = ae.predict(test_data)
        # print(res.shape)
        visualize_reconstruction(test_data[300:336], res[300:336])

    if save_finetune:
        pickle.dump(ae, open(ae_finetuned, 'wb'))

    if load_finetune:
        print('loading pre-trained encoding layers...')
        ae = pickle.load(open(ae_finetuned, 'rb'))
        ae.initialize()

    if load_finetune_diff:
        print('loading pre-trained diff image encoding layers...')
        diff_ae = pickle.load(open(ae_finetuned_diff, 'rb'))
        diff_ae.initialize()

    load_convae = False
    if load_convae:
        print('loading pre-trained convolutional autoencoder...')
        encoder = load_model('models/conv_encoder_norm.dat')
        inputs_raw = las.layers.get_all_layers(encoder)[0].input_var
    else:
        inputs_raw = T.tensor3('inputs_raw', dtype='float32')
        inputs_diff = T.tensor3('inputs_diff', dtype='float32')

    window = T.iscalar('theta')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.ivector('targets')
    lr = theano.shared(np.array(learning_rate, dtype=theano.config.floatX), name='learning_rate')
    lr_decay = np.array(decay_rate, dtype=theano.config.floatX)

    if update_rule == 'sgdm' or update_rule == 'sgdnm':
        mm = theano.shared(np.array(momentum, dtype=theano.config.floatX), name='momentum')

    print('constructing end to end model...')

    if model == 'adenet_v2_1':
        network, l_fuse = adenet_v2_1.create_model(ae, diff_ae, (None, None, 1200), inputs_raw,
                                                   (None, None), mask,
                                                   (None, None, 1200), inputs_diff,
                                                   250, window, 26, fusiontype,
                                                   w_init_fn=weight_init_fn,
                                                   use_peepholes=use_peepholes)

    print_network(network)
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.categorical_crossentropy(predictions, targets))
    if update_rule == 'adadelta':
        updates = las.updates.adadelta(cost, all_params, learning_rate=lr)
    if update_rule == 'sgdm':
        updates = las.updates.sgd(cost, all_params, learning_rate=lr)
        updates = las.updates.apply_momentum(updates, all_params, momentum=mm)
    if update_rule == 'sgdnm':
        updates = las.updates.sgd(cost, all_params, learning_rate=lr)
        updates = las.updates.apply_nesterov_momentum(updates, all_params, momentum=mm)
    if update_rule == 'adam':
        updates = las.updates.adam(cost, all_params)

    train = theano.function(
        [inputs_raw, targets, mask, inputs_diff, window],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs_raw, targets, mask, inputs_diff, window],
                                         cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_predictions, targets))
    compute_test_cost = theano.function(
        [inputs_raw, targets, mask, inputs_diff, window], test_cost, allow_input_downcast=True)

    val_fn = theano.function([inputs_raw, mask, inputs_diff, window], test_predictions, allow_input_downcast=True)

    # We'll train the network with 10 epochs of 30 minibatches each
    print('begin training...')
    cost_train = []
    cost_val = []
    class_rate = []
    EPOCH_SIZE = 20
    BATCH_SIZE = 26
    WINDOW_SIZE = 9
    STRIP_SIZE = 3
    val_window = circular_list(validation_window)
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
    diff_val = gen_seq_batch_from_idx(test_diff_data, idxs_val,
                                      test_vidlen_vec, integral_lens_val, np.max(test_vidlen_vec))

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

    def early_stop2(cost_window, min_val_cost, threshold):
        if len(cost_window) < 2:
            return False
        else:
            count = 0
            for cost in cost_window:
                if cost > min_val_cost:
                    count += 1
                if count == threshold:
                    return True

    for epoch in range(num_epoch):
        time_start = time.time()
        for i in range(EPOCH_SIZE):
            X, y, m, batch_idxs = next(datagen)
            diff = gen_seq_batch_from_idx(train_diff_data, batch_idxs,
                                          train_vidlen_vec, integral_lens, np.max(train_vidlen_vec))
            if update_rule == 'adam':
                print_str = 'Epoch {} batch {}/{}: {} examples with {} using default params'.format(
                    epoch + 1, i + 1, EPOCH_SIZE, len(X), update_rule)
            if update_rule == 'adadelta':
                print_str = 'Epoch {} batch {}/{}: {} examples at learning rate = {:.4f} with {}'.format(
                    epoch + 1, i + 1, EPOCH_SIZE, len(X), float(lr.get_value()), update_rule)
            if update_rule == 'sgdm' or update_rule == 'sgdnm':
                print_str = 'Epoch {} batch {}/{}: {} examples at learning rate = {:.4f}, ' \
                            'momentum = {:.4f} with {}'.format(
                    epoch + 1, i + 1, EPOCH_SIZE, len(X), float(lr.get_value()), float(mm.get_value()), update_rule)
            print(print_str, end='')
            sys.stdout.flush()
            train(X, y, m, diff, WINDOW_SIZE)
            print('\r', end='')
        cost = compute_train_cost(X, y, m, diff, WINDOW_SIZE)
        val_cost = compute_test_cost(X_val, y_val, mask_val, diff_val, WINDOW_SIZE)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model(X_val, y_val, mask_val, diff_val, WINDOW_SIZE, val_fn)
        class_rate.append(cr)

        print("Epoch {} train cost = {}, validation cost = {}, "
              "generalization loss = {:.3f}, GQ = {:.3f}, classification rate = {:.3f} ({:.1f}sec)"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if val_cost < best_val:
            best_val = val_cost
            best_conf = val_conf
            best_cr = cr
        else:
            if epoch >= t1 and (update_rule == 'sgdm' or update_rule == 'sgdnm'):
                lr.set_value(max(lr.get_value() * lr_decay, 0.001))
                if mm_schedule:
                    mm.set_value(mm_schedule.pop(0))

        if epoch >= validation_window and early_stop2(val_window, best_val, validation_window):
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
    if not options['no_plot']:
        plot_confusion_matrix(best_conf, letters, fmt='latex')
        plot_validation_cost(cost_train, cost_val, class_rate, 'e2e_valid_cost')

    if 'write_results' in options:
        results_file = options['write_results']
        with open(results_file, mode='a') as f:
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(update_rule, learning_rate, decay_rate, momentum,
                                                          decay_start, t1, validation_window,
                                                          weight_init, use_peepholes))

            s = ','.join([str(v) for v in cost_train])
            f.write('{}\n'.format(s))

            s = ','.join([str(v) for v in cost_val])
            f.write('{}\n'.format(s))

            s = ','.join([str(v) for v in class_rate])
            f.write('{}\n'.format(s))

            f.write('{},{},{}\n'.format(fusiontype, best_cr, best_val))

if __name__ == '__main__':
    main()
