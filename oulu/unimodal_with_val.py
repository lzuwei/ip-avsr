from __future__ import print_function
import sys
sys.path.insert(0, '../')
import pickle
import time
import ConfigParser
import argparse

import matplotlib
matplotlib.use('Agg')

from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from utils.io import *
from custom.objectives import temporal_softmax_loss
from custom.nonlinearities import *
from utils.regularization import early_stop2

import theano.tensor as T
import theano

import lasagne as las
import numpy as np
from lasagne.updates import adam

from modelzoo import deltanet, deltanet_majority_vote


def load_dbn(path='models/oulu_ae.mat'):
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


def print_layer_shape(layer):
    print('[L] {}: {}'.format(layer.name, las.layers.get_output_shape(layer)))


def print_network(network):
    layers = las.layers.get_all_layers(network)
    for layer in layers:
        print_layer_shape(layer)


def split_data(X, y, subjects, video_lens, train_ids, val_ids, test_ids):
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


def read_data_split_file(path, sep=','):
    with open(path) as f:
        subjects = f.readline().split(sep)
        subjects = [int(s) for s in subjects]
    return subjects


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
    :param window_size: size of window for computing delta coefficients
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, window_size)
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


def parse_options():
    options = dict()
    options['config'] = 'config/unimodal.ini'
    options['write_results'] = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file to use, default=config/unimodal.ini')
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

    print(config.items('data'))
    print(config.items('models'))
    print(config.items('training'))

    print('preprocessing dataset...')
    data = load_mat_file(config.get('data', 'images'))

    ae_pretrained = config.get('models', 'pretrained')
    lstm_units = int(config.get('models', 'lstm_units'))
    output_classes = int(config.get('models', 'output_classes'))
    weight_init = config.get('models', 'weight_init')
    delta_window = config.getint('models', 'delta_window')
    nonlinearity = select_nonlinearity(config.get('models', 'nonlinearity'))

    weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'glorot':
        weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'norm':
        weight_init_fn = las.init.Normal(0.1)
    if weight_init == 'uniform':
        weight_init_fn = las.init.Uniform()
    if weight_init == 'ortho':
        weight_init_fn = las.init.Orthogonal()

    learning_rate = float(config.get('training', 'learning_rate'))
    no_epochs = config.getint('training', 'no_epochs')
    use_peepholes = config.getboolean('training', 'use_peepholes')
    epochsize = config.getint('training', 'epochsize')
    batchsize = config.getint('training', 'batchsize')
    validation_window = config.getint('training', 'validation_window')

    # 53 subjects, 70 utterances, 5 view angles
    # s[x]_v[y]_u[z].mp4
    # resized, height, width = (26, 44)
    # ['dataMatrix', 'targetH', 'targetsPerVideoVec', 'videoLengthVec', '__header__', 'targetsVec',
    # '__globals__', 'iterVec', 'filenamesVec', 'dataMatrixCells', 'subjectsVec', 'targetW', '__version__']

    print(data.keys())
    X = data['dataMatrix'].astype('float32')  # .reshape((-1, 26, 44), order='f').reshape((-1, 26 * 44))
    y = data['targetsVec'].astype('int32')
    y = y.reshape((len(y),))
    uniques = np.unique(y)
    print('number of classifications: {}'.format(len(uniques)))
    subjects = data['subjectsVec'].astype('int')
    subjects = subjects.reshape((len(subjects),))
    video_lens = data['videoLengthVec'].astype('int')
    video_lens = video_lens.reshape((len(video_lens,)))

    train_subject_ids = read_data_split_file('data/train.txt')
    val_subject_ids = read_data_split_file('data/val.txt')
    test_subject_ids = read_data_split_file('data/test.txt')
    print('Train: {}'.format(train_subject_ids))
    print('Validation: {}'.format(val_subject_ids))
    print('Test: {}'.format(test_subject_ids))
    train_X, train_y, train_vidlens, train_subjects, \
    val_X, val_y, val_vidlens, val_subjects, \
    test_X, test_y, test_vidlens, test_subjects = \
        split_data(X, y, subjects, video_lens, train_subject_ids, val_subject_ids, test_subject_ids)

    assert train_X.shape[0] + val_X.shape[0] + test_X.shape[0] == len(X)
    assert train_y.shape[0] + val_y.shape[0] + test_y.shape[0] == len(y)
    assert train_vidlens.shape[0] + val_vidlens.shape[0] + test_vidlens.shape[0] == len(video_lens)
    assert train_subjects.shape[0] + val_subjects.shape[0] + test_subjects.shape[0] == len(subjects)

    train_X = normalize_input(train_X, centralize=True)
    val_X = normalize_input(val_X, centralize=True)
    test_X = normalize_input(test_X, centralize=True)

    weights, biases = load_dbn(ae_pretrained)

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
    network = deltanet_majority_vote.create_model_using_pretrained_encoder(weights, biases, (None, None, 1144), inputs,
                                                                           (None, None), mask, lstm_units,
                                                                           window, output_classes,
                                                                           weight_init_fn, use_peepholes, nonlinearity)

    print_network(network)
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = temporal_softmax_loss(predictions, targets, mask)
    updates = adam(cost, all_params, learning_rate)

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
            train(X, y, m, delta_window)
            print('\r', end='')
        cost = compute_train_cost(X, y, m, delta_window)
        val_cost = compute_test_cost(X_val, y_val, mask_val, delta_window)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model2(X_val, y_val_evaluate, mask_val, delta_window, val_fn)
        class_rate.append(cr)

        if val_cost < best_val:
            best_val = val_cost
            test_cr, test_conf = evaluate_model2(X_test, y_test, mask_test, delta_window, val_fn)
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
    print('classification rate: {}, validation loss: {}'.format(test_cr, best_val))
    print('confusion matrix: ')
    plot_confusion_matrix(test_conf, phrases, fmt='grid')
    plot_validation_cost(cost_train, cost_val, class_rate)

    # fc1 = las.layers.get_all_layers(network)[2]
    # visualize_layer(fc1, 26, 44, 20, 20)

    # play with visualizations
    # fc1 = las.layers.get_all_layers(network)[2]
    # w = fc1.W.get_value()
    # trainexamples = train_X[:100]
    # trainexamples = reshape_images_order(trainexamples, (26, 44))
    # visualize_activations(w, trainexamples, (26, 44), range(20), 'results/deltanet')

if __name__== '__main__':
    main()
