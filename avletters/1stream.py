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
from custom.nonlinearities import select_nonlinearity

import theano.tensor as T
import theano

import lasagne as las
import numpy as np
from lasagne.updates import adam

from modelzoo import deltanet_majority_vote, deltanet_v1
from utils.plotting_utils import print_network


def load_decoder(path, shapes, nonlinearities):
    nn = sio.loadmat(path)
    weights = []
    biases = []
    shapes = [int(s) for s in shapes.split(',')]
    nonlinearities = [select_nonlinearity(nonlinearity) for nonlinearity in nonlinearities.split(',')]
    for i in range(len(shapes)):
        weights.append(nn['w{}'.format(i+1)].astype('float32'))
        biases.append(nn['b{}'.format(i+1)][0].astype('float32'))
    return weights, biases, shapes, nonlinearities


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


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


def presplit_dataprocessing(data_matrix, vidlens, config, stream_name, **kwargs):
    reorderdata = config.getboolean(stream_name, 'reorderdata')
    diffimage = config.getboolean(stream_name, 'diffimage')
    meanremove = config.getboolean(stream_name, 'meanremove')
    samplewisenormalize = config.getboolean(stream_name, 'samplewisenormalize')
    if reorderdata:
        imagesize = kwargs['imagesize']
        data_matrix = reorder_data(data_matrix, imagesize)
    if meanremove:
        data_matrix = sequencewise_mean_image_subtraction(data_matrix, vidlens)
    if diffimage:
        data_matrix = compute_diff_images(data_matrix, vidlens)
    if samplewisenormalize:
        data_matrix = normalize_input(data_matrix)
    return data_matrix


def postsplit_datapreprocessing(train_X, val_X, config, stream_name):
    featurewisenormalize = config.getboolean(stream_name, 'featurewisenormalize')
    if featurewisenormalize:
        train_X, mean, std = featurewise_normalize_sequence(train_X)
        val_X = (val_X - mean) / std
    return train_X, val_X


def parse_options():
    options = dict()
    options['config'] = 'config/1stream_raw.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='[CONFIG_FILE] config file to use, default=../cuave/config/1stream.ini')
    parser.add_argument('--write_results', help='[FILE] write results to file')
    parser.add_argument('--learning_rate', help='[LEARNING_RATE] learning rate')
    parser.add_argument('--save_best', help='[FILE] save the best model')
    parser.add_argument('--save_plot', help='[FILE_PREFIX] plot the train/validation '
                                            'loss curve using user supplied prefix')
    args = parser.parse_args()
    if args.config:
        options['config'] = args.config
    if args.write_results:
        options['write_results'] = args.write_results
    if args.learning_rate:
        options['learning_rate'] = float(args.learning_rate)
    if args.save_best:
        options['save_best'] = args.save_best
    if args.save_plot:
        options['save_plot'] = args.save_plot
    return options


def main():
    configure_theano()
    options = parse_options()
    config_file = options['config']
    config = ConfigParser.ConfigParser()
    config.read(config_file)

    print('CLI options: {}'.format(options.items()))

    print('Reading Config File: {}...'.format(config_file))
    print(config.items('stream1'))
    print(config.items('lstm_classifier'))
    print(config.items('training'))

    print('preprocessing dataset...')

    # stream 1
    s1_data = load_mat_file(config.get('stream1', 'data'))
    s1_imagesize = tuple([int(d) for d in config.get('stream1', 'imagesize').split(',')])
    s1 = config.get('stream1', 'model')
    s1_inputdim = config.getint('stream1', 'input_dimensions')
    s1_shape = config.get('stream1', 'shape')
    s1_nonlinearities = config.get('stream1', 'nonlinearities')

    # lstm classifier
    fusiontype = config.get('lstm_classifier', 'fusiontype')
    weight_init = options['weight_init'] if 'weight_init' in options else config.get('lstm_classifier', 'weight_init')
    use_peepholes = options['use_peepholes'] if 'use_peepholes' in options else config.getboolean('lstm_classifier',
                                                                                                  'use_peepholes')
    use_blstm = True if config.has_option('lstm_classifier', 'use_blstm') else False

    windowsize = config.getint('lstm_classifier', 'windowsize')
    output_classes = config.getint('lstm_classifier', 'output_classes')
    output_classnames = config.get('lstm_classifier', 'output_classnames').split(',')
    lstm_size = config.getint('lstm_classifier', 'lstm_size')
    matlab_target_offset = config.getboolean('lstm_classifier', 'matlab_target_offset')

    # capture training parameters
    validation_window = int(options['validation_window']) \
        if 'validation_window' in options else config.getint('training', 'validation_window')
    num_epoch = int(options['num_epoch']) if 'num_epoch' in options else config.getint('training', 'num_epoch')
    learning_rate = options['learning_rate'] if 'learning_rate' in options \
        else config.getfloat('training', 'learning_rate')
    epochsize = config.getint('training', 'epochsize')
    batchsize = config.getint('training', 'batchsize')

    weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'glorot':
        weight_init_fn = las.init.GlorotUniform()
    if weight_init == 'norm':
        weight_init_fn = las.init.Normal(0.1)
    if weight_init == 'uniform':
        weight_init_fn = las.init.Uniform()
    if weight_init == 'ortho':
        weight_init_fn = las.init.Orthogonal()

    s1_data_matrix = s1_data['dataMatrix'].astype('float32')

    #visualize_images(s1_data_matrix[800:864])

    targets_vec = s1_data['targetsVec'].reshape((-1,))
    subjects_vec = s1_data['subjectsVec'].reshape((-1,))
    vidlen_vec = s1_data['videoLengthVec'].reshape((-1,))
    iter_vec = s1_data['iterVec'].reshape((-1,))

    if matlab_target_offset:
        targets_vec -= 1

    s1_data_matrix = presplit_dataprocessing(s1_data_matrix, vidlen_vec, config, 'stream1', imagesize=s1_imagesize)

    indexes = create_split_index(len(s1_data_matrix), vidlen_vec, iter_vec)
    s1_train_vidlens, s1_val_vidlens = split_videolen(vidlen_vec, iter_vec)

    # split the data
    s1_train_X = s1_data_matrix[indexes == True]
    s1_train_y = targets_vec[indexes == True].reshape((-1,))
    s1_val_X = s1_data_matrix[indexes == False]
    s1_val_y = targets_vec[indexes == False].reshape((-1,))

    s1_train_X, s1_val_X = postsplit_datapreprocessing(s1_train_X, s1_val_X, config, 'stream1')

    s1_has_encoder = config.getboolean('stream1', 'has_encoder')

    # IMPT: the encoder was trained with fortan ordered images, so to visualize
    # convert all the images to C order using reshape_images_order()
    # output = dbn.predict(test_X)
    # test_X = reshape_images_order(test_X, (26, 44))
    # output = reshape_images_order(output, (26, 44))
    # visualize_reconstruction(test_X[:36, :], output[:36, :], shape=(26, 44))

    window = T.iscalar('theta')
    inputs1 = T.tensor3('inputs1', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.imatrix('targets')

    print('constructing end to end model...')
    if s1_has_encoder:
        ae1 = load_decoder(s1, s1_shape, s1_nonlinearities)
        network = deltanet_majority_vote.create_model(ae1, (None, None, s1_inputdim), inputs1,
                                                      (None, None), mask,
                                                      lstm_size, window, output_classes, weight_init_fn,
                                                      use_peepholes, use_blstm)
    else:
        network = deltanet_v1.create_model((None, None, s1_inputdim), inputs1,
                                           (None, None), mask, window,
                                           lstm_size, output_classes, weight_init_fn, use_peepholes, use_blstm)
    print_network(network)
    # draw_to_file(las.layers.get_all_layers(network), 'network.png')
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = temporal_softmax_loss(predictions, targets, mask)
    updates = adam(cost, all_params, learning_rate=learning_rate)

    train = theano.function(
        [inputs1, targets, mask, window],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs1, targets, mask, window],
                                         cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = temporal_softmax_loss(test_predictions, targets, mask)
    compute_test_cost = theano.function(
        [inputs1, targets, mask, window], test_cost, allow_input_downcast=True)

    val_fn = theano.function([inputs1, mask, window], test_predictions, allow_input_downcast=True)

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

    datagen = gen_lstm_batch_random(s1_train_X, s1_train_y, s1_train_vidlens, batchsize=batchsize)
    val_datagen = gen_lstm_batch_random(s1_val_X, s1_val_y, s1_val_vidlens, batchsize=len(s1_val_vidlens))

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, idxs_val = next(val_datagen)

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
            print_str = 'Epoch {} batch {}/{}: {} examples using adam with learning rate = {}'.format(
                epoch + 1, i + 1, epochsize, len(X), learning_rate)
            print(print_str, end='')
            sys.stdout.flush()
            train(X, y, m, windowsize)
            print('\r', end='')
        cost = compute_train_cost(X, y, m, windowsize)
        val_cost = compute_test_cost(X_val, y_val, mask_val, windowsize)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model2(X_val, y_val_evaluate, mask_val, windowsize, val_fn)
        class_rate.append(cr)

        print("Epoch {} train cost = {}, val cost = {}, "
              "GL loss = {:.3f}, GQ = {:.3f}, CR = {:.3f} ({:.1f}sec)"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if val_cost < best_val:
            best_val = val_cost
            best_cr = cr
            best_conf = val_conf
            best_params = las.layers.get_all_param_values(network)

        if epoch >= validation_window and early_stop2(val_window, best_val, validation_window):
            break

    print('Final Model')
    print('CR: {}, val loss: {}'.format(best_cr, best_val))

    # plot confusion matrix
    table_str = plot_confusion_matrix(best_conf, output_classnames, fmt='pipe')
    print('confusion matrix: ')
    print(table_str)

    if 'save_plot' in options:
        prefix = options['save_plot']
        plot_validation_cost(cost_train, cost_val, savefilename='{}.validloss.png'.format(prefix))
        with open('{}.confmat.txt'.format(prefix), mode='a') as f:
            f.write(table_str)
            f.write('\n\n')

    if 'write_results' in options:
        print('writing results to {}'.format(options['write_results']))
        results_file = options['write_results']
        with open(results_file, mode='a') as f:
            f.write('{},{}\n'.format(best_cr, best_val))

    if 'save_best' in options:
        print('saving best model...')
        las.layers.set_all_param_values(network, best_params)
        save_model_params(network, options['save_best'])
        print('best model saved to {}'.format(options['save_best']))


if __name__ == '__main__':
    main()
