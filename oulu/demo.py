from __future__ import print_function
import numpy as np
import cv2
import sys
sys.path.insert(0, '../')
import pickle
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
from utils.draw_net import draw_to_file

import theano.tensor as T
import theano
from custom_layers.custom import DeltaLayer
from nolearn.lasagne import NeuralNet

import lasagne as las
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, LSTMLayer, Gate, ElemwiseSumLayer, SliceLayer
from lasagne.layers import ReshapeLayer, DimshuffleLayer, ConcatLayer
from lasagne.nonlinearities import tanh, linear, sigmoid, rectify
from lasagne.updates import nesterov_momentum, adadelta, sgd, norm_constraint, adagrad
from lasagne.objectives import squared_error

from modelzoo import adenet_v3
from utils.plotting_utils import print_network


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
        (InputLayer, {'name': 'input', 'shape': (None, 1144)}),
        (DenseLayer, {'name': 'l1', 'num_units': 2000, 'nonlinearity': sigmoid, 'W': w1, 'b': b1}),
        (DenseLayer, {'name': 'l2', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w2, 'b': b2}),
        (DenseLayer, {'name': 'l3', 'num_units': 500, 'nonlinearity': sigmoid, 'W': w3, 'b': b3}),
        (DenseLayer, {'name': 'l4', 'num_units': 50, 'nonlinearity': linear, 'W': w4, 'b': b4}),
        (DenseLayer, {'name': 'l5', 'num_units': 500, 'nonlinearity': sigmoid, 'W': w5, 'b': b5}),
        (DenseLayer, {'name': 'l6', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w6, 'b': b6}),
        (DenseLayer, {'name': 'l7', 'num_units': 2000, 'nonlinearity': sigmoid, 'W': w7, 'b': b7}),
        (DenseLayer, {'name': 'output', 'num_units': 1144, 'nonlinearity': linear, 'W': w8, 'b': b8}),
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


def extract_encoder(dbn):
    dbn_layers = dbn.get_all_layers()
    encoder = NeuralNet(
        layers=[
            (InputLayer, {'name': 'input', 'shape': dbn_layers[0].shape}),
            (DenseLayer, {'name': 'l1', 'num_units': dbn_layers[1].num_units, 'nonlinearity': sigmoid,
                          'W': dbn_layers[1].W, 'b': dbn_layers[1].b}),
            (DenseLayer, {'name': 'l2', 'num_units': dbn_layers[2].num_units, 'nonlinearity': sigmoid,
                          'W': dbn_layers[2].W, 'b': dbn_layers[2].b}),
            (DenseLayer, {'name': 'l3', 'num_units': dbn_layers[3].num_units, 'nonlinearity': sigmoid,
                          'W': dbn_layers[3].W, 'b': dbn_layers[3].b}),
            (DenseLayer, {'name': 'l4', 'num_units': dbn_layers[4].num_units, 'nonlinearity': linear,
                          'W': dbn_layers[4].W, 'b': dbn_layers[4].b}),
        ],
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.5,
        objective_l2=0.005,
        verbose=1,
        regression=True
    )
    encoder.initialize()
    return encoder


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def split_data(X, y, dct, X_diff, subjects, video_lens, train_ids, val_ids, test_ids, target_filenames):
    """
    Splits the data into training and testing sets
    :param X: input X
    :param y: target y
    :param dct: dct features
    :param X_diff: difference images
    :param subjects: array of video -> subject mapping
    :param video_lens: array of video lengths for each video
    :param train_ids: list of subject ids used for training
    :param val_ids: list of subject ids used for validation
    :param test_ids: list of subject ids used for testing
    :param target_filenames: list of target video filenames
    :return: split data
    """
    # construct a subjects data matrix offset
    X_feature_dim = X.shape[1]
    X_diff_feature_dim = X_diff.shape[1]
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
    train_X_diff = np.empty((0, X_diff_feature_dim), dtype='float32')
    val_X_diff = np.empty((0, X_diff_feature_dim), dtype='float32')
    test_X_diff = np.empty((0, X_diff_feature_dim), dtype='float32')
    train_vidlens = np.empty((0,), dtype='int')
    val_vidlens = np.empty((0,), dtype='int')
    test_vidlens = np.empty((0,), dtype='int')
    train_subjects = np.empty((0,), dtype='int')
    val_subjects = np.empty((0,), dtype='int')
    test_subjects = np.empty((0,), dtype='int')
    train_filenames = list()
    val_filenames = list()
    test_filenames = list()
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
                train_X_diff = np.concatenate((train_X_diff, X_diff[current_data_idx:end_data_idx]))
                train_dct = np.concatenate((train_dct, dct[current_data_idx:end_data_idx]))
                train_vidlens = np.concatenate((train_vidlens, video_lens[current_video_idx:end_video_idx]))
                train_subjects = np.concatenate((train_subjects, subjects[current_video_idx:end_video_idx]))
                train_filenames = train_filenames + target_filenames[current_video_idx:end_video_idx]
            elif previous_subject in val_ids:
                val_X = np.concatenate((val_X, X[current_data_idx:end_data_idx]))
                val_y = np.concatenate((val_y, y[current_data_idx:end_data_idx]))
                val_X_diff = np.concatenate((val_X_diff, X_diff[current_data_idx:end_data_idx]))
                val_dct = np.concatenate((val_dct, dct[current_data_idx:end_data_idx]))
                val_vidlens = np.concatenate((val_vidlens, video_lens[current_video_idx:end_video_idx]))
                val_subjects = np.concatenate((val_subjects, subjects[current_video_idx:end_video_idx]))
                val_filenames = val_filenames + target_filenames[current_video_idx:end_video_idx]
            else:
                test_X = np.concatenate((test_X, X[current_data_idx:end_data_idx]))
                test_y = np.concatenate((test_y, y[current_data_idx:end_data_idx]))
                test_dct = np.concatenate((test_dct, dct[current_data_idx:end_data_idx]))
                test_X_diff = np.concatenate((test_X_diff, X_diff[current_data_idx:end_data_idx]))
                test_vidlens = np.concatenate((test_vidlens, video_lens[current_video_idx:end_video_idx]))
                test_subjects = np.concatenate((test_subjects, subjects[current_video_idx:end_video_idx]))
                test_filenames = test_filenames + target_filenames[current_video_idx:end_video_idx]
            previous_subject = subject
            current_video_idx = end_video_idx
            current_data_idx = end_data_idx
            subject_video_count = 1
            populate = False
    return train_X, train_y, train_dct, train_X_diff, train_vidlens, train_subjects, train_filenames, \
           val_X, val_y, val_dct, val_X_diff, val_vidlens, val_subjects, val_filenames, \
           test_X, test_y, test_dct, test_X_diff, test_vidlens, test_subjects, test_filenames


def read_data_split_file(path, sep=','):
    with open(path) as f:
        subjects = f.readline().split(sep)
        subjects = [int(s) for s in subjects]
    return subjects


def create_pretrained_encoder(weights, biases, incoming):
    l_1 = DenseLayer(incoming, 2000, W=weights[0], b=biases[0], nonlinearity=sigmoid, name='fc1')
    l_2 = DenseLayer(l_1, 1000, W=weights[1], b=biases[1], nonlinearity=sigmoid, name='fc2')
    l_3 = DenseLayer(l_2, 500, W=weights[2], b=biases[2], nonlinearity=sigmoid, name='fc3')
    l_4 = DenseLayer(l_3, 50, W=weights[3], b=biases[3], nonlinearity=linear, name='encoder')
    return l_4


def evaluate_model(X_val, y_val, mask_val, dct_val, X_diff_val, window_size, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param dct_val: validation dct features
    :param X_diff_val: validation inputs diff image
    :param window_size: size of window for computing delta coefficients
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, dct_val, X_diff_val, window_size)
    no_gps = output.shape[1]
    confusion_matrix = np.zeros((no_gps, no_gps), dtype='int')

    ix = np.argmax(output, axis=1)
    c = ix == y_val
    classification_rate = np.sum(c == True) / float(len(c))

    # construct the confusion matrix
    for i, target in enumerate(y_val):
        confusion_matrix[target, ix[i]] += 1

    return classification_rate, confusion_matrix


def get_phrase(idx):
    phrases = ['Excuse me', 'Good bye', 'Hello', 'How are you', 'Nice to meet you',
                'See you', 'I am sorry', 'Thank you', 'Have a good time', 'You are welcome']
    return phrases[idx]


def parse_options():
    options = dict()
    options['config'] = 'config/trimodal.ini'
    options['write_results'] = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file to use, default=config/trimodal.ini')
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
    ae_finetuned_diff = config.get('models', 'finetuned_diff')
    fusiontype = config.get('models', 'fusiontype')
    learning_rate = float(config.get('training', 'learning_rate'))
    decay_rate = float(config.get('training', 'decay_rate'))
    decay_start = int(config.get('training', 'decay_start'))
    do_finetune = config.getboolean('training', 'do_finetune')
    save_finetune = config.getboolean('training', 'save_finetune')
    load_finetune = config.getboolean('training', 'load_finetune')
    load_finetune_diff = config.getboolean('training', 'load_finetune_diff')
    savemodel = config.getboolean('training', 'savemodel')
    t = data['filenamesVec']
    target_filenames = list()
    for r in t:
        for j in r:
            target_filenames.append(str(j[0]))

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
    X_diff = compute_diff_images(X, video_lens)

    # mean remove dct features
    dct_feats = sequencewise_mean_image_subtraction(dct_feats, video_lens)

    train_subject_ids = read_data_split_file('data/train.txt')
    val_subject_ids = read_data_split_file('data/val.txt')
    test_subject_ids = read_data_split_file('data/test.txt')
    print('Train: {}'.format(train_subject_ids))
    print('Validation: {}'.format(val_subject_ids))
    print('Test: {}'.format(test_subject_ids))
    train_X, train_y, train_dct, train_X_diff, train_vidlens, train_subjects, train_filenames, \
    val_X, val_y, val_dct, val_X_diff, val_vidlens, val_subjects, val_filenames, \
    test_X, test_y, test_dct, test_X_diff, test_vidlens, test_subjects, test_filenames = \
        split_data(X, y, dct_feats, X_diff, subjects, video_lens, train_subject_ids,
                   val_subject_ids, test_subject_ids, target_filenames)

    assert train_X.shape[0] + val_X.shape[0] + test_X.shape[0] == len(X)
    assert train_y.shape[0] + val_y.shape[0] + test_y.shape[0] == len(y)
    assert train_vidlens.shape[0] + val_vidlens.shape[0] + test_vidlens.shape[0] == len(video_lens)
    assert train_subjects.shape[0] + val_vidlens.shape[0] + test_subjects.shape[0] == len(subjects)

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

    if load_finetune:
        print('loading finetuned encoder: {}...'.format(ae_finetuned))
        ae = pickle.load(open(ae_finetuned, 'rb'))
        ae.initialize()

    if load_finetune_diff:
        print('loading finetuned encoder: {}...'.format(ae_finetuned_diff))
        ae_diff = pickle.load(open(ae_finetuned_diff, 'rb'))
        ae_diff.initialize()

    window = T.iscalar('theta')
    dct = T.tensor3('dct', dtype='float32')
    inputs = T.tensor3('inputs', dtype='float32')
    inputs_diff = T.tensor3('inputs_diff', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.ivector('targets')

    print('loading end to end model...')
    network, l_fuse = adenet_v3.create_model(ae, ae_diff, (None, None, 1144), inputs,
                                             (None, None), mask,
                                             (None, None, 90), dct,
                                             (None, None, 1144), inputs_diff,
                                             250, window, 10, fusiontype)
    all_param_values = load_model('models/3stream.dat')
    all_params = las.layers.get_all_params(network)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)

    print_network(network)
    print('compiling model...')

    '''
    train = theano.function(
        [inputs, targets, mask, dct, inputs_diff, window],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs, targets, mask, dct, inputs_diff, window],
                                         cost, allow_input_downcast=True)

    '''
    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_predictions, targets))
    val_fn = theano.function([inputs, mask, dct, inputs_diff, window], test_predictions, allow_input_downcast=True)

    datagen = gen_lstm_batch_random(train_X, train_y, train_vidlens, batchsize=10)
    integral_lens = compute_integral_len(train_vidlens)

    val_datagen = gen_lstm_batch_random(val_X, val_y, val_vidlens, batchsize=len(val_vidlens))
    test_datagen = gen_lstm_batch_random(test_X, test_y, test_vidlens, batchsize=len(test_vidlens))

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, idxs_val = next(val_datagen)
    integral_lens_val = compute_integral_len(val_vidlens)
    dct_val = gen_seq_batch_from_idx(val_dct, idxs_val, val_vidlens, integral_lens_val, np.max(val_vidlens))
    X_diff_val = gen_seq_batch_from_idx(val_X_diff, idxs_val, val_vidlens, integral_lens_val, np.max(val_vidlens))

    # we use the test set to check final classification rate
    X_test, y_test, mask_test, idxs_test = next(test_datagen)
    integral_lens_test = compute_integral_len(test_vidlens)
    dct_test = gen_seq_batch_from_idx(test_dct, idxs_test, test_vidlens, integral_lens_test, np.max(test_vidlens))
    X_diff_test = gen_seq_batch_from_idx(test_X_diff, idxs_test, test_vidlens, integral_lens_test, np.max(test_vidlens))

    raw_input("Press Enter to start demo...")

    for idx in range(len(X_test)):
        videofile = '../examples/data/{}'.format(test_filenames[idxs_test[idx]])
        print('video file: {}'.format(videofile))
        cap = cv2.VideoCapture(videofile)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('frame', gray)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pred = val_fn(X_test[idx:idx + 1], mask_test[idx:idx + 1], dct_test[idx:idx + 1], X_diff_test[idx:idx + 1], 9)
        pred_idx = np.argmax(pred, axis=1)
        print('Prediction: {}, Target: {}'.format(get_phrase(pred_idx[0]), get_phrase(y_test[idx])))
        raw_input("Press Enter to continue...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
