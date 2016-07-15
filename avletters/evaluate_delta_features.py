import sys
sys.path.insert(0, '../')
import time
import pickle
import theano
import theano.tensor as T

import lasagne as las
from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from custom_layers.custom import DeltaLayer
from modelzoo import adenet_v1, deltanet, adenet_v2, adenet_v3, autoencoder, deltanet_v1
from end_to_end import load_finetuned_dbn

import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, LSTMLayer, Gate, ElemwiseSumLayer, SliceLayer
from lasagne.layers import ReshapeLayer, DimshuffleLayer, ConcatLayer, BatchNormLayer, batch_norm
from lasagne.nonlinearities import tanh, linear, sigmoid, rectify, leaky_rectify
from lasagne.updates import nesterov_momentum, adadelta, sgd, norm_constraint
from lasagne.objectives import squared_error


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def compile_delta_features():
    # create input
    input_var = T.tensor3('input', dtype='float32')
    win_var = T.iscalar('theta')
    weights, biases = autoencoder.load_dbn()

    '''
    activations = [sigmoid, sigmoid, sigmoid, linear, sigmoid, sigmoid, sigmoid, linear]
    layersizes = [2000, 1000, 500, 50, 500, 1000, 2000, 1200]
    ae = autoencoder.create_model(l_input, weights, biases, activations, layersizes)
    print_network(ae)
    reconstruct = las.layers.get_output(ae)
    reconstruction_fn = theano.function([input_var], reconstruct, allow_input_downcast=True)
    recon_img = reconstruction_fn(test_data_resized)
    visualize_reconstruction(test_data_resized[225:250], recon_img[225:250])
    '''
    l_input = InputLayer((None, None, 1200), input_var, name='input')

    symbolic_batchsize = l_input.input_var.shape[0]
    symbolic_seqlen = l_input.input_var.shape[1]
    en_activations = [sigmoid, sigmoid, sigmoid, linear]
    en_layersizes = [2000, 1000, 500, 50]

    l_reshape1 = ReshapeLayer(l_input, (-1, l_input.shape[-1]), name='reshape1')
    l_encoder = autoencoder.create_model(l_reshape1, weights[:4], biases[:4], en_activations, en_layersizes)
    encoder_len = las.layers.get_output_shape(l_encoder)[-1]
    l_reshape2 = ReshapeLayer(l_encoder, (symbolic_batchsize, symbolic_seqlen, encoder_len), name='reshape2')
    l_delta = DeltaLayer(l_reshape2, win_var, name='delta')
    l_slice = SliceLayer(l_delta, indices=slice(50, None), axis=-1, name='slice')  # extract the delta coefficients
    l_reshape3 = ReshapeLayer(l_slice, (-1, l_slice.output_shape[-1]), name='reshape3')
    print_network(l_reshape3)

    delta_features = las.layers.get_output(l_reshape3)
    delta_fn = theano.function([input_var, win_var], delta_features, allow_input_downcast=True)

    return delta_fn


def compile_encoder(encoderpath=None):
    # create input
    if encoderpath:
        l_encoder = pickle.load(open(encoderpath, 'rb'))
        input_var = las.layers.get_all_layers(l_encoder)[0].input_var
        visualize_layer(las.layers.get_all_layers(l_encoder)[2], 40, 30)
    else:
        input_var = T.matrix('input', dtype='float32')
        weights, biases = autoencoder.load_dbn()
        en_activations = [sigmoid, sigmoid, sigmoid, linear]
        en_layersizes = [2000, 1000, 500, 50]
        l_input = InputLayer((None, 1200), input_var, name='input')
        l_encoder = autoencoder.create_model(l_input, weights[:4], biases[:4], en_activations, en_layersizes)
    print_network(l_encoder)

    encoded_features = las.layers.get_output(l_encoder)
    encode_fn = theano.function([input_var], encoded_features, allow_input_downcast=True)
    return encode_fn


def generate_lstm_parameters():
    gate_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)
    return gate_parameters, cell_parameters


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


def evaluate_model1(X_val, y_val, mask_val, win_var, eval_fn):
    """
    Evaluate a lstm model
    :param X_val: validation inputs
    :param y_val: validation targets
    :param mask_val: input masks for variable sequences
    :param eval_fn: evaluation function
    :return: classification rate, confusion matrix
    """
    output = eval_fn(X_val, mask_val, win_var)
    no_gps = output.shape[1]
    confusion_matrix = np.zeros((no_gps, no_gps), dtype='int')

    ix = np.argmax(output, axis=1)
    c = ix == y_val
    classification_rate = np.sum(c == True) / float(len(c))

    # construct the confusion matrix
    for i, target in enumerate(y_val):
        confusion_matrix[target, ix[i]] += 1

    return classification_rate, confusion_matrix


def concat_first_second_deltas(X, vidlenvec):
    """
    Compute and concatenate 1st and 2nd order derivatives of input X given a sequence list
    :param X: input feature vector X
    :param vidlenvec: temporal sequence of X
    :return: A matrix of shape(num rows of intput X, X + 1st order X + 2nd order X)
    """
    # construct a new feature matrix
    feature_len = X.shape[1]
    Y = np.zeros((X.shape[0], feature_len * 3))  # new feature vector with 1st, 2nd delta
    start = 0
    for vidlen in vidlenvec:
        end = start + vidlen
        seq = X[start: end]  # (vidlen, feature_len)
        first_order = deltas(seq.T)
        second_order = deltas(first_order)
        assert first_order.shape == (feature_len, vidlen)
        assert second_order.shape == (feature_len, vidlen)
        assert len(seq) == vidlen
        seq = np.concatenate((seq, first_order.T, second_order.T), axis=1)
        for idx, j in enumerate(range(start, end)):
            Y[j] = seq[idx]
        start += vidlen
    return Y


def main():
    configure_theano()
    print('preprocessing dataset...')
    data = load_mat_file('data/allData_mouthROIs.mat')

    # create the necessary variable mappings
    data_matrix = data['dataMatrix']
    data_matrix_len = data_matrix.shape[0]
    targets_vec = data['targetsVec']
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']

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

    # indexes for a particular letter
    # idx = [i for i, elem in enumerate(test_targets) if elem == 20]

    # resize the input data to 40 x 30
    train_data_resized = resize_images(train_data).astype(np.float32)

    # normalize the inputs [0 - 1]
    train_data_resized = normalize_input(train_data_resized, centralize=True)

    test_data_resized = resize_images(test_data).astype(np.float32)
    test_data_resized = normalize_input(test_data_resized, centralize=True)

    print('compute delta features and featurewise normalize...')
    encode_fn = compile_encoder('models/end2end_encoder.dat')
    deltafeatures = concat_first_second_deltas(encode_fn(train_data_resized), train_vidlen_vec)[:, -100:]
    deltafeatures_val = concat_first_second_deltas(encode_fn(test_data_resized), test_vidlen_vec)[:, -100:]

    deltafeatures, mean, std = featurewise_normalize_sequence(deltafeatures)
    deltafeatures_val = (deltafeatures_val - mean) / std

    print('train delta features: {}'.format(deltafeatures.shape))
    print('validation delta features: {}'.format(deltafeatures_val.shape))

    gate_p, cell_p = generate_lstm_parameters()

    # create lstm
    input_var = T.tensor3('input', dtype='float32')
    mask_var = T.matrix('mask', dtype='uint8')
    target_var = T.ivector('target')
    window_var = T.iscalar('window')
    lr = theano.shared(np.array(0.7, dtype=theano.config.floatX), name='learning_rate')
    lr_decay = np.array(0.80, dtype=theano.config.floatX)

    l_input = InputLayer((None, None, 100), input_var, name='input')
    l_mask = InputLayer((None, None), mask_var, name='mask')
    l_lstm = LSTMLayer(
        l_input, 250,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_p, forgetgate=gate_p,
        cell=cell_p, outgate=gate_p,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm')
    l_forward_slice1 = SliceLayer(l_lstm, -1, 1, name='slice1')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    network = DenseLayer(
        l_forward_slice1, num_units=26, nonlinearity=las.nonlinearities.softmax, name='output')
    print_network(network)

    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.categorical_crossentropy(predictions, target_var))
    updates = las.updates.adadelta(cost, all_params, learning_rate=lr)
    # updates = las.updates.adam(cost, all_params, learning_rate=lr)

    train = theano.function(
        [input_var, target_var, mask_var],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([input_var, target_var, mask_var],
                                         cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_predictions, target_var))
    compute_test_cost = theano.function(
        [input_var, target_var, mask_var], test_cost, allow_input_downcast=True)

    val_fn = theano.function([input_var, mask_var], test_predictions, allow_input_downcast=True)

    # We'll train the network with 10 epochs of 30 minibatches each
    print('begin training...')
    cost_train = []
    cost_val = []
    class_rate = []
    NUM_EPOCHS = 40
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

    # create train and eval loop
    data_gen = gen_lstm_batch_random(deltafeatures, train_targets, train_vidlen_vec, batchsize=BATCH_SIZE)
    data_gen_val = gen_lstm_batch_random(deltafeatures_val, test_targets, test_vidlen_vec,
                                         batchsize=len(test_vidlen_vec))

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, _ = next(data_gen_val)

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
            X, y, m, _ = next(data_gen)
            train(X, y, m)
        cost = compute_train_cost(X, y, m)
        val_cost = compute_test_cost(X_val, y_val, mask_val)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model(X_val, y_val, mask_val, val_fn)
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
        if epoch > 8:
            lr.set_value(lr.get_value() * lr_decay)

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

    print('Best Model')
    print('classification rate: {}, validation loss: {}'.format(best_cr, best_val))
    print('confusion matrix: ')
    plot_confusion_matrix(best_conf, letters, fmt='grid')
    plot_validation_cost(cost_train, cost_val, class_rate)


def extract_encoder(network, inputshape, start, end):
    layers = las.layers.get_all_layers(network)
    weights = []
    biases = []
    activations = []
    layersizes = []

    for l in layers[start:end]:
        weights.append(l.W)
        biases.append(l.b)
        activations.append(l.nonlinearity)
        layersizes.append(l.num_units)

    input = T.matrix('input', dtype='float32')
    encoder = InputLayer(inputshape, input, name='input')
    encoder = autoencoder.create_pretrained_encoder(encoder, weights, biases, activations, layersizes)
    return encoder


def train_deltanet(save=True):
    configure_theano()
    print('preprocessing dataset...')
    data = load_mat_file('data/allData_mouthROIs.mat')

    # create the necessary variable mappings
    data_matrix = data['dataMatrix']
    data_matrix_len = data_matrix.shape[0]
    targets_vec = data['targetsVec']
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']

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

    # indexes for a particular letter
    # idx = [i for i, elem in enumerate(test_targets) if elem == 20]

    # resize the input data to 40 x 30
    train_data_resized = resize_images(train_data).astype(np.float32)

    # normalize the inputs [0 - 1]
    train_data_resized = normalize_input(train_data_resized, centralize=True)

    test_data_resized = resize_images(test_data).astype(np.float32)
    test_data_resized = normalize_input(test_data_resized, centralize=True)

    input_var = T.tensor3('input', dtype='float32')
    mask_var = T.matrix('mask', dtype='uint8')
    window_var = T.iscalar('window')
    target_var = T.ivector('target')
    lr = theano.shared(np.array(1.0, dtype=theano.config.floatX), name='learning_rate')
    lr_decay = np.array(0.90, dtype=theano.config.floatX)

    dbn = load_finetuned_dbn('models/dbn_finetune.dat')
    network = deltanet_v1.create_model(dbn, (None, None, 1200), input_var, (None, None), mask_var, 250, window_var)
    print_network(network)

    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.categorical_crossentropy(predictions, target_var))
    updates = las.updates.adadelta(cost, all_params, learning_rate=lr)
    # updates = las.updates.adam(cost, all_params, learning_rate=lr)

    train = theano.function(
        [input_var, target_var, mask_var, window_var],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([input_var, target_var, mask_var, window_var],
                                         cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_predictions, target_var))
    compute_test_cost = theano.function(
        [input_var, target_var, mask_var, window_var], test_cost, allow_input_downcast=True)

    val_fn = theano.function([input_var, mask_var, window_var], test_predictions, allow_input_downcast=True)

    # We'll train the network with 10 epochs of 30 minibatches each
    print('begin training...')
    cost_train = []
    cost_val = []
    class_rate = []
    NUM_EPOCHS = 40
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

    # create train and eval loop
    data_gen = gen_lstm_batch_random(train_data_resized, train_targets, train_vidlen_vec, batchsize=BATCH_SIZE)
    data_gen_val = gen_lstm_batch_random(test_data_resized, test_targets, test_vidlen_vec,
                                         batchsize=len(test_vidlen_vec))

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val, _ = next(data_gen_val)

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
            X, y, m, _ = next(data_gen)
            train(X, y, m, WINDOW_SIZE)
        cost = compute_train_cost(X, y, m, WINDOW_SIZE)
        val_cost = compute_test_cost(X_val, y_val, mask_val, WINDOW_SIZE)
        cost_train.append(cost)
        cost_val.append(val_cost)
        train_strip[epoch % STRIP_SIZE] = cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        cr, val_conf = evaluate_model1(X_val, y_val, mask_val, WINDOW_SIZE, val_fn)
        class_rate.append(cr)

        print("Epoch {} train cost = {}, validation cost = {}, "
              "generalization loss = {:.3f}, GQ = {:.3f}, classification rate = {:.3f} ({:.1f}sec)"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if val_cost < best_val:
            best_val = val_cost
            best_conf = val_conf
            best_cr = cr
            if best_cr > 0.55:
                print('saving a good encoder...')
                encoder = extract_encoder(network, (None, 1200), 2, 6)
                pickle.dump(encoder, open('models/end2end_encoder.dat', 'wb'))

        if epoch >= VALIDATION_WINDOW and early_stop(val_window):
            break
        # learning rate decay
        if epoch > 12:
            lr.set_value(lr.get_value() * lr_decay)

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

    print('Best Model')
    print('classification rate: {}, validation loss: {}'.format(best_cr, best_val))
    print('confusion matrix: ')
    plot_confusion_matrix(best_conf, letters, fmt='grid')
    plot_validation_cost(cost_train, cost_val, class_rate)

if __name__ == '__main__':
    main()
    # train_deltanet()
