import sys
sys.path.insert(0, '../')
import os
import time
import pickle
import logging

import theano.tensor as T
import theano

import lasagne as las
from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from custom_layers.custom import DeltaLayer


import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, LSTMLayer, Gate, ElemwiseSumLayer, SliceLayer
from lasagne.layers import ReshapeLayer, DimshuffleLayer, ConcatLayer, BatchNormLayer, batch_norm
from lasagne.nonlinearities import tanh, linear, sigmoid, rectify
from lasagne.updates import nesterov_momentum, adadelta, sgd, norm_constraint
from lasagne.objectives import squared_error
from nolearn.lasagne import NeuralNet


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def test_delta():
    a = np.array([[1,1,1,1,1,1,1,1,10], [2,2,2,2,2,2,2,2,20], [3,3,3,3,3,3,3,3,30], [4,4,4,4,4,4,4,4,40]])
    aa = deltas(a, 9)
    print(aa)


def load_dbn(path='../../DBNExample/pretrained_nn.mat'):
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
        l1_num_units=1000, l1_nonlinearity=sigmoid,
        l2_num_units=1000, l2_nonlinearity=sigmoid,
        l3_num_units=1000, l3_nonlinearity=sigmoid,
        l4_num_units=50, l4_nonlinearity=linear,
        l5_num_units=1000, l5_nonlinearity=sigmoid,
        l6_num_units=1000, l6_nonlinearity=sigmoid,
        l7_num_units=1000, l7_nonlinearity=sigmoid,
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
    l_4 = DenseLayer(l_3, 50, W=weights[3], b=biases[3], nonlinearity=linear, name='encoder')
    return l_4


def create_fc_encoder(incoming):
    l_1 = DenseLayer(incoming, 1000, nonlinearity=rectify, name='fc1')
    l_drop1 = DropoutLayer(l_1, name='drop1')
    l_2 = DenseLayer(l_drop1, 1000, nonlinearity=rectify, name='fc2')
    l_drop2 = DropoutLayer(l_2, name='drop2')
    l_3 = DenseLayer(l_drop2, 1000, nonlinearity=rectify, name='fc3')
    l_drop3 = DropoutLayer(l_3, name='drop3')
    l_4 = DenseLayer(l_drop3, 50, nonlinearity=linear, name='encoder')
    return l_4


def print_layer_shape(layer):
    print('[L] {}: {}'.format(layer.name, las.layers.get_output_shape(layer)))


def print_network(network):
    layers = las.layers.get_all_layers(network)
    for layer in layers:
        print_layer_shape(layer)


def create_end_to_end_model(dbn, input_shape, input_var, mask_shape, mask_var,
                            lstm_size=250, win=T.iscalar('theta)')):

    dbn_layers = dbn.get_all_layers()
    weights = []
    biases = []
    weights.append(dbn_layers[1].W)
    weights.append(dbn_layers[2].W)
    weights.append(dbn_layers[3].W)
    weights.append(dbn_layers[4].W)
    biases.append(dbn_layers[1].b)
    biases.append(dbn_layers[2].b)
    biases.append(dbn_layers[3].b)
    biases.append(dbn_layers[4].b)

    l_in = InputLayer(input_shape, input_var, 'input')
    l_mask = InputLayer(mask_shape, mask_var, 'mask')

    symbolic_batchsize = l_in.input_var.shape[0]
    symbolic_seqlen = l_in.input_var.shape[1]

    print_layer_shape(l_in)
    l_reshape1 = ReshapeLayer(l_in, (-1, input_shape[-1]), name='reshape1')
    print_layer_shape(l_reshape1)
    # l_encoder = create_fc_encoder(l_reshape1)
    l_encoder = create_pretrained_encoder(weights, biases, l_reshape1)
    print_layer_shape(l_encoder)
    encoder_len = las.layers.get_output_shape(l_encoder)[-1]
    l_reshape2 = ReshapeLayer(l_encoder, (symbolic_batchsize, symbolic_seqlen, encoder_len), name='reshape2')
    print_layer_shape(l_reshape2)
    l_delta = DeltaLayer(l_reshape2, win, name='delta')
    print_layer_shape(l_delta)

    gate_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    N_HIDDEN = lstm_size
    l_lstm = LSTMLayer(
        l_delta, N_HIDDEN,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='f_lstm1')
    print_layer_shape(l_lstm)

    # The "backwards" layer is the same as the first,
    # except that the backwards argument is set to True.
    l_lstm_back = LSTMLayer(
        l_in, N_HIDDEN, ingate=gate_parameters,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True, name='b_lstm1')
    print_layer_shape(l_lstm_back)

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum1 = ElemwiseSumLayer([l_lstm, l_lstm_back], name='sum1')
    print_layer_shape(l_sum1)

    # implement drop-out regularization
    l_dropout = DropoutLayer(l_sum1, name='dropout1')
    print_layer_shape(l_dropout)

    l_lstm2 = LSTMLayer(
        l_dropout, N_HIDDEN,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='f_lstm2')
    print_layer_shape(l_lstm2)

    # The "backwards" layer is the same as the first,
    # except that the backwards argument is set to True.
    l_lstm_back2 = LSTMLayer(
        l_dropout, N_HIDDEN, ingate=gate_parameters,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True, name='b_lstm2')
    print_layer_shape(l_lstm_back2)

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum2 = ElemwiseSumLayer([l_lstm2, l_lstm_back2])
    print_layer_shape(l_sum2)

    l_forward_slice1 = SliceLayer(l_sum2, -1, 1, name='slice1')
    print_layer_shape(l_forward_slice1)

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_out = DenseLayer(
        l_forward_slice1, num_units=26, nonlinearity=las.nonlinearities.softmax, name='output')
    print_layer_shape(l_out)

    return l_out


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


def construct_lstm(input_size, lstm_size, output_size, train_data_gen, val_data_gen):

    # All gates have initializers for the input-to-gate and hidden state-to-gate
    # weight matrices, the cell-to-gate weight vector, the bias vector, and the nonlinearity.
    # The convention is that gates use the standard sigmoid nonlinearity,
    # which is the default for the Gate class.
    gate_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    # prepare the input layers
    # By setting the first and second dimensions to None, we allow
    # arbitrary minibatch sizes with arbitrary sequence lengths.
    # The number of feature dimensions is 150, as described above.
    l_in = InputLayer(shape=(None, None, input_size))
    # This input will be used to provide the network with masks.
    # Masks are expected to be matrices of shape (n_batch, n_time_steps);
    # both of these dimensions are variable for us so we will use
    # an input shape of (None, None)
    l_mask = InputLayer(shape=(None, None))

    # Our LSTM will have 180 hidden/cell units as published in paper
    N_HIDDEN = lstm_size
    l_lstm = LSTMLayer(
        l_in, N_HIDDEN,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5.)

    # The "backwards" layer is the same as the first,
    # except that the backwards argument is set to True.
    l_lstm_back = LSTMLayer(
        l_in, N_HIDDEN, ingate=gate_parameters,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True)
    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum = ElemwiseSumLayer([l_lstm, l_lstm_back])

    # implement drop-out regularization
    l_dropout = DropoutLayer(l_sum)

    l_lstm2 = LSTMLayer(
        l_dropout, N_HIDDEN,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5.)

    # The "backwards" layer is the same as the first,
    # except that the backwards argument is set to True.
    l_lstm_back2 = LSTMLayer(
        l_dropout, N_HIDDEN, ingate=gate_parameters,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True)

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum2 = ElemwiseSumLayer([l_lstm2, l_lstm_back2])

    '''
    l_dropout2 = DropoutLayer(l_sum2)

    l_lstm3 = LSTMLayer(
        l_dropout2, N_HIDDEN,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5.)

    # The "backwards" layer is the same as the first,
    # except that the backwards argument is set to True.
    l_lstm_back3 = LSTMLayer(
        l_dropout2, N_HIDDEN, ingate=gate_parameters,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True)

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum3 = ElemwiseSumLayer([l_lstm3, l_lstm_back3])
    '''
    # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer.
    # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
    l_forward_slice = SliceLayer(l_sum2, -1, 1)

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_out = DenseLayer(
        l_forward_slice, num_units=output_size, nonlinearity=las.nonlinearities.softmax)

    # Now, the shape will be n_batch*n_timesteps, output_size. We can then reshape to
    # n_batch, n_timesteps to get a single value for each timstep from each sequence
    # l_out = las.layers.ReshapeLayer(l_dense, (n_batch, n_time_steps))

    # Symbolic variable for the target network output.
    # It will be of shape n_batch, because there's only 1 target value per sequence.
    target_values = T.ivector('target_output')

    # This matrix will tell the network the length of each sequences.
    # The actual values will be supplied by the gen_data function.
    mask = T.matrix('mask')

    # lasagne.layers.get_output produces an expression for the output of the net
    network_output = las.layers.get_output(l_out)

    # The value we care about is the final value produced for each sequence
    # so we simply slice it out.
    # predicted_values = network_output[:, -1]

    # Our cost will be categorical cross entropy error
    cost = T.mean(las.objectives.categorical_crossentropy(network_output, target_values))
    # cost = T.mean((predicted_values - target_values) ** 2)
    # Retrieve all parameters from the network
    all_params = las.layers.get_all_params(l_out)
    # Compute adam updates for training
    # updates = las.updates.adam(cost, all_params)
    updates = adadelta(cost, all_params)
    # Theano functions for training and computing cost
    train = theano.function(
        [l_in.input_var, target_values, l_mask.input_var],
        cost, updates=updates, allow_input_downcast=True)

    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost, allow_input_downcast=True)

    probs = theano.function([l_in.input_var, l_mask.input_var], network_output, allow_input_downcast=True)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = next(val_data_gen)

    # We'll train the network with 10 epochs of 100 minibatches each
    cost_train = []
    cost_val = []
    class_rate = []
    NUM_EPOCHS = 20
    EPOCH_SIZE = 26
    for epoch in range(NUM_EPOCHS):
        for _ in range(EPOCH_SIZE):
            X, y, m = next(train_data_gen)
            train(X, y, m)
        cost_train.append(compute_cost(X, y, m))
        cost_val.append(compute_cost(X_val, y_val, mask_val))
        cr, _ = evaluate_model(X_val, y_val, mask_val, probs)
        class_rate.append(cr)

        # one good value to early stop using GL technique, alpha = 0.10 (10% worst)
        gl = cost_val[-1] / np.min(cost_val) - 1
        # PQ, GL / Pk(t) where Pk(t) = 1000 * (sum(training strip error) / k * min(training strip error) - 1

        print("Epoch {} train cost = {}, validation cost = {}, generalization loss = {}, classification rate = {}"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, cr))
    cr, conf = evaluate_model(X_val, y_val, mask_val, probs)
    print('Final Model')
    print('classification rate: {}'.format(cr))
    print('confusion matrix: ')
    plot_confusion_matrix(conf, fmt='grid')
    plot_validation_cost(cost_train, cost_val, class_rate)


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

    finetune = False
    if finetune:
        print('fine-tuning...')
        dbn = load_dbn('models/1k_bottleneck_ae.mat')
        dbn.initialize()
        dbn.fit(train_data_resized, train_data_resized)
        # res = dbn.predict(test_data_resized)
        # print(res.shape)
        # visualize_reconstruction(test_data_resized[500:525], res[500:525])

    save = False
    if save:
        pickle.dump(dbn, open('models/1k_bot_dbn_finetune.dat', 'wb'))

    load = True
    if load:
        print('loading pre-trained encoding layers...')
        dbn = pickle.load(open('models/1k_bot_dbn_finetune.dat', 'rb'))
        dbn.initialize()
        # exit()

    window = T.iscalar('theta')
    inputs = T.tensor3('inputs', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    targets = T.ivector('targets')

    print('constructing end to end model...')
    network = create_end_to_end_model(dbn, (None, None, 1200), inputs,
                                      (None, None), mask, 250, window)
    print('compiling model...')
    predictions = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.categorical_crossentropy(predictions, targets))
    updates = las.updates.adadelta(cost, all_params)


    use_max_constraint = False
    if use_max_constraint:
        MAX_NORM = 4
        for param in las.layers.get_all_params(network, regularizable=True):
            if param.ndim > 1:  # only apply to dimensions larger than 1, exclude biases
                updates[param] = norm_constraint(param, MAX_NORM * las.utils.compute_norms(param.get_value()).mean())

    train = theano.function(
        [inputs, targets, mask, window],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function([inputs, targets, mask, window], cost, allow_input_downcast=True)

    test_predictions = las.layers.get_output(network, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_predictions, targets))
    compute_test_cost = theano.function(
        [inputs, targets, mask, window], test_cost, allow_input_downcast=True)

    val_fn = theano.function([inputs, mask, window], test_predictions, allow_input_downcast=True)

    # We'll train the network with 10 epochs of 30 minibatches each
    print('begin training...')
    cost_train = []
    cost_val = []
    class_rate = []
    NUM_EPOCHS = 30
    EPOCH_SIZE = 30
    BATCH_SIZE = 26
    WINDOW_SIZE = 9
    STRIP_SIZE = 3
    MAX_LOSS = 0.2
    VALIDATION_WINDOW = 4
    val_window = circular_list(VALIDATION_WINDOW)
    train_strip = np.zeros((STRIP_SIZE,))

    datagen = gen_lstm_batch_random(train_data_resized, train_targets, train_vidlen_vec, batchsize=BATCH_SIZE)
    val_datagen = gen_lstm_batch_random(test_data_resized, test_targets, test_vidlen_vec,
                                        batchsize=len(test_vidlen_vec))

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = next(val_datagen)

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
        for _ in range(EPOCH_SIZE):
            X, y, m = next(datagen)
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

        cr, _ = evaluate_model(X_val, y_val, mask_val, WINDOW_SIZE, val_fn)
        class_rate.append(cr)

        print("Epoch {} train cost = {}, validation cost = {}, "
              "generalization loss = {:.3f}, GQ = {:.3f}, classification rate = {:.3f} ({:.1f}sec)"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        # if epoch >= VALIDATION_WINDOW and early_stop(val_window):
        #    break

    cr, conf = evaluate_model(X_val, y_val, mask_val, WINDOW_SIZE, val_fn)

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

    print('Final Model')
    print('classification rate: {}'.format(cr))
    print('confusion matrix: ')
    plot_confusion_matrix(conf, letters, fmt='grid')
    plot_validation_cost(cost_train, cost_val, class_rate)

if __name__ == '__main__':
    main()
