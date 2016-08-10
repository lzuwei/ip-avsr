import sys
sys.path.insert(0, '../')
import os
import time
import pickle

import matplotlib
matplotlib.use('Agg')  # Change matplotlib backend, in case we have no X server running..

import theano.tensor as T
import theano

import lasagne as las
from utils.preprocessing import *
from utils.plotting_utils import *
from utils.data_structures import circular_list
from utils.datagen import *
from utils.io import *
from utils.draw_net import draw_to_file

import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, LSTMLayer, Gate, ElemwiseSumLayer, SliceLayer
from lasagne.nonlinearities import tanh, linear, sigmoid
from lasagne.updates import nesterov_momentum, adadelta, sgd
from lasagne.objectives import squared_error
from nolearn.lasagne import NeuralNet


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def test_delta():
    a = np.array([[1,1,1,1,1,1,1,1,10], [2,2,2,2,2,2,2,2,20], [3,3,3,3,3,3,3,3,30], [4,4,4,4,4,4,4,4,40]])
    aa = deltas(a, 9)
    print(aa)


def load_dbn(path='../../DBNExample/avletters_ae.mat'):
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
        (DenseLayer, {'name': 'l1', 'num_units': 2000, 'nonlinearity': sigmoid, 'W': w1, 'b': b1}),
        (DenseLayer, {'name': 'l2', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w2, 'b': b2}),
        (DenseLayer, {'name': 'l3', 'num_units': 500, 'nonlinearity': sigmoid, 'W': w3, 'b': b3}),
        (DenseLayer, {'name': 'l4', 'num_units': 50, 'nonlinearity': linear, 'W': w4, 'b': b4}),
        (DenseLayer, {'name': 'l5', 'num_units': 500, 'nonlinearity': sigmoid, 'W': w5, 'b': b5}),
        (DenseLayer, {'name': 'l6', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w6, 'b': b6}),
        (DenseLayer, {'name': 'l7', 'num_units': 2000, 'nonlinearity': sigmoid, 'W': w7, 'b': b7}),
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


def test_plot_validation_cost():
    train_error = [10, 9, 8, 7, 6, 5, 4, 3]
    val_error = [15, 14, 13, 12, 11, 10, 9, 8]
    class_rate = [80, 81, 82, 83, 84, 85, 86, 87]
    plot_validation_cost(train_error, val_error, class_rate)


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


# We flatten the sequence into the batch dimension before calculating the loss
def calc_cross_ent(net_output, targets):
    preds = T.reshape(net_output, (-1, 26))
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)
    return cost


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
    l_in = InputLayer(shape=(None, None, input_size), name='input')
    # This input will be used to provide the network with masks.
    # Masks are expected to be matrices of shape (n_batch, n_time_steps);
    # both of these dimensions are variable for us so we will use
    # an input shape of (None, None)
    l_mask = InputLayer(shape=(None, None), name='mask')

    # Our LSTM will have 250 hidden/cell units
    N_HIDDEN = lstm_size
    l_lstm = LSTMLayer(
        l_in, N_HIDDEN,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm1')

    '''
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
    # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer.
    # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
    l_forward_slice = SliceLayer(l_lstm, -1, 1, name='slice')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_out = DenseLayer(
        l_forward_slice, num_units=output_size, nonlinearity=las.nonlinearities.softmax, name='output')

    print_network(l_out)
    # draw_to_file(las.layers.get_all_layers(l_out), 'network.png')

    # Symbolic variable for the target network output.
    # It will be of shape n_batch, because there's only 1 target value per sequence.
    target_values = T.ivector('target_output')

    # This matrix will tell the network the length of each sequences.
    # The actual values will be supplied by the gen_data function.
    mask = T.matrix('mask')

    # lasagne.layers.get_output produces an expression for the output of the net
    prediction = las.layers.get_output(l_out)

    # The value we care about is the final value produced for each sequence
    # so we simply slice it out.
    # predicted_values = network_output[:, -1]

    # Our cost will be categorical cross entropy error
    cost = T.mean(las.objectives.categorical_crossentropy(prediction, target_values))
    # cost = T.mean((predicted_values - target_values) ** 2)
    # Retrieve all parameters from the network
    all_params = las.layers.get_all_params(l_out, trainable=True)
    # Compute adam updates for training
    # updates = las.updates.adam(cost, all_params)
    updates = adadelta(cost, all_params)
    # Theano functions for training and computing cost
    train = theano.function(
        [l_in.input_var, target_values, l_mask.input_var],
        cost, updates=updates, allow_input_downcast=True)
    compute_train_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost, allow_input_downcast=True)

    test_prediction = las.layers.get_output(l_out, deterministic=True)
    test_cost = T.mean(las.objectives.categorical_crossentropy(test_prediction, target_values))
    compute_val_cost = theano.function([l_in.input_var, target_values, l_mask.input_var],
                                       test_cost, allow_input_downcast=True)
    val_fn = theano.function([l_in.input_var, l_mask.input_var], test_prediction, allow_input_downcast=True)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = next(val_data_gen)

    # We'll train the network with 10 epochs of 100 minibatches each
    cost_train = []
    cost_val = []
    class_rate = []
    best_val = float('inf')
    best_conf = None
    best_cr = 0.0
    NUM_EPOCHS = 30
    EPOCH_SIZE = 26
    STRIP_SIZE = 3
    MAX_LOSS = 0.05
    VALIDATION_WINDOW = 4
    val_window = circular_list(VALIDATION_WINDOW)
    train_strip = np.zeros((STRIP_SIZE,))

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
            X, y, m, _ = next(train_data_gen)
            train(X, y, m)
        train_cost = compute_train_cost(X, y, m)
        val_cost = compute_val_cost(X_val, y_val, mask_val)
        cr, conf = evaluate_model(X_val, y_val, mask_val, val_fn)
        cost_train.append(train_cost)
        cost_val.append(val_cost)
        class_rate.append(cr)
        train_strip[epoch % STRIP_SIZE] = train_cost
        val_window.push(val_cost)

        gl = 100 * (cost_val[-1] / np.min(cost_val) - 1)
        pk = 1000 * (np.sum(train_strip) / (STRIP_SIZE * np.min(train_strip)) - 1)
        pq = gl / pk

        print("Epoch {} train cost = {}, validation cost = {}, "
              "generalization loss = {:.3f}, GQ = {:.3f}, classification rate = {:.3f} ({:.1f}sec)"
              .format(epoch + 1, cost_train[-1], cost_val[-1], gl, pq, cr, time.time() - time_start))

        if val_cost < best_val:
            best_val = val_cost
            best_cr = cr
            best_conf = conf

        if epoch >= VALIDATION_WINDOW and early_stop(val_window):
            break

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

    print('Final Model')
    print('classification rate: {}'.format(best_cr))
    print('validation loss: {}'.format(best_val))
    print('confusion matrix: ')
    plot_confusion_matrix(best_conf, letters, fmt='grid')
    plot_validation_cost(cost_train, cost_val, class_rate)


def main():
    configure_theano()
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
    idx = [i for i, elem in enumerate(test_targets) if elem == 20]

    print(train_data.shape)
    print(test_data.shape)
    print(sum([train_data.shape[0], test_data.shape[0]]))

    # resize the input data to 40 x 30
    train_data_resized = resize_images(train_data).astype(np.float32)

    # normalize the inputs [0 - 1]
    train_data_resized = normalize_input(train_data_resized, centralize=True)

    test_data_resized = resize_images(test_data).astype(np.float32)
    test_data_resized = normalize_input(test_data_resized, centralize=True)

    finetune = False
    if finetune:
        dbn = load_dbn()
        dbn.initialize()
        dbn.fit(train_data_resized, train_data_resized)

    save = False
    if save:
        pickle.dump(dbn, open('models/avletters_ae_finetune.dat', 'wb'))

    load = True
    if load:
        dbn = pickle.load(open('models/avletters_ae_finetune.dat', 'rb'))
        dbn.initialize()

    encoder = extract_encoder(dbn)
    X_encode = encoder.predict(train_data_resized)
    print('encoded shape: {}'.format(X_encode.shape))

    # group the data into sequences to find deltas
    # X_encode = concat_first_second_deltas(X_encode, train_vidlen_vec)
    # assert X_encode.shape == (train_data_resized.shape[0], 150)

    # training vectors, z-normalise with mean 0, std 1 sample sequence-wise
    X_encode, train_feature_mean, train_feature_std = featurewise_normalize_sequence(X_encode)

    # encode and normalize test data using training feature mean, std
    X_encode_test = encoder.predict(test_data_resized)
    # X_encode_test = concat_first_second_deltas(X_encode_test, test_vidlen_vec)
    X_encode_test = (X_encode_test - train_feature_mean) / train_feature_std
    # assert X_encode_test.shape == (test_data_resized.shape[0], 150)
    input_size = X_encode_test.shape[1]
    print('bottleneck features encoded with size: {}, train lstm...'.format(input_size))

    train_lstm_gen = gen_lstm_batch_random(X_encode, train_targets, train_vidlen_vec)
    val_lstm_gen = gen_lstm_batch_seq(X_encode_test, test_targets, test_vidlen_vec, batchsize=len(test_vidlen_vec))
    construct_lstm(input_size, 250, 26, train_lstm_gen, val_lstm_gen)

    # X_pred = dbn.predict(test_data_resized)
    # visualize_reconstruction(test_data_resized[4625:4650, :], X_pred[4625:4650, :])
    # visualize_layer(dbn.get_all_layers()[1], 40, 50)


if __name__ == '__main__':
    main()
