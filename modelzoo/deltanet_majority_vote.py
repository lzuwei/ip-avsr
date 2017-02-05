import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate
from lasagne.nonlinearities import tanh
from lasagne.init import GlorotUniform

from custom.layers import DeltaLayer, create_blstm, create_lstm
from modelzoo.pretrained_encoder import create_pretrained_encoder, create_encoder
from utils.io import load_model_params


def create_model(dbn, input_shape, input_var, mask_shape, mask_var,
                 lstm_size=250, win=T.iscalar('theta)'),
                 output_classes=26, w_init_fn=GlorotUniform, use_peepholes=False, use_blstm=True):

    weights, biases, shapes, nonlinearities = dbn

    gate_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    l_in = InputLayer(input_shape, input_var, 'input')
    l_mask = InputLayer(mask_shape, mask_var, 'mask')

    symbolic_batchsize = l_in.input_var.shape[0]
    symbolic_seqlen = l_in.input_var.shape[1]

    l_reshape1 = ReshapeLayer(l_in, (-1, input_shape[-1]), name='reshape1')
    l_encoder = create_pretrained_encoder(l_reshape1, weights, biases,
                                          shapes,
                                          nonlinearities,
                                          ['fc1', 'fc2', 'fc3', 'bottleneck'])
    encoder_len = las.layers.get_output_shape(l_encoder)[-1]
    l_reshape2 = ReshapeLayer(l_encoder, (symbolic_batchsize, symbolic_seqlen, encoder_len), name='reshape2')
    l_delta = DeltaLayer(l_reshape2, win, name='delta')

    if use_blstm:
        l_lstm, l_lstm_back = create_blstm(l_delta, l_mask, lstm_size, cell_parameters, gate_parameters, 'blstm1',
                                           use_peepholes)

        # We'll combine the forward and backward layer output by summing.
        # Merge layers take in lists of layers to merge as input.
        l_sum1 = ElemwiseSumLayer([l_lstm, l_lstm_back], name='sum1')
        # reshape, flatten to 2 dimensions to run softmax on all timesteps
        l_reshape3 = ReshapeLayer(l_sum1, (-1, lstm_size), name='reshape3')
    else:
        l_lstm = create_lstm(l_delta, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm', use_peepholes)
        l_reshape3 = ReshapeLayer(l_lstm, (-1, lstm_size), name='reshape3')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape3, num_units=output_classes, nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen, output_classes), name='output')

    return l_out


def load_saved_model(model_path, stream_params, input_shape, input_var, mask_shape, mask_var,
                     lstm_size=250, win=T.iscalar('theta)'),
                     output_classes=26, w_init_fn=GlorotUniform(), use_peepholes=False, use_blstm=True):
    """
    loads a saved model
    :param model_path: path to model parameters
    :param stream_params: stream parameters in a tuple of
    ([layer 1 dimension, ..., layer N dimension], [layer 1 nonlinearity, ..., layer N nonlinearity]
    :param input_shape: input shape eg: (None, None, 1500)
    :param input_var: input theano variable
    :param mask_shape: mask shape eg: (None, None) if variable lengths
    :param mask_var: mask theano variable
    :param lstm_size: number of lstm units for lstm layer
    :param win: window theano variable
    :param output_classes: number of output classes
    :param w_init_fn: weight initialization function used for initializing model
    :param use_peepholes: use peepholes for lstm layers
    :return: saved model
    """

    shapes, nonlinearities = stream_params

    gate_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    l_in = InputLayer(input_shape, input_var, 'input')
    l_mask = InputLayer(mask_shape, mask_var, 'mask')

    symbolic_batchsize = l_in.input_var.shape[0]
    symbolic_seqlen = l_in.input_var.shape[1]

    l_reshape1 = ReshapeLayer(l_in, (-1, input_shape[-1]), name='reshape1')
    l_encoder = create_encoder(l_reshape1, shapes, nonlinearities, ['fc1', 'fc2', 'fc3', 'bottleneck'])
    encoder_len = las.layers.get_output_shape(l_encoder)[-1]
    l_reshape2 = ReshapeLayer(l_encoder, (symbolic_batchsize, symbolic_seqlen, encoder_len), name='reshape2')
    l_delta = DeltaLayer(l_reshape2, win, name='delta')

    if use_blstm:
        l_lstm, l_lstm_back = create_blstm(l_delta, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm',
                                           use_peepholes)

        # We'll combine the forward and backward layer output by summing.
        # Merge layers take in lists of layers to merge as input.
        l_sum1 = ElemwiseSumLayer([l_lstm, l_lstm_back], name='sum1')
        # reshape, flatten to 2 dimensions to run softmax on all timesteps
        l_reshape3 = ReshapeLayer(l_sum1, (-1, lstm_size), name='reshape3')
    else:
        l_lstm = create_lstm(l_delta, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm', use_peepholes)
        l_reshape3 = ReshapeLayer(l_lstm, (-1, lstm_size), name='reshape3')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape3, num_units=output_classes, nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen, output_classes), name='output')
    load_model_params(l_out, model_path)
    return l_out


def extract_encoder_weights(network, names, saveas):
    """
    extract encoder weights of the given model
    :param network: trained model
    :param names: names of layer weights to extract
    :param saveas: names to save to in a list of tuples [(weight name, bias name), ...]
    :return: dictionary containing weights and biases of the encoding layers
    """
    layers = las.layers.get_all_layers(network)
    d = {}
    for i, name in enumerate(names):
        for l in layers:
            if l.name == name:
                weight = l.W.container.data
                bias = l.b.container.data
                d[saveas[i][0]] = weight
                d[saveas[i][1]] = bias
                break
    return d


def extract_lstm_weights(network, names, saveas):
    """
    extract lstm weights of a given model
    :param network: trained model
    :param names: names of lstm layer weights to extract
    :param saveas: names to save to in a list with prefix [prefix1, prefix2]
    :return: dictionary containing weights and biases of the lstm layers
    """
    layers = las.layers.get_all_layers(network)
    d = {}
    for i, name in enumerate(names):
        for l in layers:
            if l.name == name:
                w_hid_to_cell = l.W_hid_to_cell.container.data
                w_hid_to_forgetgate = l.W_hid_to_forgetgate.container.data
                w_hid_to_ingate = l.W_hid_to_ingate.container.data
                w_hid_to_outgate = l.W_hid_to_outgate.container.data
                w_in_to_cell = l.W_in_to_cell.container.data
                w_in_to_forgetgate = l.W_in_to_forgetgate.container.data
                w_in_to_ingate = l.W_in_to_ingate.container.data
                w_in_to_outgate = l.W_in_to_outgate.container.data
                b_cell = l.b_cell.container.data
                b_forgetgate = l.b_forgetgate.container.data
                b_ingate = l.b_ingate.container.data
                b_outgate = l.b_outgate.container.data
                d['{}_w_hid_to_cell'.format(saveas[i])] = w_hid_to_cell
                d['{}_w_hid_to_forgetgate'.format(saveas[i])] = w_hid_to_forgetgate
                d['{}_w_hid_to_ingate'.format(saveas[i])] = w_hid_to_ingate
                d['{}_w_hid_to_outgate'.format(saveas[i])] = w_hid_to_outgate
                d['{}_w_in_to_cell'.format(saveas[i])] = w_in_to_cell
                d['{}_w_in_to_forgetgate'.format(saveas[i])] = w_in_to_forgetgate
                d['{}_w_in_to_ingate'.format(saveas[i])] = w_in_to_ingate
                d['{}_w_in_to_outgate'.format(saveas[i])] = w_in_to_outgate
                d['{}_b_cell'.format(saveas[i])] = b_cell
                d['{}_b_forgetgate'.format(saveas[i])] = b_forgetgate
                d['{}_b_ingate'.format(saveas[i])] = b_ingate
                d['{}_b_outgate'.format(saveas[i])] = b_outgate
                break
    return d
