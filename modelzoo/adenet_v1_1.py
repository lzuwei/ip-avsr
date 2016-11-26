import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate, DropoutLayer, BatchNormLayer
from lasagne.nonlinearities import tanh, sigmoid, linear

from custom.layers import DeltaLayer


def create_pretrained_encoder(weights, biases, incoming):
    l_1 = DenseLayer(incoming, 2000, W=weights[0], b=biases[0], nonlinearity=sigmoid, name='fc1')
    l_2 = DenseLayer(l_1, 1000, W=weights[1], b=biases[1], nonlinearity=sigmoid, name='fc2')
    l_3 = DenseLayer(l_2, 500, W=weights[2], b=biases[2], nonlinearity=sigmoid, name='fc3')
    l_4 = DenseLayer(l_3, 50, W=weights[3], b=biases[3], nonlinearity=linear, name='bottleneck')
    return l_4


def create_blstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name):

    if cell_parameters is None:
        cell_parameters = Gate()
    if gate_parameters is None:
        gate_parameters = Gate()

    l_lstm = LSTMLayer(
        l_incoming, hidden_units,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='f_{}'.format(name))

    # The "backwards" layer is the same as the first,
    # except that the backwards argument is set to True.
    l_lstm_back = LSTMLayer(
        l_incoming, hidden_units, ingate=gate_parameters,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True, name='b_{}'.format(name))

    return l_lstm, l_lstm_back


def create_model(dbn, input_shape, input_var, mask_shape, mask_var,
                 dct_shape, dct_var, lstm_size=250, win=T.iscalar('theta)'),
                 output_classes=26):

    dbn_layers = dbn.get_all_layers()
    weights = []
    biases = []
    weights.append(dbn_layers[1].W.astype('float32'))
    weights.append(dbn_layers[2].W.astype('float32'))
    weights.append(dbn_layers[3].W.astype('float32'))
    weights.append(dbn_layers[4].W.astype('float32'))
    biases.append(dbn_layers[1].b.astype('float32'))
    biases.append(dbn_layers[2].b.astype('float32'))
    biases.append(dbn_layers[3].b.astype('float32'))
    biases.append(dbn_layers[4].b.astype('float32'))

    gate_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=las.init.Orthogonal(),
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    l_in = InputLayer(input_shape, input_var, 'input')
    l_mask = InputLayer(mask_shape, mask_var, 'mask')
    l_dct = InputLayer(dct_shape, dct_var, 'dct')

    symbolic_batchsize = l_in.input_var.shape[0]
    symbolic_seqlen = l_in.input_var.shape[1]

    l_reshape1 = ReshapeLayer(l_in, (-1, input_shape[-1]), name='reshape1')
    l_encoder = create_pretrained_encoder(weights, biases, l_reshape1)
    l_encoder_bn = BatchNormLayer(l_encoder, name='batchnorm1')
    encoder_len = las.layers.get_output_shape(l_encoder)[-1]
    l_reshape2 = ReshapeLayer(l_encoder_bn, (symbolic_batchsize, symbolic_seqlen, encoder_len), name='reshape2')
    l_delta = DeltaLayer(l_reshape2, win, name='delta')

    l_concat = ConcatLayer([l_delta, l_dct], axis=2, name='concat')

    l_dropout1 = DropoutLayer(l_concat, name='dropout1')

    l_lstm, l_lstm_back = create_blstm(l_dropout1, l_mask, lstm_size * 2, cell_parameters, gate_parameters, 'lstm1')

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum1 = ElemwiseSumLayer([l_lstm, l_lstm_back], name='sum1')

    # implement drop-out regularization
    l_dropout2 = DropoutLayer(l_sum1, name='dropout2')

    l_lstm2, l_lstm2_back = create_blstm(l_dropout2, l_mask, lstm_size * 2, cell_parameters, gate_parameters, 'lstm2')

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum2 = ElemwiseSumLayer([l_lstm2, l_lstm2_back])

    l_forward_slice1 = SliceLayer(l_sum2, -1, 1, name='slice1')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_out = DenseLayer(
        l_forward_slice1, num_units=output_classes, nonlinearity=las.nonlinearities.softmax, name='output')

    return l_out
