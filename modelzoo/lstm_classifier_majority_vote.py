import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate
from lasagne.nonlinearities import tanh
from custom.layers import create_blstm, create_lstm


def create_model(input_shape, input_var, mask_shape, mask_var, lstm_size=250, output_classes=26,
                 w_init=las.init.GlorotUniform(), use_peepholes=False, use_blstm=True):
    gate_parameters = Gate(
        W_in=w_init, W_hid=w_init,
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=w_init, W_hid=w_init,
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    l_in = InputLayer(input_shape, input_var, 'input')
    l_mask = InputLayer(mask_shape, mask_var, 'mask')

    symbolic_seqlen = l_in.input_var.shape[1]
    if use_blstm:
        f_lstm, b_lstm = create_blstm(l_in, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm', use_peepholes)
        l_sum = ElemwiseSumLayer([f_lstm, b_lstm], name='sum')

        # reshape to (num_examples * seq_len, lstm_size)
        l_reshape = ReshapeLayer(l_sum, (-1, lstm_size), name='reshape')
    else:
        l_lstm = create_lstm(l_in, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm', use_peepholes)
        l_reshape = ReshapeLayer(l_lstm, (-1, lstm_size), name='reshape')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape, num_units=output_classes, nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen, output_classes), name='output')
    return l_out
