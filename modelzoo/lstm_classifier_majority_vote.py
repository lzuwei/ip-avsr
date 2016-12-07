import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate, DropoutLayer
from lasagne.nonlinearities import tanh, sigmoid, linear


def create_lstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name):
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

    return l_lstm


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


def create_model(input_shape, input_var, mask_shape, mask_var, lstm_size=250, output_classes=26,
                 w_init=las.init.Orthogonal()):
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

    f_lstm, b_lstm = create_blstm(l_in, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm')

    l_sum = ElemwiseSumLayer([f_lstm, b_lstm], name='sum')

    # reshape to (num_examples * seq_len, lstm_size)
    l_reshape = ReshapeLayer(l_sum, (-1, lstm_size))

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape, num_units=output_classes, nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen, output_classes), name='output')
    return l_out
