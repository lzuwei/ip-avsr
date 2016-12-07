import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate, DropoutLayer
from lasagne.nonlinearities import tanh, sigmoid, linear
from lasagne.layers import batch_norm, BatchNormLayer

from custom.layers import DeltaLayer, AdaptiveElemwiseSumLayer
from modelzoo.pretrained_encoder import create_pretrained_encoder


def create_blstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name, use_peepholes=True):

    if cell_parameters is None:
        cell_parameters = Gate()
    if gate_parameters is None:
        gate_parameters = Gate()

    l_lstm = LSTMLayer(
        l_incoming, hidden_units, peepholes=use_peepholes,
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
        l_incoming, hidden_units, ingate=gate_parameters, peepholes=use_peepholes,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True, name='b_{}'.format(name))

    return l_lstm, l_lstm_back


def create_lstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name, use_peepholes=True):

    if cell_parameters is None:
        cell_parameters = Gate()
    if gate_parameters is None:
        gate_parameters = Gate()

    l_lstm = LSTMLayer(
        l_incoming, hidden_units, peepholes=use_peepholes,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='f_{}'.format(name))

    return l_lstm


def create_model(dbn, input_shape, input_var, mask_shape, mask_var,
                 dct_shape, dct_var, lstm_size=250, win=T.iscalar('theta)'),
                 output_classes=26, fusiontype='sum', w_init_fn=las.init.Orthogonal(),
                 use_peepholes=True):

    dbn_layers = dbn.get_all_layers()
    weights = []
    biases = []
    shapes = [2000, 1000, 500, 50]
    nonlinearities = [rectify, rectify, rectify, linear]
    weights.append(dbn_layers[1].W.astype('float32'))
    weights.append(dbn_layers[2].W.astype('float32'))
    weights.append(dbn_layers[3].W.astype('float32'))
    weights.append(dbn_layers[4].W.astype('float32'))
    biases.append(dbn_layers[1].b.astype('float32'))
    biases.append(dbn_layers[2].b.astype('float32'))
    biases.append(dbn_layers[3].b.astype('float32'))
    biases.append(dbn_layers[4].b.astype('float32'))

    gate_parameters = Gate(
        W_in=las.init.Orthogonal(), W_hid=w_init_fn,
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
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
    l_encoder = create_pretrained_encoder(l_reshape1, weights, biases, shapes, nonlinearities,
                                          ['fc1', 'fc2', 'fc3', 'bottleneck'])
    encoder_len = las.layers.get_output_shape(l_encoder)[-1]
    l_reshape2 = ReshapeLayer(l_encoder, (symbolic_batchsize, symbolic_seqlen, encoder_len), name='reshape2')
    l_delta = DeltaLayer(l_reshape2, win, name='delta')

    l_lstm_bn = LSTMLayer(
        l_delta, lstm_size, peepholes=use_peepholes,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm_bn')

    l_lstm_dct = LSTMLayer(
        l_dct, lstm_size, peepholes=use_peepholes,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm_dct')

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.

    if fusiontype == 'sum':
        l_fuse = ElemwiseSumLayer([l_lstm_bn, l_lstm_dct], name='sum1')
    elif fusiontype == 'adasum':
        l_fuse = AdaptiveElemwiseSumLayer([l_lstm_bn, l_lstm_dct], name='adasum')
    elif fusiontype == 'concat':
        l_fuse = ConcatLayer([l_lstm_bn, l_lstm_dct], axis=2, name='concat')

    f_lstm_agg = create_lstm(l_fuse, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm_agg')

    # reshape to (num_examples * seq_len, lstm_size)
    l_reshape3 = ReshapeLayer(f_lstm_agg, (-1, lstm_size))

    # l_forward_slice1 = SliceLayer(l_sum2, -1, 1, name='slice1')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape3, num_units=output_classes, nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen, output_classes), name='output')

    return l_out, l_fuse
