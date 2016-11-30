import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate, DropoutLayer, GlobalPoolLayer
from lasagne.nonlinearities import tanh, sigmoid, linear
from lasagne.layers import batch_norm, BatchNormLayer

from custom.layers import DeltaLayer, AdaptiveElemwiseSumLayer


def create_pretrained_encoder(weights, biases, names, incoming):
    l_1 = DenseLayer(incoming, 2000, W=weights[0], b=biases[0], nonlinearity=sigmoid, name=names[0])
    l_2 = DenseLayer(l_1, 1000, W=weights[1], b=biases[1], nonlinearity=sigmoid, name=names[1])
    l_3 = DenseLayer(l_2, 500, W=weights[2], b=biases[2], nonlinearity=sigmoid, name=names[2])
    l_4 = DenseLayer(l_3, 50, W=weights[3], b=biases[3], nonlinearity=linear, name=names[3])
    return l_4


def create_lstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name, use_peepholes=True):

    if cell_parameters is None:
        cell_parameters = Gate()
    if gate_parameters is None:
        gate_parameters = Gate()

    l_lstm = LSTMLayer(
        l_incoming, hidden_units,
        # We need to specify a separate input for masks
        mask_input=l_mask, peepholes=use_peepholes,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='f_{}'.format(name))

    return l_lstm


def extract_weights(ae):
    weights = []
    biases = []
    ae_layers = ae.get_all_layers()
    weights.append(ae_layers[1].W.astype('float32'))
    weights.append(ae_layers[2].W.astype('float32'))
    weights.append(ae_layers[3].W.astype('float32'))
    weights.append(ae_layers[4].W.astype('float32'))
    biases.append(ae_layers[1].b.astype('float32'))
    biases.append(ae_layers[2].b.astype('float32'))
    biases.append(ae_layers[3].b.astype('float32'))
    biases.append(ae_layers[4].b.astype('float32'))

    return weights, biases


def create_model(ae, diff_ae, input_shape, input_var, mask_shape, mask_var,
                 diff_shape, diff_var, lstm_size=250, win=T.iscalar('theta)'),
                 output_classes=26, fusiontype='concat', w_init_fn=las.init.Orthogonal(),
                 use_peepholes=True):

    bn_weights, bn_biases = extract_weights(ae)
    diff_weights, diff_biases = extract_weights(diff_ae)

    gate_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    l_raw = InputLayer(input_shape, input_var, 'raw_im')
    l_mask = InputLayer(mask_shape, mask_var, 'mask')
    l_diff = InputLayer(diff_shape, diff_var, 'diff_im')

    symbolic_batchsize_raw = l_raw.input_var.shape[0]
    symbolic_seqlen_raw = l_raw.input_var.shape[1]
    symbolic_batchsize_diff = l_diff.input_var.shape[0]
    symbolic_seqlen_diff = l_diff.input_var.shape[1]

    l_reshape1_raw = ReshapeLayer(l_raw, (-1, input_shape[-1]), name='reshape1_raw')
    l_encoder_raw = create_pretrained_encoder(bn_weights, bn_biases, ['fc1_raw', 'fc2_raw', 'fc3_raw', 'bottleneck_raw'],
                                              l_reshape1_raw)
    raw_len = las.layers.get_output_shape(l_encoder_raw)[-1]

    l_reshape2_raw = ReshapeLayer(l_encoder_raw,
                                  (symbolic_batchsize_raw, symbolic_seqlen_raw, raw_len),
                                  name='reshape2_raw')
    l_delta_raw = DeltaLayer(l_reshape2_raw, win, name='delta_raw')

    # diff images
    l_reshape1_diff = ReshapeLayer(l_diff, (-1, diff_shape[-1]), name='reshape1_diff')
    l_encoder_diff = create_pretrained_encoder(diff_weights, diff_biases,
                                               ['fc1_diff', 'fc2_diff', 'fc3_diff', 'bottleneck_diff'],
                                               l_reshape1_diff)
    diff_len = las.layers.get_output_shape(l_encoder_diff)[-1]
    l_reshape2_diff = ReshapeLayer(l_encoder_diff,
                                   (symbolic_batchsize_diff, symbolic_seqlen_diff, diff_len),
                                   name='reshape2_diff')
    l_delta_diff = DeltaLayer(l_reshape2_diff, win, name='delta_diff')

    l_lstm_raw = LSTMLayer(
        l_delta_raw, int(lstm_size), peepholes=use_peepholes,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm_raw')

    l_lstm_diff = LSTMLayer(
        l_delta_diff, lstm_size, peepholes=use_peepholes,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm_diff')

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    if fusiontype == 'adasum':
        l_fuse = AdaptiveElemwiseSumLayer([l_lstm_raw, l_lstm_diff], name='adasum1')
    elif fusiontype == 'sum':
        l_fuse = ElemwiseSumLayer([l_lstm_raw, l_lstm_diff], name='sum1')
    elif fusiontype == 'concat':
        l_fuse = ConcatLayer([l_lstm_raw, l_lstm_diff], axis=-1, name='concat')

    f_lstm_agg = create_lstm(l_fuse, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm_agg')

    # reshape to (num_examples * seq_len, lstm_size)
    l_reshape3 = ReshapeLayer(f_lstm_agg, (-1, lstm_size))

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape3, num_units=output_classes,
        nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen_raw, output_classes), name='output')

    return l_out, l_fuse
