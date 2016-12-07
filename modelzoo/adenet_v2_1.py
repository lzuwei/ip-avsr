import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate, DropoutLayer
from lasagne.nonlinearities import tanh, sigmoid, linear, rectify

from custom.layers import DeltaLayer, AdaptiveElemwiseSumLayer
from modelzoo.pretrained_encoder import create_pretrained_encoder


def create_blstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name, use_peepholes=True):

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

    # The "backwards" layer is the same as the first,
    # except that the backwards argument is set to True.
    l_lstm_back = LSTMLayer(
        l_incoming, hidden_units, ingate=gate_parameters, peepholes=use_peepholes,
        mask_input=l_mask, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        learn_init=True, grad_clipping=5., backwards=True, name='b_{}'.format(name))

    return l_lstm, l_lstm_back


def extract_weights(ae):
    weights = []
    biases = []
    shapes = [2000, 1000, 500, 50]
    nonlinearities = [rectify, rectify, rectify, linear]
    ae_layers = ae.get_all_layers()
    weights.append(ae_layers[1].W.astype('float32'))
    weights.append(ae_layers[2].W.astype('float32'))
    weights.append(ae_layers[3].W.astype('float32'))
    weights.append(ae_layers[4].W.astype('float32'))
    biases.append(ae_layers[1].b.astype('float32'))
    biases.append(ae_layers[2].b.astype('float32'))
    biases.append(ae_layers[3].b.astype('float32'))
    biases.append(ae_layers[4].b.astype('float32'))

    return weights, biases, shapes, nonlinearities


def create_model(ae, diff_ae, input_shape, input_var, mask_shape, mask_var,
                 diff_shape, diff_var, lstm_size=250, win=T.iscalar('theta)'),
                 output_classes=26, fusiontype='concat', w_init_fn=las.init.Orthogonal(),
                 use_peepholes=True):

    bn_weights, bn_biases, bn_shapes, bn_nonlinearities = extract_weights(ae)
    diff_weights, diff_biases, diff_shapes, diff_nonlinearities = extract_weights(diff_ae)

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
    l_encoder_raw = create_pretrained_encoder(l_reshape1_raw, bn_weights, bn_biases, bn_shapes, bn_nonlinearities,
                                              ['fc1_raw', 'fc2_raw', 'fc3_raw', 'bottleneck_raw'])
    raw_len = las.layers.get_output_shape(l_encoder_raw)[-1]

    l_reshape2_raw = ReshapeLayer(l_encoder_raw,
                                  (symbolic_batchsize_raw, symbolic_seqlen_raw, raw_len),
                                  name='reshape2_raw')
    l_delta_raw = DeltaLayer(l_reshape2_raw, win, name='delta_raw')

    # diff images
    l_reshape1_diff = ReshapeLayer(l_diff, (-1, diff_shape[-1]), name='reshape1_diff')
    l_encoder_diff = create_pretrained_encoder(l_reshape1_diff, diff_weights, diff_biases, diff_shapes,
                                               diff_nonlinearities,
                                               ['fc1_diff', 'fc2_diff', 'fc3_diff', 'bottleneck_diff'])
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

    # l_drop_agg = DropoutLayer(l_sum1, name='dropout_agg')

    f_lstm_agg, b_lstm_agg = create_blstm(l_fuse, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm_agg')
    l_sum2 = ElemwiseSumLayer([f_lstm_agg, b_lstm_agg], name='sum2')


    '''
    l_lstm_agg = LSTMLayer(
        l_drop_agg, lstm_size * 2,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm_agg')

    # implement drop-out regularization
    l_dropout = DropoutLayer(l_sum1, p=0.4, name='dropout1')

    l_lstm2, l_lstm2_back = create_blstm(l_dropout, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm2')

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum2 = ElemwiseSumLayer([l_lstm2, l_lstm2_back])
    '''

    # l_sum2 = ElemwiseSumLayer([f_lstm_agg, b_lstm_agg], name='sum2')

    l_forward_slice1 = SliceLayer(l_sum2, -1, 1, name='slice1')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_out = DenseLayer(
        l_forward_slice1, num_units=output_classes,
        nonlinearity=las.nonlinearities.softmax, name='output')

    return l_out, l_fuse
