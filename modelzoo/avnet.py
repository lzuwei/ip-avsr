import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate, DropoutLayer, GlobalPoolLayer
from lasagne.nonlinearities import tanh, linear, rectify

from custom.layers import DeltaLayer, AdaptiveElemwiseSumLayer, create_blstm
from modelzoo.pretrained_encoder import create_pretrained_encoder


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


def create_pretrained_substream(weights, biases, input_shape, input_var, mask_shape, mask_var, name,
                                lstm_size=250, win=T.iscalar('theta'), nonlinearity=rectify,
                                w_init_fn=las.init.Orthogonal(), use_peepholes=True):
    gate_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    l_input = InputLayer(input_shape, input_var, 'input_'+name)
    l_mask = InputLayer(mask_shape, mask_var, 'mask')

    symbolic_batchsize_raw = l_input.input_var.shape[0]
    symbolic_seqlen_raw = l_input.input_var.shape[1]

    l_reshape1_raw = ReshapeLayer(l_input, (-1, input_shape[-1]), name='reshape1_'+name)
    l_encoder_raw = create_pretrained_encoder(l_reshape1_raw, weights, biases,
                                              [2000, 1000, 500, 50],
                                              [nonlinearity, nonlinearity, nonlinearity, linear],
                                              ['fc1_'+name, 'fc2_'+name, 'fc3_'+name, 'bottleneck_'+name])
    input_len = las.layers.get_output_shape(l_encoder_raw)[-1]

    l_reshape2 = ReshapeLayer(l_encoder_raw,
                                  (symbolic_batchsize_raw, symbolic_seqlen_raw, input_len),
                                  name='reshape2_'+name)
    l_delta = DeltaLayer(l_reshape2, win, name='delta_'+name)

    l_lstm = LSTMLayer(
        l_delta, int(lstm_size), peepholes=use_peepholes,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name='lstm_'+name)

    return l_lstm


def create_model(substreams, mask_shape, mask_var, lstm_size=250, output_classes=26,
                 fusiontype='concat', w_init_fn=las.init.Orthogonal(), use_peepholes=True):

    gate_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        b=las.init.Constant(0.))
    cell_parameters = Gate(
        W_in=w_init_fn, W_hid=w_init_fn,
        # Setting W_cell to None denotes that no cell connection will be used.
        W_cell=None, b=las.init.Constant(0.),
        # By convention, the cell nonlinearity is tanh in an LSTM.
        nonlinearity=tanh)

    l_mask = InputLayer(mask_shape, mask_var, 'mask')
    symbolic_seqlen_raw = l_mask.input_var.shape[1]

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    if fusiontype == 'adasum':
        l_fuse = AdaptiveElemwiseSumLayer(substreams, name='adasum1')
    elif fusiontype == 'sum':
        l_fuse = ElemwiseSumLayer(substreams, name='sum1')
    elif fusiontype == 'concat':
        l_fuse = ConcatLayer(substreams, axis=-1, name='concat')

    f_lstm_agg, b_lstm_agg = create_blstm(l_fuse, l_mask, lstm_size, cell_parameters, gate_parameters, 'lstm_agg')
    l_sum2 = ElemwiseSumLayer([f_lstm_agg, b_lstm_agg], name='sum2')

    # reshape to (num_examples * seq_len, lstm_size)
    l_reshape3 = ReshapeLayer(l_sum2, (-1, lstm_size), name='reshape3')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape3, num_units=output_classes,
        nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen_raw, output_classes), name='output')

    return l_out, l_fuse
