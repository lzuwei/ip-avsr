import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate
from lasagne.nonlinearities import tanh, linear, rectify

from custom.layers import DeltaLayer, create_blstm
from modelzoo.pretrained_encoder import create_pretrained_encoder


def create_model_using_pretrained_encoder(weights, biases, input_shape, input_var, mask_shape, mask_var,
                                          lstm_size=250, win=T.iscalar('theta'), output_classes=26,
                                          w_init_fn=las.init.Orthogonal(),
                                          use_peepholes=False, nonlinearities=rectify):
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
                                          [2000, 1000, 500, 50],
                                          [nonlinearities, nonlinearities, nonlinearities, linear],
                                          ['fc1', 'fc2', 'fc3', 'bottleneck'])
    encoder_len = las.layers.get_output_shape(l_encoder)[-1]
    l_reshape2 = ReshapeLayer(l_encoder, (symbolic_batchsize, symbolic_seqlen, encoder_len), name='reshape2')
    l_delta = DeltaLayer(l_reshape2, win, name='delta')

    l_lstm, l_lstm_back = create_blstm(l_delta, l_mask, lstm_size, cell_parameters, gate_parameters, 'bstm1',
                                       use_peepholes)

    # We'll combine the forward and backward layer output by summing.
    # Merge layers take in lists of layers to merge as input.
    l_sum1 = ElemwiseSumLayer([l_lstm, l_lstm_back], name='sum1')

    # reshape, flatten to 2 dimensions to run softmax on all timesteps
    l_reshape3 = ReshapeLayer(l_sum1, (-1, lstm_size), name='reshape3')

    # Now, we can apply feed-forward layers as usual.
    # We want the network to predict a classification for the sequence,
    # so we'll use a the number of classes.
    l_softmax = DenseLayer(
        l_reshape3, num_units=output_classes, nonlinearity=las.nonlinearities.softmax, name='softmax')

    l_out = ReshapeLayer(l_softmax, (-1, symbolic_seqlen, output_classes), name='output')

    return l_out


def create_model(dbn, input_shape, input_var, mask_shape, mask_var,
                 lstm_size=250, win=T.iscalar('theta)'),
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

    return create_model_using_pretrained_encoder(weights, biases, input_shape, input_var, mask_shape, mask_var,
                                                 lstm_size, win, output_classes)

