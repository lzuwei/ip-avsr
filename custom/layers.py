import numpy as np
from lasagne.layers import Layer, ElemwiseMergeLayer, LSTMLayer, Gate
from lasagne.init import Normal
from lasagne.utils import unroll_scan
import theano.tensor as T
import theano
import utils.signal


def create_lstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name, use_peepholes=False):
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
        learn_init=True, grad_clipping=5., name=name)
    return l_lstm


def create_pretrained_lstm(lstm_weights, prefix, l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters,
                           name, use_peepholes=False, backwards=False):
    l_lstm = LSTMLayer(
        l_incoming, hidden_units, peepholes=use_peepholes,
        # We need to specify a separate input for masks
        mask_input=l_mask,
        # Here, we supply the gate parameters for each gate
        ingate=gate_parameters, forgetgate=gate_parameters,
        cell=cell_parameters, outgate=gate_parameters,
        # We'll learn the initialization and use gradient clipping
        learn_init=True, grad_clipping=5., name=name, backwards=backwards)

    l_lstm.W_hid_to_cell.container.data = lstm_weights['{}_w_hid_to_cell'.format(prefix)].astype('float32')
    l_lstm.W_hid_to_forgetgate.container.data = lstm_weights['{}_w_hid_to_forgetgate'.format(prefix)].astype('float32')
    l_lstm.W_hid_to_ingate.container.data = lstm_weights['{}_w_hid_to_ingate'.format(prefix)].astype('float32')
    l_lstm.W_hid_to_outgate.container.data = lstm_weights['{}_w_hid_to_outgate'.format(prefix)].astype('float32')
    l_lstm.W_in_to_cell.container.data = lstm_weights['{}_w_in_to_cell'.format(prefix)].astype('float32')
    l_lstm.W_in_to_forgetgate.container.data = lstm_weights['{}_w_in_to_forgetgate'.format(prefix)].astype('float32')
    l_lstm.W_in_to_ingate.container.data = lstm_weights['{}_w_in_to_ingate'.format(prefix)].astype('float32')
    l_lstm.W_in_to_outgate.container.data = lstm_weights['{}_w_in_to_outgate'.format(prefix)].astype('float32')
    l_lstm.b_cell.container.data = lstm_weights['{}_b_cell'.format(prefix)].astype('float32').reshape((-1,))
    l_lstm.b_forgetgate.container.data = lstm_weights['{}_b_forgetgate'.format(prefix)].astype('float32').reshape((-1,))
    l_lstm.b_ingate.container.data = lstm_weights['{}_b_ingate'.format(prefix)].astype('float32').reshape((-1,))
    l_lstm.b_outgate.container.data = lstm_weights['{}_b_outgate'.format(prefix)].astype('float32').reshape((-1,))
    return l_lstm


def create_blstm(l_incoming, l_mask, hidden_units, cell_parameters, gate_parameters, name, use_peepholes=False):

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


class ZNormalizeLayer(Layer):
    """
       Layer to z-normalize to input sequence,
    """

    def __init__(self, incoming, **kwargs):
        super(ZNormalizeLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        # compute featurewise mean and std for the minibatch
        orig_shape = input.shape
        temp = T.reshape(input, (-1, orig_shape[-1]))
        means = T.mean(input, 0, dtype=input.dtype)
        stds = T.std(input, 0)
        temp = (temp - means) / stds
        input = T.reshape(temp, orig_shape)
        return input

    def get_output_shape_for(self, input_shape):
        return input_shape


class DeltaLayer(Layer):
    """
    Layer to add delta coefficients to input sequence,
    Appends 1st and 2nd order delta and acceleration coefficients to input sequence
    """
    def __init__(self, incoming, window, **kwargs):
        super(DeltaLayer, self).__init__(incoming, **kwargs)
        self.window = window

    def get_output_for(self, input, **kwargs):

        # compute delta coefficients for multiple sequences
        res, _ = theano.scan(utils.signal.append_delta_coeff, sequences=input, non_sequences=self.window)
        return res

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[-1] * 3


class MajorityVotingLayer(Layer):
    """
    Layer to compute the majority votes across multiple outputs
    Computes the argmax for each output prediction
    Casts a vote for each prediction and returns the combined votes as
    a single consolidated output
    """
    def __init__(self, incoming, num_classes, **kwargs):
        """
        Constructs a Majority voting layer
        :param incoming: incoming layer
        :param num_classes: number of classification classes
        :param kwargs: arguments to pass down
        """
        super(MajorityVotingLayer, self).__init__(incoming, **kwargs)
        self.num_classes = num_classes

    def get_output_for(self, input, **kwargs):
        s = input.shape
        votes = T.zeros((s[0], s[-1]), dtype=input.dtype)
        a = T.argmax(input, axis=-1)
        # for each class, count number of occurrences across all time steps
        for idx in range(self.num_classes):
            count = T.sum(T.eq(a, idx), axis=-1)
            votes = T.set_subtensor(votes[:, idx], count)
        return T.nnet.softmax(votes)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]


class MeanPoolLayer(Layer):
    """
    Layer to compute the Mean across the last axis
    """
    def __init__(self, incoming, mask, **kwargs):
        """
        Constructs a Mean Pooling layer
        :param incoming: incoming layer
        :param kwargs: arguments to pass down
        """
        super(MeanPoolLayer, self).__init__(incoming, **kwargs)
        self.mask = mask

    def get_output_for(self, input, **kwargs):
        mask = self.mask.input_var
        input = (input * mask).sum(axis=-1)
        input = input / mask.sum(axis=-1)
        return input

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]


class AdaptiveElemwiseSumLayer(ElemwiseMergeLayer):
    """
    This layer performs an elementwise sum of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    coeffs: list or scalar
        A same-sized list of coefficients, or a single coefficient that
        is to be applied to all instances. By default, these will not
        be included in the learnable parameters of this layer.

    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`

    Notes
    -----
    Depending on your architecture, this can be used to avoid the more
    costly :class:`ConcatLayer`. For example, instead of concatenating layers
    before a :class:`DenseLayer`, insert separate :class:`DenseLayer` instances
    of the same number of output units and add them up afterwards. (This avoids
    the copy operations in concatenation, but splits up the dot product.)
    """
    def __init__(self, incomings, coeffs=Normal(std=0.01, mean=1.0), cropping=None, **kwargs):
        super(AdaptiveElemwiseSumLayer, self).__init__(incomings, T.add,
                                                       cropping=cropping, **kwargs)
        '''
        if isinstance(coeffs, list):
            if len(coeffs) != len(incomings):
                raise ValueError("Mismatch: got %d coeffs for %d incomings" %
                                 (len(coeffs), len(incomings)))
        else:
            coeffs = [coeffs] * len(incomings)
        '''
        self.coeffs = []
        for i in range(len(incomings)):
            coeff = theano.shared(np.float32(1.0), 'adacoeff{}'.format(i))
            self.coeffs.append(self.add_param(coeff, coeff.shape, trainable=True, scaling_param=True))

    def get_output_for(self, inputs, **kwargs):
        # if needed multiply each input by its coefficient
        inputs = [input * coeff if coeff != 1 else input
                  for coeff, input in zip(self.coeffs, inputs)]

        # pass scaled inputs to the super class for summing
        return super(AdaptiveElemwiseSumLayer, self).get_output_for(inputs, **kwargs)


def test_vote():
    a = [[[1,2,3],[1,2,3],[1,2,3]],
         [[1,3,1],[1,3,1],[1,3,1]],
         [[5,0,0],[0,5,0],[0,0,5]],
         [[1,0,0],[0,1,0],[1,0,0]]]
    a = np.array(a)
    s = a.shape
    votes = np.zeros((s[0], s[-1]))
    a = np.argmax(a, axis=-1)
    for i in range(s[0]):
        for idx in range(s[-1]):
            t = a[i] == idx
            count = np.sum(t)
            votes[i][idx] = count

    return votes
