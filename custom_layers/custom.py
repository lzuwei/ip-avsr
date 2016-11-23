import numpy as np
from lasagne.layers import Layer, ElemwiseMergeLayer
from lasagne.init import Normal
from lasagne.utils import unroll_scan
import theano.tensor as T
import theano
import utils.signal


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
    def __init__(self, incoming, window, **kwargs):
        super(MajorityVotingLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        votes = theano.tensor.zeros(input.shape()[1:])
        for p in input:
            idx = theano.tensor.argmax(p)
            votes[idx] += 1
        return votes

    def get_output_shape_for(self, input_shape):
        return input_shape[1:]


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