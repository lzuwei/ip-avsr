from lasagne.layers import Layer
import theano
import utils.signal


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
