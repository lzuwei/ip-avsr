from lasagne.nonlinearities import *


def select_nonlinearity(string):
    nonlinearities = {'rectify': rectify,
                      'sigmoid': sigmoid,
                      'leaky_rectify': leaky_rectify,
                      'very_leaky_rectify': very_leaky_rectify,
                      'tanh': tanh,
                      'linear': linear,
                      'softmax': softmax,
                      'softplus': softplus,
                      'elu': elu,
                      'scaled_tanh': ScaledTanh,
                      'identity': identity}
    return nonlinearities[string]
