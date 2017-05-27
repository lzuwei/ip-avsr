from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import utils


def generate_lr_map(params, lr_config, default):
    """
    generate a layerwise learning map.
    to change the values of the learning rate at different epochs eg: learning rate decay
    use a tensor.shared object. To set the value of the variable use
    tensor.shared.set_value() to set the value of the variable
    tensor.shared.get_value() to get the value of the variable
    Ensure the variable type for the variable learning rates are the same type as the model weights.
    Typically you can call lasagne.utils.floatX(0.001) to ensure this.
    
    :param params: model parameters
    :param lr_config: learning rate configuration map
    :param default: default value of learning rate if key not found for layer
    :return: learning rate map
    """
    lr_map = {}
    for param in params:
        layer_name = param.name[:param.name.rfind('.')]
        if layer_name in lr_config:
            lr_map[param] = lr_config[layer_name]
        else:
            lr_map[param] = default
    return lr_map


def adam_vlr(loss_or_grads, params, lr_map, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """Adam updates with Variable Learning Rates

    Adam updates implemented as in [1]_.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    lr_map : dictionary of floats
        Learning rate map containing layer name and associated learning rate
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(utils.floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1

    for param, g_t in zip(params, all_grads):
        a_t = lr_map[param]*T.sqrt(one-beta2**t)/(one-beta1**t)
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates
