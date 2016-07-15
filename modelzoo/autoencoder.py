import theano.tensor as T

import lasagne as las
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, ElemwiseSumLayer
from lasagne.layers import Gate, DropoutLayer
from lasagne.nonlinearities import tanh, sigmoid, linear

import scipy.io as sio


def load_dbn(path='models/avletters_ae.mat'):
    """
    load a pretrained dbn from path
    :param path: path to the .mat dbn
    :return: pretrained deep belief network
    """
    # create the network using weights from pretrain_nn.mat
    nn = sio.loadmat(path)
    w = []
    b = []
    w.append(nn['w1'])
    w.append(nn['w2'])
    w.append(nn['w3'])
    w.append(nn['w4'])
    w.append(nn['w5'])
    w.append(nn['w6'])
    w.append(nn['w7'])
    w.append(nn['w8'])
    b.append(nn['b1'][0])
    b.append(nn['b2'][0])
    b.append(nn['b3'][0])
    b.append(nn['b4'][0])
    b.append(nn['b5'][0])
    b.append(nn['b6'][0])
    b.append(nn['b7'][0])
    b.append(nn['b8'][0])
    return w, b


def create_model(incoming, weights, biases, activations, layersizes):
    """
    Create an autoencoder given pretrained weights and activations
    :param: incoming: incoming layer (input layer)
    :param weights: layer weights
    :param biases: layer biases
    :param activations: activation functions for each layer
    :param layersizes: num hidden units for each layer
    :return: autoencoder model
    """
    for i, w in enumerate(weights):
        incoming = DenseLayer(incoming, layersizes[i], w, biases[i], activations[i], name='fc{}'.format(i + 1))
    return incoming


def create_pretrained_encoder(incoming, weights, biases, activations, layersizes):
    l_1 = DenseLayer(incoming, layersizes[0], W=weights[0], b=biases[0], nonlinearity=activations[0], name='fc1')
    l_2 = DenseLayer(l_1, layersizes[1], W=weights[1], b=biases[1], nonlinearity=activations[1], name='fc2')
    l_3 = DenseLayer(l_2, layersizes[2], W=weights[2], b=biases[2], nonlinearity=activations[2], name='fc3')
    l_4 = DenseLayer(l_3, layersizes[3], W=weights[3], b=biases[3], nonlinearity=activations[3], name='bottleneck')
    return l_4