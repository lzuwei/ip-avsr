from __future__ import print_function
import sys
sys.path.insert(0, '../')
import os
import time
import pickle
import ConfigParser

import theano.tensor as T
import theano

import matplotlib
# matplotlib.use('Agg')  # Change matplotlib backend, in case we have no X server running..

from utils.preprocessing import *
from utils.plotting_utils import *
from utils.io import *

import numpy as np
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import tanh, linear, sigmoid, rectify, leaky_rectify
from lasagne.updates import nesterov_momentum, adadelta, sgd, norm_constraint
from lasagne.objectives import squared_error
from nolearn.lasagne import NeuralNet


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def load_ae(path, train_params):
    """
    load a pretrained dbn from path
    :param path: path to the .mat dbn
    :return: pretrained deep belief network
    """
    # create the network using weights from pretrain_nn.mat
    nn = sio.loadmat(path)
    w1 = nn['w1']
    w2 = nn['w2']
    w3 = nn['w3']
    w4 = nn['w4']
    w5 = nn['w5']
    w6 = nn['w6']
    w7 = nn['w7']
    w8 = nn['w8']
    b1 = nn['b1'][0]
    b2 = nn['b2'][0]
    b3 = nn['b3'][0]
    b4 = nn['b4'][0]
    b5 = nn['b5'][0]
    b6 = nn['b6'][0]
    b7 = nn['b7'][0]
    b8 = nn['b8'][0]

    layers = [
        (InputLayer, {'name': 'input', 'shape': (None, 1200)}),
        (DenseLayer, {'name': 'l1', 'num_units': 2000, 'nonlinearity': sigmoid, 'W': w1, 'b': b1}),
        (DenseLayer, {'name': 'l2', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w2, 'b': b2}),
        (DenseLayer, {'name': 'l3', 'num_units': 500, 'nonlinearity': sigmoid, 'W': w3, 'b': b3}),
        (DenseLayer, {'name': 'l4', 'num_units': 50, 'nonlinearity': linear, 'W': w4, 'b': b4}),
        (DenseLayer, {'name': 'l5', 'num_units': 500, 'nonlinearity': sigmoid, 'W': w5, 'b': b5}),
        (DenseLayer, {'name': 'l6', 'num_units': 1000, 'nonlinearity': sigmoid, 'W': w6, 'b': b6}),
        (DenseLayer, {'name': 'l7', 'num_units': 2000, 'nonlinearity': sigmoid, 'W': w7, 'b': b7}),
        (DenseLayer, {'name': 'output', 'num_units': 1200, 'nonlinearity': linear, 'W': w8, 'b': b8}),
    ]

    '''
    dbn = NeuralNet(
        layers=layers,
        max_epochs=30,
        objective_loss_function=squared_error,
        update=nesterov_momentum,
        regression=True,
        verbose=1,
        update_learning_rate=0.001,
        update_momentum=0.05,
        objective_l2=0.005,
    )
    '''

    dbn = NeuralNet(
        layers=layers,
        max_epochs=10,
        objective_loss_function=squared_error,
        update=adadelta,
        regression=True,
        verbose=1,
        update_learning_rate=0.01,
        # update_learning_rate=0.001,
        # update_momentum=0.05,
        objective_l2=0.005,
    )
    return dbn


def main():
    configure_theano()
    config_file = 'config/finetuner.ini'
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    print('loading config file: {}'.format(config_file))

    print('preprocessing dataset...')
    data = load_mat_file(config.get('data', 'images'))
    ae_pretrained = config.get('models', 'pretrained')
    ae_finetuned = config.get('models', 'finetuned')
    do_finetune = config.getboolean('training', 'do_finetune')
    save_finetune = config.getboolean('training', 'save_finetune')
    load_finetune = config.getboolean('training', 'load_finetune')
    train_params = dict()
    train_params['max_epochs'] = config.getint('training', 'max_epochs')
    train_params['learning_rate'] = config.getfloat('training', 'learning_rate')
    train_params['objective_l2'] = config.getfloat('training', 'objective_l2')

    # create the necessary variable mappings
    data_matrix = data['dataMatrix'].astype('float32')
    data_matrix_len = data_matrix.shape[0]
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']

    indexes = create_split_index(data_matrix_len, vid_len_vec, iter_vec)
    train_vidlen_vec, test_vidlen_vec = split_videolen(vid_len_vec, iter_vec)
    assert len(train_vidlen_vec) == 520
    assert len(test_vidlen_vec) == 260
    assert np.sum(vid_len_vec) == data_matrix_len

    data_matrix = normalize_input(data_matrix)

    # split the data
    train_data = data_matrix[indexes == True]
    test_data = data_matrix[indexes == False]

    if do_finetune:
        print('performing finetuning...')
        ae = load_ae(ae_pretrained, train_params)
        ae.initialize()
        #ae.fit(train_data, train_data)
        #res = ae.predict(test_data)
        # print(res.shape)
        #visualize_reconstruction(test_data[300:336], res[300:336])

    if save_finetune:
        print('saving finetuned encoder: {}...'.format(ae_finetuned))
        pickle.dump(ae, open(ae_finetuned, 'wb'))

    if load_finetune:
        print('loading finetuned encoder: {}'.format(ae_finetuned))
        ae = load_ae(ae_pretrained, train_params)
        # ae = pickle.load(open(ae_finetuned, 'rb'))
        ae.initialize()
        print('performing prediction...')
        res = ae.predict(test_data)
        visualize_reconstruction(test_data[300:336], res[300:336])
        print('done!')


if __name__ == '__main__':
    main()
