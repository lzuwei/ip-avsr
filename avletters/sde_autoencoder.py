import sys
import os
import time

import theano.tensor as T

import lasagne as las
from preprocessing import *
# import sklearn.datasets

import numpy as np
import scipy.io
from lasagne.layers import get_output, InputLayer, DenseLayer, GaussianNoiseLayer, DropoutLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, linear, sigmoid
from lasagne.updates import nesterov_momentum, adadelta
from lasagne.objectives import squared_error
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from nolearn.lasagne import visualize
from nolearn.lasagne import visualize
from nolearn.lasagne.visualize import plot_loss
from lasagne.init import GlorotUniform
from plotting_utils import tile_raster_images
import theano
from matplotlib import pyplot as plt


def build_encoder(input_var=None, encoder_units=None):
    layer_input = las.layers.InputLayer(shape=(None, 1200),
                                        input_var=input_var)
    layer_encoder = las.layers.DenseLayer(
        layer_input, num_units=encoder_units,
        nonlinearity=las.nonlinearities.sigmoid,
        W=las.init.GlorotUniform())
    layer_decoder = las.layers.DenseLayer(layer_encoder, num_units=1200,
                                          nonlinearity=None,
                                          W=layer_encoder.W.T)
    return layer_decoder


def build_encoder_layers(input_size, encode_size, sigma=0.5):
    """
    builds an autoencoder with gaussian noise layer
    :param input_size: input size
    :param encode_size: encoded size
    :param sigma: gaussian noise standard deviation
    :return: Weights of encoder layer, denoising autoencoder layer
    """
    W = theano.shared(GlorotUniform().sample(shape=(input_size, encode_size)))

    layers = [
        (InputLayer, {'shape': (None, input_size)}),
        (GaussianNoiseLayer, {'name': 'corrupt', 'sigma': sigma}),
        (DenseLayer, {'name': 'encoder', 'num_units': encode_size, 'nonlinearity': sigmoid, 'W': W}),
        (DenseLayer, {'name': 'decoder', 'num_units': input_size, 'nonlinearity': linear, 'W': W.T}),
    ]
    return W, layers


def build_bottleneck_layer(input_size, encode_size, sigma=0.3):
    W = theano.shared(GlorotUniform().sample(shape=(input_size, encode_size)))

    layers = [
        (InputLayer, {'shape': (None, input_size)}),
        (GaussianNoiseLayer, {'name': 'corrupt', 'sigma': sigma}),
        (DenseLayer, {'name': 'encoder', 'num_units': encode_size, 'nonlinearity': linear, 'W': W}),
        (DenseLayer, {'name': 'decoder', 'num_units': input_size, 'nonlinearity': linear, 'W': W.T}),
    ]
    return W, layers


def create_decoder(input_size, decode_size, weights):
    decoder_layers = [
        (InputLayer, {'shape': (None, input_size)}),
        (DenseLayer, {'name': 'decoder', 'num_units': decode_size, 'nonlinearity': linear, 'W': weights})
    ]

    decoder = NeuralNet(
        layers=decoder_layers,
        max_epochs=50,
        objective_loss_function=squared_error,
        update=adadelta,
        regression=True,
        verbose=1
    )
    return decoder


def normalize_input(input, centralize=True, quantize=False):

    def center(item):
        item = item - item.mean()
        item = item / np.std(item)
        return item

    def rescale(item):
        min = np.min(item)
        max = np.max(item)
        item = (item - min) / (max - min)
        return item

    for i, item in enumerate(input):
        if centralize:
            input[i] = center(item)
        if quantize:
            input[i] = rescale(item)
    return input


def main():
    data = load_av_letters('data/allData_mouthROIs.mat')

    # create the necessary variable mappings
    data_matrix = data['dataMatrix']
    data_matrix_len = data_matrix.shape[0]
    targets_vec = data['targetsVec']
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']

    indexes = create_split_index(data_matrix_len, vid_len_vec, iter_vec)

    # split the data
    train_data = data_matrix[indexes == True]
    train_targets = targets_vec[indexes == True]
    test_data = data_matrix[indexes == False]
    test_targets = targets_vec[indexes == False]

    idx = [i for i, elem in enumerate(test_targets) if elem == 20]

    print(train_data.shape)
    print(test_data.shape)
    print(sum([train_data.shape[0], test_data.shape[0]]))

    # resize the input data to 40 x 30
    train_data_resized = resize_images(train_data).astype(np.float32)

    # normalize the inputs [0 - 1]
    train_data_resized = normalize_input(train_data_resized, centralize=True)

    test_data_resized = resize_images(test_data).astype(np.float32)
    test_data_resized = normalize_input(test_data_resized, centralize=True)

    dic = {}
    dic['trainDataResized'] = train_data_resized
    dic['testDataResized'] = test_data_resized

    """second experiment: overcomplete sigmoid encoder/decoder, squared loss"""
    encode_size = 2500
    sigma = 0.5

    # to get tied weights in the encoder/decoder, create this shared weightMatrix
    # 1200 x 2000
    w1, layer1 = build_encoder_layers(1200, 2500, sigma)

    ae1 = NeuralNet(
        layers=layer1,
        max_epochs=50,
        objective_loss_function=squared_error,
        update=adadelta,
        regression=True,
        verbose=1
    )

    load = True
    save = False
    if load:
        print('[LOAD] layer 1...')
        ae1.load_params_from('layer1.dat')
    else:
        print('[TRAIN] layer 1...')
        ae1.fit(train_data_resized, train_data_resized)

    # save params
    if save:
        print('[SAVE] layer 1...')
        ae1.save_params_to('layer1.dat')

    train_encoded1 = ae1.get_output('encoder', train_data_resized)  # 12293 x 2000

    w2, layer2 = build_encoder_layers(2500, 1250)
    ae2 = NeuralNet(
        layers=layer2,
        max_epochs=50,
        objective_loss_function=squared_error,
        update=adadelta,
        regression=True,
        verbose=1
    )

    load2 = True
    if load2:
        print('[LOAD] layer 2...')
        ae2.load_params_from('layer2.dat')
    else:
        print('[TRAIN] layer 2...')
        ae2.fit(train_encoded1, train_encoded1)

    save2 = False
    if save2:
        print('[SAVE] layer 2...')
        ae2.save_params_to('layer2.dat')

    train_encoded2 = ae2.get_output('encoder', train_encoded1)  # 12293 x 1250

    w3, layer3 = build_encoder_layers(1250, 600)
    ae3 = NeuralNet(
        layers=layer3,
        max_epochs=100,
        objective_loss_function=squared_error,
        update=adadelta,
        regression=True,
        verbose=1
    )

    load3 = True
    if load3:
        print('[LOAD] layer 3...')
        ae3.load_params_from('layer3.dat')
    else:
        ae3.fit(train_encoded2, train_encoded2)

    save3 = False
    if save3:
        print('[SAVE] layer 3...')
        ae3.save_params_to('layer3.dat')

    train_encoded3 = ae3.get_output('encoder', train_encoded2)  # 12293 x 1250

    w4, layer4 = build_bottleneck_layer(600, 100)
    ae4 = NeuralNet(
        layers=layer4,
        max_epochs=100,
        objective_loss_function=squared_error,
        update=adadelta,
        regression=True,
        verbose=1
    )

    load4 = False
    if load4:
        print('[LOAD] layer 4...')
        ae4.load_params_from('layer4.dat')
    else:
        print('[TRAIN] layer 4...')
        ae4.fit(train_encoded3, train_encoded3)

    save4 = True
    if save4:
        print('[SAVE] layer 4...')
        ae4.save_params_to('layer4.dat')

    test_enc1 = ae1.get_output('encoder', test_data_resized)
    test_enc2 = ae2.get_output('encoder', test_enc1)
    test_enc3 = ae3.get_output('encoder', test_enc2)
    test_enc4 = ae4.get_output('encoder', test_enc3)

    decoder4 = create_decoder(100, 600, w4.T)
    decoder4.initialize()
    decoder3 = create_decoder(600, 1250, w3.T)
    decoder3.initialize()
    decoder2 = create_decoder(1250, 2500, w2.T)
    decoder2.initialize()  # initialize the net
    decoder1 = create_decoder(2500, 1200, w1.T)
    decoder1.initialize()

    test_dec3 = decoder4.predict(test_enc4)
    test_dec2 = decoder3.predict(test_dec3)
    test_dec1 = decoder2.predict(test_dec2)
    X_pred = decoder1.predict(test_dec1)

    # plot_loss(ae3)
    # plot_loss(ae2)
    # plot_loss(ae1)
    tile_raster_images(X_pred[4625:4650, :], (30, 40), (5, 5), tile_spacing=(1, 1))
    plt.title('reconstructed')
    tile_raster_images(test_data_resized[4625:4650, :], (30, 40), (5, 5), tile_spacing=(1, 1))
    plt.title('original')
    plt.show()

    """
    W_encode = ae1.layers_['encoder'].W.get_value()
    tile_raster_images(W_encode.T, (30, 40), (50, 50), tile_spacing=(1, 1))
    plt.title('filters')
    plt.show()
    """

if __name__ == '__main__':
    main()
