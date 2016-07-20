from __future__ import print_function
import os, sys, urllib, gzip, time, argparse
sys.path.insert(0, '../')
try:
    import cPickle as pickle
except:
    import pickle
sys.setrecursionlimit(10000)
import numpy as np
import theano
import theano.tensor as T
import lasagne as las
import matplotlib
matplotlib.use('Agg')  # Change matplotlib backend, in case we have no X server running..
import matplotlib.pyplot as plt

# Lasagne Imports
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer, BatchNormLayer, batch_norm
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, linear, sigmoid, ScaledTanh
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import visualize

# GPU detection
from lasagne.layers import Conv2DLayer, Deconv2DLayer
from lasagne.layers import MaxPool2DLayer
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')

from utils.plotting_utils import print_network, visualize_reconstruction, visualize_layer, plot_validation_cost
from utils.datagen import batch_iterator, sequence_batch_iterator
from utils.preprocessing import *
from modelzoo import avletters_convae, avletters_convae_bndrop, avletters_convae_bn
from utils.io import *


def configure_theano():
    theano.config.floatX = 'float32'
    sys.setrecursionlimit(10000)


def create_scaled_tanh(scale_in=0.5, scale_out=2.4):
    """
    create a scaled hyperbolic tangent to avoid saturation given input range
    of {-1, 1}. Refer to
    :param scale_in:
    :param scale_out:
    :return: scaled hyperbolic tangent callable

    References
    ----------
    .. [1] LeCun, Yann A., et al. (1998):
       Efficient BackProp,
       http://link.springer.com/chapter/10.1007/3-540-49430-8_2,
       http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    .. [2] Masci, Jonathan, et al. (2011):
       Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction,
       http://link.springer.com/chapter/10.1007/978-3-642-21735-7_7,
       http://people.idsia.ch/~ciresan/data/icann2011.pdf
    """
    return ScaledTanh(scale_in, scale_out)


def create_model(input_var, input_shape, options):
    conv_num_filters1 = 100
    conv_num_filters2 = 150
    conv_num_filters3 = 200
    filter_size1 = 5
    filter_size2 = 5
    filter_size3 = 3
    pool_size = 2
    encode_size = options['BOTTLENECK']
    dense_mid_size = options['DENSE']
    pad_in = 'valid'
    pad_out = 'full'
    scaled_tanh = create_scaled_tanh()

    input = InputLayer(shape=input_shape, input_var=input_var, name='input')
    conv2d1 = Conv2DLayer(input, num_filters=conv_num_filters1, filter_size=filter_size1, pad=pad_in, name='conv2d1', nonlinearity=scaled_tanh)
    maxpool2d2 = MaxPool2DLayer(conv2d1, pool_size=pool_size, name='maxpool2d2')
    conv2d3 = Conv2DLayer(maxpool2d2, num_filters=conv_num_filters2, filter_size=filter_size2, pad=pad_in, name='conv2d3', nonlinearity=scaled_tanh)
    maxpool2d4 = MaxPool2DLayer(conv2d3, pool_size=pool_size, name='maxpool2d4', pad=(1,0))
    conv2d5 = Conv2DLayer(maxpool2d4, num_filters=conv_num_filters3, filter_size=filter_size3, pad=pad_in, name='conv2d5', nonlinearity=scaled_tanh)
    reshape6 = ReshapeLayer(conv2d5, shape=([0], -1), name='reshape6')  # 3000
    reshape6_output = reshape6.output_shape[1]
    dense7 = DenseLayer(reshape6, num_units=dense_mid_size, name='dense7', nonlinearity=scaled_tanh)
    bottleneck = DenseLayer(dense7, num_units=encode_size, name='bottleneck', nonlinearity=linear)
    # print_network(bottleneck)
    dense8 = DenseLayer(bottleneck, num_units=dense_mid_size, W=bottleneck.W.T, name='dense8', nonlinearity=linear)
    dense9 = DenseLayer(dense8, num_units=reshape6_output, W=dense7.W.T, nonlinearity=scaled_tanh, name='dense9')
    reshape10 = ReshapeLayer(dense9, shape=([0], conv_num_filters3, 3, 5), name='reshape10')  # 32 x 4 x 7
    deconv2d11 = Deconv2DLayer(reshape10, conv2d5.input_shape[1], conv2d5.filter_size, stride=conv2d5.stride,
                               W=conv2d5.W, flip_filters=not conv2d5.flip_filters, name='deconv2d11', nonlinearity=scaled_tanh)
    upscale2d12 = Upscale2DLayer(deconv2d11, scale_factor=pool_size, name='upscale2d12')
    deconv2d13 = Deconv2DLayer(upscale2d12, conv2d3.input_shape[1], conv2d3.filter_size, stride=conv2d3.stride,
                               W=conv2d3.W, flip_filters=not conv2d3.flip_filters, name='deconv2d13', nonlinearity=scaled_tanh)
    upscale2d14 = Upscale2DLayer(deconv2d13, scale_factor=pool_size, name='upscale2d14')
    deconv2d15 = Deconv2DLayer(upscale2d14, conv2d1.input_shape[1], conv2d1.filter_size, stride=conv2d1.stride,
                               crop=(1, 0), W=conv2d1.W, flip_filters=not conv2d1.flip_filters, name='deconv2d14', nonlinearity=scaled_tanh)
    reshape16 = ReshapeLayer(deconv2d15, ([0], -1), name='reshape16')
    print_network(reshape16)
    return reshape16


def apply_gaussian_noise(input, noise_factor=0.4):
    input_noisy = input + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input.shape)
    input_noisy = np.clip(input_noisy, 0., 1.)
    return input_noisy


def generate_data(path='data/allData_mouthROIs.mat'):
    print('preprocessing dataset...')
    data = load_mat_file(path)

    # create the necessary variable mappings
    data_matrix = data['dataMatrix']
    data_matrix_len = data_matrix.shape[0]
    targets_vec = data['targetsVec']
    vid_len_vec = data['videoLengthVec']
    iter_vec = data['iterVec']

    indexes = create_split_index(data_matrix_len, vid_len_vec, iter_vec)
    train_vidlen_vec, test_vidlen_vec = split_videolen(vid_len_vec, iter_vec)
    assert len(train_vidlen_vec) == 520
    assert len(test_vidlen_vec) == 260
    assert np.sum(vid_len_vec) == data_matrix_len

    # split the data
    train_data = data_matrix[indexes == True]
    train_targets = targets_vec[indexes == True]
    train_targets = train_targets.reshape((len(train_targets),))
    test_data = data_matrix[indexes == False]
    test_targets = targets_vec[indexes == False]
    test_targets = test_targets.reshape((len(test_targets),))

    # indexes for a particular letter
    # idx = [i for i, elem in enumerate(test_targets) if elem == 20]

    # resize the input data to 40 x 30
    train_data_resized = resize_images(train_data).astype(np.float32)

    # normalize the inputs [0 - 1]
    train_data_resized = normalize_input(train_data_resized, centralize=True)

    test_data_resized = resize_images(test_data).astype(np.float32)
    test_data_resized = normalize_input(test_data_resized, centralize=True)

    # reshape to 30, 40 (row, col) image
    train_data_resized = np.reshape(train_data_resized, (-1, 30, 40))
    test_data_resized = np.reshape(test_data_resized, (-1, 30, 40))
    # plt.imshow(train_data_resized[0], cmap='gray')

    return train_data_resized, test_data_resized


def batch_compute_cost(X, y, no_strides, cost_fn):
    cost = 0.0
    stride_size = len(X) / no_strides
    for j in range(no_strides):
        j *= stride_size
        cost += cost_fn(X[j:j + stride_size], y[j:j + stride_size])
    return cost / float(no_strides)


def parse_options():
    options = dict()
    options['NUM_EPOCHS'] = 20
    options['EPOCH_SIZE'] = 96
    options['NO_STRIDES'] = 3
    options['VAL_NO_STRIDES'] = 3
    options['DENSE'] = 300
    options['BOTTLENECK'] = 50
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs to run')
    parser.add_argument('--bottleneck', help='bottleneck size')
    parser.add_argument('--dense', help='dense layer size')
    args = parser.parse_args()
    if args.epochs:
        options['NUM_EPOCHS'] = int(args.epochs)
    if args.bottleneck:
        options['BOTTLENECK'] = int(args.bottleneck)
    if args.dense:
        options['DENSE'] = int(args.dense)
    return options


def main():
    configure_theano()
    options = parse_options()
    X, X_val = generate_data()

    # X = np.reshape(X, (-1, 1, 30, 40))[:-5]
    print('X type and shape:', X.dtype, X.shape)
    print('X.min():', X.min())
    print('X.max():', X.max())

    # X_val = np.reshape(X_val, (-1, 1, 30, 40))[:-1]
    print('X_val type and shape:', X_val.dtype, X_val.shape)
    print('X_val.min():', X_val.min())
    print('X_val.max():', X_val.max())

    # we need our target to be 1 dimensional
    X_out = X.reshape((X.shape[0], -1))
    X_val_out = X_val.reshape((X_val.shape[0], -1))
    print('X_out:', X_out.dtype, X_out.shape)
    print('X_val_out', X_val_out.dtype, X_val_out.shape)

    # X_noisy = apply_gaussian_noise(X_out)
    # visualize_reconstruction(X_noisy[0:25], X_out[0:25], shape=(28, 28))
    # X = np.reshape(X_noisy, (-1, 1, 28, 28))

    print('constructing and compiling model...')
    # input_var = T.tensor4('input', dtype='float32')
    input_var = T.tensor3('input', dtype='float32')
    target_var = T.matrix('output', dtype='float32')
    lr = theano.shared(np.array(0.8, dtype=theano.config.floatX), name='learning_rate')
    lr_decay = np.array(0.5, dtype=theano.config.floatX)

    # try building a reshaping layer
    # network = create_model(input_var, (None, 1, 30, 40), options)
    l_input = InputLayer((None, None, 1200), input_var, name='input')
    l_input = ReshapeLayer(l_input, (-1, 1, 30, 40), name='reshape_input')
    # l_input = InputLayer((None, 1, 30, 40), input_var, name='input')
    network, encoder = avletters_convae_bn.create_model(l_input, options)

    print('AE Network architecture:')
    print_network(network)

    recon = las.layers.get_output(network, deterministic=False)
    all_params = las.layers.get_all_params(network, trainable=True)
    cost = T.mean(las.objectives.squared_error(recon, target_var))
    updates = las.updates.adadelta(cost, all_params, lr)
    # updates = las.updates.apply_nesterov_momentum(updates, all_params, momentum=0.8)

    train = theano.function([input_var, target_var], recon, updates=updates, allow_input_downcast=True)
    train_cost_fn = theano.function([input_var, target_var], cost, allow_input_downcast=True)

    eval_recon = las.layers.get_output(network, deterministic=True)
    eval_cost = T.mean(las.objectives.squared_error(eval_recon, target_var))
    eval_cost_fn = theano.function([input_var, target_var], eval_cost, allow_input_downcast=True)
    recon_fn = theano.function([input_var], eval_recon, allow_input_downcast=True)

    NUM_EPOCHS = options['NUM_EPOCHS']
    EPOCH_SIZE = options['EPOCH_SIZE']
    NO_STRIDES = options['NO_STRIDES']
    VAL_NO_STRIDES = options['VAL_NO_STRIDES']

    print('begin training for {} epochs...'.format(NUM_EPOCHS))
    datagen = batch_iterator(X, X_out, 128)

    costs = []
    val_costs = []
    for epoch in range(NUM_EPOCHS):
        time_start = time.time()
        for i in range(EPOCH_SIZE):
            batch_X, batch_y = next(datagen)
            print_str = 'Epoch {} batch {}/{}: {} examples'.format(epoch + 1, i + 1, EPOCH_SIZE, len(batch_X))
            print(print_str, end='')
            sys.stdout.flush()
            batch_X = batch_X.reshape((-1, 1, 1200))
            train(batch_X, batch_y)
            print('\r', end='')

        cost = batch_compute_cost(X, X_out, NO_STRIDES, train_cost_fn)
        val_cost = batch_compute_cost(X_val, X_val_out, VAL_NO_STRIDES, eval_cost_fn)
        costs.append(cost)
        val_costs.append(val_cost)

        print("Epoch {} train cost = {}, validation cost = {} ({:.1f}sec) "
              .format(epoch + 1, cost, val_cost, time.time() - time_start))
        if epoch >= 10 and epoch % 10 == 0:
            lr.set_value(lr.get_value() * lr_decay)
            print('Learning rate decay to: {}'.format(lr.get_value()))

    X_val_recon = recon_fn(X_val)
    visualize_reconstruction(X_val_out[450:550], X_val_recon[450:550], shape=(30, 40), savefilename='avletters')
    plot_validation_cost(costs, val_costs, None, savefilename='valid_cost')

    conv2d1 = las.layers.get_all_layers(network)[3]
    visualize.plot_conv_weights(conv2d1, (10, 10)).savefig('conv2d1.png')

    print('saving encoder...')
    save_model(encoder, 'models/conv_encoder.dat')
    save_model(network, 'models/conv_ae.dat')


if __name__ == '__main__':
    main()
