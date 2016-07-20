from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer, BatchNormLayer, batch_norm
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, linear, sigmoid, ScaledTanh
from lasagne.layers import Conv2DLayer, Deconv2DLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer


def create_scaled_tanh(scale_in=2./3, scale_out=1.7159):
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


def extract_encoder(network):
    pass


def create_model(incoming, options):
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
    dropout0 = DropoutLayer(incoming, p=0.2, name='dropout0')
    conv2d1 = Conv2DLayer(dropout0, num_filters=conv_num_filters1, filter_size=filter_size1, pad=pad_in, name='conv2d1', nonlinearity=scaled_tanh)
    bn2 = BatchNormLayer(conv2d1, name='batchnorm2')
    maxpool2d3 = MaxPool2DLayer(bn2, pool_size=pool_size, name='maxpool2d3')
    conv2d4 = Conv2DLayer(maxpool2d3, num_filters=conv_num_filters2, filter_size=filter_size2, pad=pad_in, name='conv2d4', nonlinearity=scaled_tanh)
    bn5 = BatchNormLayer(conv2d4, name='batchnorm5')
    maxpool2d6 = MaxPool2DLayer(bn5, pool_size=pool_size, name='maxpool2d6', pad=(1,0))
    conv2d7 = Conv2DLayer(maxpool2d6, num_filters=conv_num_filters3, filter_size=filter_size3, pad=pad_in, name='conv2d7', nonlinearity=scaled_tanh)
    bn8 = BatchNormLayer(conv2d7, name='batchnorm8')
    reshape9 = ReshapeLayer(bn8, shape=([0], -1), name='reshape9')  # 3000
    reshape9_output = reshape9.output_shape[1]
    dense10 = DenseLayer(reshape9, num_units=dense_mid_size, name='dense7', nonlinearity=scaled_tanh)
    bn11 = BatchNormLayer(dense10, name='batchnorm11')
    bottleneck = DenseLayer(bn11, num_units=encode_size, name='bottleneck', nonlinearity=linear)
    # print_network(bottleneck)
    dense12 = DenseLayer(bottleneck, num_units=dense_mid_size, W=bottleneck.W.T, name='dense12', nonlinearity=linear)
    dense13 = DenseLayer(dense12, num_units=reshape9_output, W=dense10.W.T, nonlinearity=scaled_tanh, name='dense13')
    reshape14 = ReshapeLayer(dense13, shape=([0], conv_num_filters3, 3, 5), name='reshape14')  # 32 x 4 x 7
    deconv2d19 = Deconv2DLayer(reshape14, conv2d7.input_shape[1], conv2d7.filter_size, stride=conv2d7.stride,
                               W=conv2d7.W, flip_filters=not conv2d7.flip_filters, name='deconv2d19', nonlinearity=scaled_tanh)
    upscale2d16 = Upscale2DLayer(deconv2d19, scale_factor=pool_size, name='upscale2d16')
    deconv2d17 = Deconv2DLayer(upscale2d16, conv2d4.input_shape[1], conv2d4.filter_size, stride=conv2d4.stride,
                               W=conv2d4.W, flip_filters=not conv2d4.flip_filters, name='deconv2d17', nonlinearity=scaled_tanh)
    upscale2d18 = Upscale2DLayer(deconv2d17, scale_factor=pool_size, name='upscale2d18')
    deconv2d19 = Deconv2DLayer(upscale2d18, conv2d1.input_shape[1], conv2d1.filter_size, stride=conv2d1.stride,
                               crop=(1, 0), W=conv2d1.W, flip_filters=not conv2d1.flip_filters, name='deconv2d14', nonlinearity=scaled_tanh)
    reshape20 = ReshapeLayer(deconv2d19, ([0], -1), name='reshape20')
    return reshape20, bottleneck
