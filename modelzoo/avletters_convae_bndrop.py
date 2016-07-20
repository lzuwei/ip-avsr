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
    bn1 = BatchNormLayer(conv2d1, name='batchnorm1')
    maxpool2d2 = MaxPool2DLayer(bn1, pool_size=pool_size, name='maxpool2d2')
    dropout1 = DropoutLayer(maxpool2d2, name='dropout1')
    conv2d3 = Conv2DLayer(dropout1, num_filters=conv_num_filters2, filter_size=filter_size2, pad=pad_in, name='conv2d3', nonlinearity=scaled_tanh)
    bn2 = BatchNormLayer(conv2d3, name='batchnorm2')
    maxpool2d4 = MaxPool2DLayer(bn2, pool_size=pool_size, name='maxpool2d4', pad=(1,0))
    dropout2 = DropoutLayer(maxpool2d4, name='dropout2')
    conv2d5 = Conv2DLayer(dropout2, num_filters=conv_num_filters3, filter_size=filter_size3, pad=pad_in, name='conv2d5', nonlinearity=scaled_tanh)
    bn3 = BatchNormLayer(conv2d5, name='batchnorm3')
    reshape6 = ReshapeLayer(bn3, shape=([0], -1), name='reshape6')  # 3000
    reshape6_output = reshape6.output_shape[1]
    dropout3 = DropoutLayer(reshape6, name='dropout3')
    dense7 = DenseLayer(dropout3, num_units=dense_mid_size, name='dense7', nonlinearity=scaled_tanh)
    bn4 = BatchNormLayer(dense7, name='batchnorm4')
    dropout4 = DropoutLayer(bn4, name='dropout4')
    bottleneck = DenseLayer(dropout4, num_units=encode_size, name='bottleneck', nonlinearity=linear)
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
    return reshape16, bottleneck
