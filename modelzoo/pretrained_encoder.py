from lasagne.layers import DenseLayer


def create_pretrained_encoder(incoming, weights, biases, shapes, nonlinearities, names):
    encoder = DenseLayer(incoming, shapes[0], W=weights[0], b=biases[0], nonlinearity=nonlinearities[0], name=names[0])
    for i, num_units in enumerate(shapes[1:], 1):
        encoder = DenseLayer(encoder, shapes[i], W=weights[i], b=biases[i],
                             nonlinearity=nonlinearities[i], name=names[i])
    return encoder


def create_encoder(incoming, shapes, nonlinearities, names):
    encoder = DenseLayer(incoming, shapes[0], nonlinearity=nonlinearities[0], name=names[0])
    for i, num_units in enumerate(shapes[1:], 1):
        encoder = DenseLayer(encoder, shapes[i], nonlinearity=nonlinearities[i], name=names[i])
    return encoder
