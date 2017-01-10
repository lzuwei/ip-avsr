import matplotlib.pylab as plt
import numpy
import math
from tabulate import tabulate
from matplotlib import cm
import lasagne as las
from PIL import Image


# import pdb

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True,
                       cmap='gray'):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats
  
    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = numpy.zeros(out_shape,
                                                 dtype='uint8' if output_pixel_vals else out_array.dtype
                                                 ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing,
                                                        scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        # print(tile_shape)
        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = _scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] \
                        = this_img * (255 if output_pixel_vals else 1)
        plt.figure()
        plt.imshow(out_array, interpolation='nearest', cmap=cmap)

        return out_array


def _scale_to_unit_interval(ndar, eps=1e-8):
    """
    Scales all values in the ndarray ndar to be between 0 and 1
    """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def plot_validation_cost(train_error, val_error, class_rate=None, savefilename=None):
    epochs = range(len(train_error))
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, train_error, label='train error')
    ax1.plot(epochs, val_error, label='validation error')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('cost')
    plt.title('Validation Cost')
    lines = ax1.get_lines()
    # Shrink current axis's height by 10% on the bottom
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    if class_rate is not None:
        ax2 = plt.twinx(ax1)
        ax2.plot(epochs, class_rate, label='classification rate', color='r')
        ax2.set_ylabel('classification rate')
        lines.extend(ax2.get_lines())
        ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                          box.width, box.height * 0.9])

    labels = [l.get_label() for l in lines]
    # Put a legend below current axis
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=False, shadow=False, ncol=5)
    # ax1.legend(lines, labels, loc='lower right')
    if savefilename:
        plt.savefig(savefilename)
    plt.show()


def visualize_images(images, shape=(30, 40), savefilename=None):
    w = int(math.sqrt(len(images)))
    vis = tile_raster_images(images, shape, (w, w), tile_spacing=(1, 1))
    plt.title('images')
    plt.show()
    if savefilename:
        o = Image.fromarray(vis)
        o.save('{}.png'.format(savefilename))


def visualize_sequence(sequence, shape=(30, 40), savefilename=None, title='sequence'):
    cols = int(math.ceil(len(sequence) / 2.0))
    vis = tile_raster_images(sequence, shape, (2, cols), tile_spacing=(1, 1))
    plt.title(title)
    plt.axis('off')
    plt.show()
    if savefilename:
        o = Image.fromarray(vis)
        o.save('{}.png'.format(savefilename))


def visualize_reconstruction(original, reconstructed, shape=(30, 40),
                             savefilename=None, title1='original', title2='reconstructed'):
    w = int(math.sqrt(len(original)))
    orig = tile_raster_images(original, shape, (w, w), tile_spacing=(1, 1))
    plt.title(title1)
    recon = tile_raster_images(reconstructed, shape, (w, w), tile_spacing=(1, 1))
    plt.title(title2)
    plt.show()
    if savefilename:
        o = Image.fromarray(orig)
        o.save('{}_orig.png'.format(savefilename))
        r = Image.fromarray(recon)
        r.save('{}_recon.png'.format(savefilename))


def visualize_layer(layer, row, col, w, h):
    W_encode = layer.W.get_value()
    tile_raster_images(W_encode.T, (row, col), (w, h), tile_spacing=(1, 1))
    plt.title('filters')
    plt.show()


def visualize_activations(weights, examples, shape, weight_idx_to_visualize, savefilename=None):
    """
    Visualize the activations. Does not check bounds of weight index to visualize,
    ensure it falls within the bounds of the weights being passed in.
    :param weights: matrix of weights
    :param examples: examples to visualize
    :param shape: shape of image
    :param weight_idx_to_visualize: list of weight indexes to visualize
    """
    for i in weight_idx_to_visualize:
        w = weights[:, i]
        activations = w * examples
        if savefilename:
            visualize_reconstruction(examples, activations, shape,
                                     '{}_w{}'.format(savefilename, i), 'Raw', 'Activations')
        else:
            visualize_reconstruction(examples, activations, shape,
                                     None, 'Raw', 'Activations')


def plot_confusion_matrix(conf_mat, headers, fmt='pipe', savefilename=None):
    """
    pretty print confusion matrix in various formats
    plain, simple, grid, fancy_grid, pipe, orgtbl, rst, mediawiki, html, latex, latex_booktabs
    defaults to 'pipe' which is a markdown parseable format.
    :param conf_mat: confusion matrix
    :param headers: list of headers eg: ['a','b','c','d']
    :param fmt: pipe, grid, latex, html, rst, simple,
    :return:
    """
    data = []
    for i, header in enumerate(headers):
        row = [header] + conf_mat[i].tolist()
        data.append(row)
    table = tabulate(data, headers, tablefmt=fmt)
    if savefilename:
        with open(savefilename, mode='a') as f:
            f.write(table)
            f.write('\n')
    return table


def reshape_images_order(X, shape, fr='f', to='c'):
    assert len(shape) == 2
    for i, x in enumerate(X):
        X[i] = x.reshape(shape, order=fr).reshape((shape[0] * shape[1]), order=to)
    return X


def show_image(data, shape, order='f', cmap=cm.gray):
    """
    display an image from a 1d vector
    :param data: 1d vector containing image information
    :param shape: actual image dimensions
    :param order: 'c' or 'f'
    :param cmap: colour map, defaults to grayscale
    :return:
    """
    img = data.reshape(shape, order=order)
    plt.imshow(img, cmap=cmap)
    plt.show()


def print_layer_shape(layer):
    """
    print out layer name and shape
    :param layer: neural layer
    :return:
    """
    print('[L] {}: {}'.format(layer.name, las.layers.get_output_shape(layer)))


def print_network(network):
    """
    print out network structure
    :param network: neural net
    :return:
    """
    layers = las.layers.get_all_layers(network)
    for layer in layers:
        print_layer_shape(layer)


def test_plot_validation_cost():
    train_error = [10, 9, 8, 7, 6, 5, 4, 3]
    val_error = [15, 14, 13, 12, 11, 10, 9, 8]
    class_rate = [80, 81, 82, 83, 84, 85, 86, 87]
    plot_validation_cost(train_error, val_error, class_rate)
