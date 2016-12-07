"""
preprocess the images
"""
import sys
sys.path.append('../')
import argparse
from utils.io import load_mat_file, save_mat
from utils.preprocessing import normalize_input
from utils.preprocessing import sequencewise_mean_image_subtraction, compute_diff_images
from utils.plotting_utils import reshape_images_order


def reorder_images(data, shape):
    data = reshape_images_order(data, shape)
    return data


def samplewise_normalize(data):
    data = normalize_input(data)
    return data


def remove_mean(data, vidlens):
    data = sequencewise_mean_image_subtraction(data, vidlens)
    return data


def diff_image(data, vidlens):
    data = compute_diff_images(data, vidlens)
    return data


def parse_options():
    options = dict()
    options['remove_mean'] = False
    options['diff_image'] = False
    options['samplewise_norm'] = False
    options['no_reorder'] = False
    options['output'] = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_mean', action='store_true', help='remove mean image')
    parser.add_argument('--diff_image', action='store_true', help='compute difference of image')
    parser.add_argument('--samplewise_norm', action='store_true', help='samplewise normalize')
    parser.add_argument('--no_reorder', action='store_true', help='disable data reordering from f to c')
    parser.add_argument('--output', help='write output to .mat file')
    parser.add_argument('input', nargs='+', help='input ouluvs2 .mat file to preprocess')
    args = parser.parse_args()
    if args.remove_mean:
        options['remove_mean'] = args.remove_mean
    if args.diff_image:
        options['diff_image'] = args.diff_image
    if args.samplewise_norm:
        options['samplewise_norm'] = args.samplewise_norm
    if args.no_reorder:
        options['no_reorder'] = args.no_reorder
    if args.output:
        options['output'] = args.output
    if args.input:
        options['input'] = args.input[0]
    return options


def main():
    options = parse_options()
    data = load_mat_file(options['input'])
    dataMatrix = data['dataMatrix'].astype('float32')
    vidlens = data['videoLengthVec'].reshape((-1,))

    if not options['no_reorder']:
        dataMatrix = reorder_images(dataMatrix, (26, 44))
    if options['samplewise_norm']:
        dataMatrix = samplewise_normalize(dataMatrix)
    if options['remove_mean']:
        dataMatrix = remove_mean(dataMatrix, vidlens)
    if options['diff_image']:
        dataMatrix = diff_image(dataMatrix, vidlens)

    data['dataMatrix'] = dataMatrix
    if options['output']:
        save_mat(data, options['output'])
    print('data prepared!')


if __name__ == '__main__':
    main()
