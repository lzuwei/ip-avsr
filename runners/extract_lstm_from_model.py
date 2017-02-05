from __future__ import print_function
import sys
sys.path.insert(0, '../')
import numpy as np
import theano.tensor as T
import argparse
from modelzoo import deltanet_majority_vote
from utils.io import save_mat
from custom.nonlinearities import select_nonlinearity


def parse_options():
    options = dict()
    options['shape'] = '2000,1000,500,50'
    options['nonlinearities'] = 'rectify,rectify,rectify,linear'
    options['input_dim'] = 1200
    options['lstm_size'] = 250
    options['output_classes'] = 26
    options['layer_names'] = 'f_lstm,b_lstm'
    options['use_blstm'] = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', help='shape of encoder. Default: 2000,1000,500,50')
    parser.add_argument('--input_dim', help='input dimension. Default: 1200')
    parser.add_argument('--nonlinearities', help='nolinearities used by encodeer. '
                                                 'Default: rectify,rectify,rectify,linear')
    parser.add_argument('--output', help='output file to write results')
    parser.add_argument('--lstm_size', help='lstm layer size. Default: 250')
    parser.add_argument('--output_classes', help='number of output classes. Default: 10')
    parser.add_argument('--layer_names', help='names of lstm layers to extract')
    parser.add_argument('--use_blstm', help='use blstm')
    parser.add_argument('input', help='input model.pkl file')

    args = parser.parse_args()
    options['input'] = args.input
    if args.shape:
        options['shape'] = args.shape
    if args.input_dim:
        options['input_dim'] = int(args.input_dim)
    if args.nonlinearities:
        options['nonlinearities'] = args.nonlinearities
    if args.lstm_size:
        options['lstm_size'] = int(args.lstm_size)
    if args.output_classes:
        options['output_classes'] = int(args.output_classes)
    if args.output:
        options['output'] = args.output
    if args.layer_names:
        options['layer_names'] = args.layer_names
    if args.use_blstm:
        options['use_blstm'] = True
    return options


def main():
    options = parse_options()
    print(options)
    window = T.iscalar('theta')
    inputs1 = T.tensor3('inputs1', dtype='float32')
    mask = T.matrix('mask', dtype='uint8')
    shape = [int(i) for i in options['shape'].split(',')]
    nonlinearities = [select_nonlinearity(s) for s in options['nonlinearities'].split(',')]
    layer_names = options['layer_names'].split(',')
    network = deltanet_majority_vote.load_saved_model(options['input'],
                                                      (shape, nonlinearities),
                                                      (None, None, options['input_dim']), inputs1, (None, None), mask,
                                                      options['lstm_size'], window, options['output_classes'],
                                                      use_blstm=options['use_blstm'])
    d = deltanet_majority_vote.extract_lstm_weights(network, layer_names, ['f_lstm', 'b_lstm'])
    expected_keys = ['f_lstm_w_hid_to_cell', 'f_lstm_w_hid_to_forgetgate', 'f_lstm_w_hid_to_ingate',
                     'f_lstm_w_hid_to_outgate', 'f_lstm_w_in_to_cell', 'f_lstm_w_in_to_forgetgate',
                     'f_lstm_w_in_to_ingate', 'f_lstm_w_in_to_outgate', 'f_lstm_b_cell', 'f_lstm_b_forgetgate',
                     'f_lstm_b_ingate', 'f_lstm_b_outgate',
                     'b_lstm_w_hid_to_cell', 'b_lstm_w_hid_to_forgetgate',
                     'b_lstm_w_hid_to_ingate', 'b_lstm_w_hid_to_outgate', 'b_lstm_w_in_to_cell', 'b_lstm_w_in_to_forgetgate',
                     'b_lstm_w_in_to_ingate', 'b_lstm_w_in_to_outgate', 'b_lstm_b_cell', 'b_lstm_b_forgetgate',
                     'b_lstm_b_ingate', 'b_lstm_b_outgate']
    keys = d.keys()
    for k in keys:
        assert k in expected_keys
        assert type(d[k]) == np.ndarray
    if 'output' in options:
        print('save extracted weights to {}'.format(options['output']))
        save_mat(d, options['output'])


if __name__ == '__main__':
    main()
