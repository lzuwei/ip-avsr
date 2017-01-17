import unittest
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import rectify, linear
from modelzoo import deltanet_majority_vote
from utils.io import save_mat


class TestModelIO(unittest.TestCase):
    def test_load_params(self):
        window = T.iscalar('theta')
        inputs1 = T.tensor3('inputs1', dtype='float32')
        mask = T.matrix('mask', dtype='uint8')
        network = deltanet_majority_vote.load_saved_model('../cuave/results/best_models/1stream_dct_w1s1.1.pkl',
                                                          ([200, 100, 50, 20], [rectify, rectify, rectify, linear]),
                                                          (None, None, 90), inputs1, (None, None), mask,
                                                          250, window, 10)
        d = deltanet_majority_vote.extract_encoder_weights(network, ['fc1', 'fc2', 'fc3', 'bottleneck'],
                                                           [('w1', 'b1'), ('w2', 'b2'), ('w3', 'b3'), ('w4', 'b4')])
        expected_keys = ['w1', 'w2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4']
        keys = d.keys()
        for k in keys:
            assert k in expected_keys
            assert type(d[k]) == np.ndarray
        save_mat(d, '../cuave/models/cuave_1stream_dct_w1s1.mat')


if __name__ == '__main__':
    unittest.main()
