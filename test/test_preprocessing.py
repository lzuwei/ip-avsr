import unittest
from utils.io import *
from utils.preprocessing import *


class TestPreprocessingMethods(unittest.TestCase):
    def test_forcealign(self):
        stream1 = load_mat_file('../oulu/data/allMouthROIsResized_frontal.mat')
        stream2 = load_mat_file('../oulu/data/mfcc_w3s3.mat')

        s1_data_matrix = stream1['dataMatrix'].astype('float32')
        s1_targets = stream1['targetsVec'].reshape((-1,))
        s1_vidlens = stream1['videoLengthVec'].reshape((-1,))
        s1_subjects = stream1['subjectsVec'].reshape((-1,))

        s2_data_matrix = stream2['dataMatrix'].astype('float32')
        s2_targets = stream2['targetsVec'].reshape((-1,))
        s2_vidlens = stream2['videoLengthVec'].reshape((-1,))
        s2_subjects = stream2['subjectsVec'].reshape((-1,))

        s1, s2 = force_align((s1_data_matrix, s1_targets, s1_vidlens),
                             (s2_data_matrix, s2_targets, s2_vidlens))

        s1_data_matrix, s1_targets, s1_vidlens = s1
        s2_data_matrix, s2_targets, s2_vidlens = s2

        assert len(s1_data_matrix) == len(s2_data_matrix)
        assert len(s1_targets) == len(s2_targets)
        assert np.sum(s1_vidlens) == np.sum(s2_vidlens)

    def test_multistream_forcealign(self):

        stream1 = load_mat_file('../oulu/data/allMouthROIsResized_frontal.mat')
        stream2 = load_mat_file('../oulu/data/allMouthROIsResized_frontal.mat')
        stream3 = load_mat_file('../oulu/data/dctFeats_meanrm_w2s1.mat')
        stream4 = load_mat_file('../oulu/data/mfcc_w3s3.mat')

        s1_data_matrix = stream1['dataMatrix'].astype('float32')
        s1_targets = stream1['targetsVec'].reshape((-1,))
        s1_vidlens = stream1['videoLengthVec'].reshape((-1,))
        s1_subjects = stream1['subjectsVec'].reshape((-1,))

        s2_data_matrix = stream2['dataMatrix'].astype('float32')
        s2_targets = stream2['targetsVec'].reshape((-1,))
        s2_vidlens = stream2['videoLengthVec'].reshape((-1,))
        s2_subjects = stream2['subjectsVec'].reshape((-1,))

        s3_data_matrix = stream3['dataMatrix'].astype('float32')
        s3_targets = stream3['targetsVec'].reshape((-1,))
        s3_vidlens = stream3['videoLengthVec'].reshape((-1,))
        s3_subjects = stream3['subjectsVec'].reshape((-1,))

        s4_data_matrix = stream4['dataMatrix'].astype('float32')
        s4_targets = stream4['targetsVec'].reshape((-1,))
        s4_vidlens = stream4['videoLengthVec'].reshape((-1,))
        s4_subjects = stream4['subjectsVec'].reshape((-1,))

        orig_streams = [
            (s1_data_matrix, s1_targets, s1_vidlens),
            (s2_data_matrix, s2_targets, s2_vidlens),
            (s3_data_matrix, s3_targets, s3_vidlens),
            (s4_data_matrix, s4_targets, s4_vidlens)
        ]

        a = multistream_force_align(orig_streams)
        assert len(a[0][0]) == len(a[1][0]) == len(a[2][0]) == len(a[3][0])
        assert len(a[0][1]) == len(a[1][1]) == len(a[2][1]) == len(a[3][1])
        assert len(a[0][2]) == len(a[1][2]) == len(a[2][2]) == len(a[3][2])

if __name__ == '__main__':
    unittest.main()
