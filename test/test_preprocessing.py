import unittest
from utils.io import *
from utils.preprocessing import *


class TestPreprocessingMethods(unittest.TestCase):
    def test_forcealign(self):
        stream1 = load_mat_file('../cuave/data/allData_mouthROIs_basedOnMouthCenter.mat')
        stream2 = load_mat_file('../cuave/data/mfcc_w3s3.mat')

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


if __name__ == '__main__':
    unittest.main()
