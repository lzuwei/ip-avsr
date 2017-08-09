import os
import unittest
import numpy as np
from utils.io import save_mat, load_mat_file
from utils.datagen import gen_batch_from_file


class TestGenBatchFromFile(unittest.TestCase):
    def test_large_batch(self):
        """
        test large batch where train data is larger than batch size
        :return: batch with data equal to batch size
        """
        # load the test file, and preprocess the path separator and dimensions
        data = load_mat_file('../5words/data/'
                             'data5Words_mouthROIs_basedOnMouthCenter_1pointAndMouthEyesCenter_filenames.mat')
        filenames = data['filenamePaths'].flatten()
        vidlens = data['videoLengthVec'].flatten()
        targets = (data['targetsPerVideoVec'].flatten())
        train_idxs = data['subjectsVec'].flatten() == 1
        val_idxs = data['subjectsVec'].flatten() == 2
        test_idxs = data['subjectsVec'].flatten() == 3
        train_vidlens = vidlens[train_idxs]
        val_vidlens = vidlens[val_idxs]
        test_vidlens = vidlens[test_idxs]
        train_targets = targets[train_idxs] - 1
        val_targets = targets[val_idxs] - 1
        test_targets = targets[test_idxs] - 1

        def prepare_filepaths(f):
            return os.path.join('../5words/data', str(f[0].replace('\\', '/')))

        vfunc = np.vectorize(prepare_filepaths)
        filenames = vfunc(filenames)
        training_files = filenames[train_idxs]
        val_files = filenames[val_idxs]
        test_files = filenames[test_idxs]
        datagen = gen_batch_from_file(training_files, train_targets, train_vidlens, 5551)

        for i in range(165):
            X_batch, y_batch, mask, idx = next(datagen)
            assert X_batch.shape == (30, 29, 5551)
            assert y_batch.shape == (30,)
            assert mask.shape == (30, 29)
            assert idx.shape == (30,)
        # remainder 4959 % 30
        remainder_batchsize = 4959 % 30
        X_batch, y_batch, mask, idx = next(datagen)
        assert X_batch.shape == (remainder_batchsize, 29, 5551)
        assert y_batch.shape == (remainder_batchsize,)
        assert mask.shape == (remainder_batchsize, 29)
        assert idx.shape == (remainder_batchsize,)

    def test_small_batch(self):
        """
        test when training data is smaller than batch size
        :return: batch of length equal to train data len
        """
        # load the test file, and preprocess the path separator and dimensions
        data = load_mat_file('../5words/data/'
                             'data5Words_mouthROIs_basedOnMouthCenter_1pointAndMouthEyesCenter_filenames.mat')
        filenames = data['filenamePaths'].flatten()
        vidlens = data['videoLengthVec'].flatten()
        targets = (data['targetsPerVideoVec'].flatten())
        train_idxs = data['subjectsVec'].flatten() == 1
        val_idxs = data['subjectsVec'].flatten() == 2
        test_idxs = data['subjectsVec'].flatten() == 3
        train_vidlens = vidlens[train_idxs]
        val_vidlens = vidlens[val_idxs]
        test_vidlens = vidlens[test_idxs]
        train_targets = targets[train_idxs] - 1
        val_targets = targets[val_idxs] - 1
        test_targets = targets[test_idxs] - 1

        def prepare_filepaths(f):
            return os.path.join('../5words/data', str(f[0].replace('\\', '/')))

        vfunc = np.vectorize(prepare_filepaths)
        filenames = vfunc(filenames)
        training_files = filenames[train_idxs]
        val_files = filenames[val_idxs]
        test_files = filenames[test_idxs]
        datagen = gen_batch_from_file(training_files[:10], train_targets[:10], train_vidlens[:10], 5551)
        X_batch, y_batch, mask, idx = next(datagen)

        assert X_batch.shape == (10, 29, 5551)
        assert y_batch.shape == (10,)
        assert mask.shape == (10, 29)
        assert idx.shape == (10,)


if __name__ == '__main__':
    unittest.main()