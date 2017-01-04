import numpy as np


def gen_lstm_seq_random(X, y, seqlen):
    """
    generate 1 random sequence of training data
    :param X: input
    :param y: target
    :param seqlen: lengths of video
    :return: x_train, y_target
    """
    # compute integral lengths of the video for fast offset access for data matrix
    integral_lens = [0]
    for i in range(1, len(seqlen)):
        integral_lens.append(integral_lens[i - 1] + seqlen[i - 1])

    while True:
        # permutate the video sequences for each batch
        # reset the permutation after all videos are used
        randomized = np.random.permutation(len(seqlen))
        for video_idx in randomized:
            start_seq = integral_lens[video_idx]
            end_seq = start_seq + seqlen[video_idx]
            seq_X = X[start_seq:end_seq, :]
            seq_y = y[start_seq:end_seq]
            yield seq_X, seq_y


def gen_lstm_batch_random(X, y, seqlen, batchsize=30, shuffle=True):
    """
    randomized data generator for training data
    creates an infinite loop of mini batches
    :param X: input
    :param y: target
    :param seqlen: lengths of video
    :param batchsize: number of videos per batch
    :return: x_train, y_target, input_mask, video idx used
    """
    # find the max len of all videos for creating the mask
    max_timesteps = np.max(seqlen)
    feature_len = X.shape[1]
    no_videos = len(seqlen)
    start_video = 0
    reset = False

    # compute integral lengths of the video for fast offset access for data matrix
    integral_lens = [0]
    for i in range(1, len(seqlen)):
        integral_lens.append(integral_lens[i-1] + seqlen[i - 1])

    # permutate the video sequences for each batch
    if shuffle:
        randomized = np.random.permutation(len(seqlen))
    else:
        randomized = range(len(seqlen))
    while True:
        end_video = start_video + batchsize
        if end_video >= no_videos:  # all videos iterated, reset
            batch_video_idxs = randomized[start_video:]
            # extract all the video lengths of the video idx
            reset = True
        else:
            batch_video_idxs = randomized[start_video:end_video]
        bsize = len(batch_video_idxs)
        X_batch = np.zeros((bsize, max_timesteps, feature_len), dtype=X.dtype)  # returned batch input
        y_batch = np.zeros((bsize,), dtype='uint8')
        mask = np.zeros((bsize, max_timesteps), dtype='uint8')

        # populate the batch X and batch y
        for i, idx in enumerate(batch_video_idxs):
            start = integral_lens[idx]
            l = seqlen[idx]
            end = start + l
            X_batch[i] = np.concatenate([X[start:end],
                                         np.zeros((max_timesteps - l, feature_len))])
            y_batch[i] = y[start]
            mask[i, :l] = 1  # set 1 for length of video
            mask[i, l:] = 0  # set 0 for rest of video
        if reset:
            # permutate the new video sequences for each batch
            if shuffle:
                randomized = np.random.permutation(len(seqlen))
            else:
                randomized = range(len(seqlen))
            start_video = 0
            reset = False
        else:
            start_video = end_video
        yield X_batch, y_batch, mask, batch_video_idxs


def gen_lstm_batch_seq(X, y, seqlen, batchsize=30):
    """
    generate the next batch of training data
    data generator, create an infinite loop of mini batch sizes
    :param X: input
    :param y: target
    :param seqlen: lengths of video
    :param batchsize: number of videos per batch
    :return: x_train, y_target, input_mask
    """
    # find the max len of all videos for creating the mask
    max_timesteps = np.max(seqlen)
    feature_len = X.shape[1]
    no_videos = len(seqlen)
    start_video = 0
    start_data = 0
    reset = False
    while True:
        end_video = start_video + batchsize
        if end_video > no_videos:
            # print('reached the end, restarting')
            slice_step = int(np.sum(seqlen[start_video:]))
            batch_vidlen = seqlen[start_video:]
            reset = True
        else:
            slice_step = int(np.sum(seqlen[start_video:end_video]))
            batch_vidlen = seqlen[start_video:end_video]
        # print('slice step: {}'.format(slice_step))
        # slice X, y according to the slice_step,
        # track the index of the data
        end_data = start_data + slice_step
        batch_sequence = X[start_data:end_data]
        batch_target = y[start_data:end_data]
        X_batch = np.zeros((batchsize, max_timesteps, feature_len), dtype='float32')  # returned batch input
        y_batch = np.zeros((batchsize,), dtype='uint8')
        mask = np.zeros((batchsize, max_timesteps), dtype='uint8')
        start = 0
        for i, l in enumerate(batch_vidlen):
            end = start + l
            X_batch[i] = np.concatenate([batch_sequence[start:end],
                                        np.zeros((max_timesteps-l, feature_len))])
            y_batch[i] = batch_target[start]
            mask[i, :l] = 1  # set 1 for length of video
            mask[i, l:] = 0  # set 0 for rest of video
            start = end
        if reset:
            start_video = 0
            start_data = 0
            reset = False
        else:
            start_video = end_video
            start_data = end_data
        yield X_batch, y_batch, mask


def compute_integral_len(lengths):
    # compute integral lengths of the video for fast offset access for data matrix
    integral_lens = [0]
    for i in range(1, len(lengths)):
        integral_lens.append(integral_lens[i - 1] + lengths[i - 1])
    return integral_lens


def gen_seq_batch_from_idx(data, idxs, seqlens, integral_lens, max_timesteps):
    feature_len = data.shape[-1]
    X_batch = np.zeros((len(idxs), max_timesteps, feature_len), dtype=data.dtype)

    for i, seq_id in enumerate(idxs):
        l = seqlens[seq_id]
        start = integral_lens[seq_id]
        end = start + l
        X_batch[i] = np.concatenate([data[start:end],
                                     np.zeros((max_timesteps - l, feature_len), dtype=data.dtype)])
    return X_batch


def sequence_batch_iterator(X, y, seqlen, batchsize=30):
    """
    generate the next batch of training data
    data generator, create an infinite loop of mini batch sizes
    :param X: input
    :param y: target
    :param seqlen: lengths of video
    :param batchsize: number of videos per batch
    :return: x_train, y_target, input_mask
    """
    # find the max len of all videos for creating the mask
    max_timesteps = np.max(seqlen)
    feature_len = X.shape[1]
    no_videos = len(seqlen)
    start_video = 0
    start_data = 0
    reset = False
    while True:
        end_video = start_video + batchsize
        if end_video > no_videos:
            # print('reached the end, restarting')
            slice_step = int(np.sum(seqlen[start_video:]))
            batch_vidlen = seqlen[start_video:]
            reset = True
        else:
            slice_step = int(np.sum(seqlen[start_video:end_video]))
            batch_vidlen = seqlen[start_video:end_video]
        # print('slice step: {}'.format(slice_step))
        # slice X, y according to the slice_step,
        # track the index of the data
        end_data = start_data + slice_step
        batch_sequence = X[start_data:end_data]
        batch_target = y[start_data:end_data]
        X_batch = np.zeros((batchsize, max_timesteps, feature_len), dtype='float32')  # returned batch input
        y_batch = np.zeros((batchsize,), dtype='uint8')
        mask = np.zeros((batchsize, max_timesteps), dtype='uint8')
        start = 0
        for i, l in enumerate(batch_vidlen):
            end = start + l
            X_batch[i] = np.concatenate([batch_sequence[start:end],
                                        np.zeros((max_timesteps-l, feature_len))])
            y_batch[i] = batch_target[start]
            mask[i, :l] = 1  # set 1 for length of video
            mask[i, l:] = 0  # set 0 for rest of video
            start = end
        if reset:
            start_video = 0
            start_data = 0
            reset = False
        else:
            start_video = end_video
            start_data = end_data
        yield X_batch, y_batch, mask


def batch_iterator(X, y, batchsize=128):
    """
    generate minibatches given an input X, target y
    :param X: input
    :param y: target
    :param batchsize: minibatch size
    :return: minibatch
    """
    start = 0
    reset = False
    randomized = np.random.permutation(len(X))
    while True:
        end = start + batchsize
        if end >= len(X):
            reset = True
            batch_idxs = randomized[start:]
        else:
            batch_idxs = randomized[start:end]

        batch_X = np.zeros((batchsize,) + X.shape[1:], dtype=X.dtype)
        batch_y = np.zeros((batchsize,) + y.shape[1:], dtype=y.dtype)

        for i, idx in enumerate(batch_idxs):
            batch_X[i] = X[idx]
            batch_y[i] = y[idx]
        if reset:
            randomized = np.random.permutation(len(X))
            start = 0
            reset = False
        else:
            start += end
        yield batch_X, batch_y


class SequenceBatchIterator(object):
    def __init__(self, X, y, seqlens, batchsize=30):
        """
        randomized data generator for sequence data
        creates an infinite loop of mini batches
        :param X: input
        :param y: target
        :param seqlens: lengths of video
        :param batchsize: number of sequence per batch
        :return:
        """
        self.X = X
        self.y = y
        self.seqlens = seqlens
        self.batchsize = batchsize
        # compute integral lengths of the video for fast offset access for data matrix
        self.integral_lens = compute_integral_len(seqlens)

    def next(self):
        return self.__next__()

    def __next__(self):
        """
        generate the next batch of sequences
        :return: batch, mask
        """
        # find the max len of all videos for creating the mask
        max_timesteps = np.max(self.seqlens)
        feature_len = self.X.shape[1]
        no_videos = len(self.seqlens)
        start_seq = 0
        reset = False

        # permutate the video sequences for each batch
        randomized = np.random.permutation(len(self.seqlens))
        while True:
            end_seq = start_seq + self.batchsize
            if end_seq > no_videos:
                batch_seq_idxs = randomized[start_seq:]
                # extract all the video lengths of the video idx
                reset = True
            else:
                batch_seq_idxs = randomized[start_seq:end_seq]
            # returned batch input
            X_batch = np.zeros((self.batchsize, max_timesteps, feature_len), dtype=self.X.dtype)
            y_batch = np.zeros((self.batchsize, ), dtype=self.y.dtype)
            mask = np.zeros((self.batchsize, max_timesteps), dtype='uint8')

            # populate the batch X and batch y
            for i, idx in enumerate(batch_seq_idxs):
                start = self.integral_lens[idx]
                l = self.seqlens[idx]
                end = start + l
                X_batch[i] = np.concatenate([self.X[start:end],
                                             np.zeros((max_timesteps - l, feature_len), dtype=self.X.dtype)])
                y_batch[i] = self.y[start]
                mask[i, :l] = 1  # set 1 for length of video
                mask[i, l:] = 0  # set 0 for rest of video
            if reset:
                # permutate the new video sequences for each batch
                randomized = np.random.permutation(len(self.seqlens))
                start_seq = 0
                reset = False
            else:
                start_seq = end_seq
            yield X_batch, y_batch, mask, batch_seq_idxs
