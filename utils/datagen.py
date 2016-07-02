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


def gen_lstm_batch_random(X, y, seqlen, batchsize=30):
    """
    randomized data generator for training data
    creates an infinite loop of mini batches
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
    reset = False

    # compute integral lengths of the video for fast offset access for data matrix
    integral_lens = [0]
    for i in range(1, len(seqlen)):
        integral_lens.append(integral_lens[i-1] + seqlen[i - 1])

    # permutate the video sequences for each batch
    randomized = np.random.permutation(len(seqlen))
    while True:
        end_video = start_video + batchsize
        if end_video > no_videos:
            batch_video_idxs = randomized[start_video:]
            # extract all the video lengths of the video idx
            reset = True
        else:
            batch_video_idxs = randomized[start_video:end_video]
        X_batch = np.zeros((batchsize, max_timesteps, feature_len), dtype='float32')  # returned batch input
        y_batch = np.zeros((batchsize,), dtype='uint8')
        mask = np.zeros((batchsize, max_timesteps), dtype='uint8')

        # populate the batch X and batch y
        for i, idx in enumerate(batch_video_idxs):
            start = integral_lens[idx]
            l = seqlen[idx]
            end = start + l
            X_batch[i] = np.concatenate([X[start:end],
                                         np.zeros((max_timesteps - l, feature_len))])
            y_batch[i] = y[start] - 1
            mask[i, :l] = 1  # set 1 for length of video
            mask[i, l:] = 0  # set 0 for rest of video
        if reset:
            # permutate the new video sequences for each batch
            randomized = np.random.permutation(len(seqlen))
            start_video = 0
            reset = False
        else:
            start_video = end_video
        yield X_batch, y_batch, mask


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
            y_batch[i] = batch_target[start] - 1  # get the letter for this video
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
