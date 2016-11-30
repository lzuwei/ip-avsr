import theano.tensor as tt


def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    """

    N, T, V = x.shape

    x_flat = x.reshape((N * T, V))
    y_flat = y.reshape((N * T,))
    mask_flat = mask.reshape((N * T,))
    total_frames = tt.sum(mask_flat)

    probs = tt.exp(x_flat - tt.max(x_flat, axis=1, keepdims=True))
    probs /= tt.sum(probs, axis=1, keepdims=True)
    # loss = -tt.sum(mask_flat * tt.log(probs[tt.arange(N * T), y_flat])) / N
    loss = -tt.sum(mask_flat * tt.log(probs[tt.arange(N * T), y_flat])) / total_frames

    return loss
