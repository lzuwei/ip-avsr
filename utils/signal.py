import numpy as np
import theano
import theano.tensor as T
from lasagne.utils import unroll_scan


def delta_theta(theta, curr_delta, t, THETA, Y):
    """
    compute a delta theta component at delta time step t
    :param theta: current time step theta component
    :param curr_delta: current accumulated delta_t
    :param t: current delta_t to be computed
    :param THETA: window size
    :param Y: input sequence
    :return: delta theta component for time step t
    """
    # accumulator is shaped (1, no_features), transpose to perform column wise element operations
    temp = curr_delta.T
    d_theta = theta * (Y[:, THETA + t + theta] - Y[:, THETA + t - theta]) / (2 * theta * theta)
    temp += d_theta
    temp = temp.astype('float32')
    curr_delta = temp.T
    return curr_delta


def delta_t(t, THETA, Y):
    """
    compute delta at time step t
    :param t: time step
    :param THETA: window size
    :param Y: sequence in shape (number_of_features, time_step)
    :return: delta coefficient at time step t
    """
    theta = T.arange(1, THETA + 1, dtype='int32')
    results, _ = theano.scan(delta_theta, outputs_info=T.zeros_like(Y),
                             sequences=theta, non_sequences=[t, THETA, Y])
    # only interested in the final results, discard the intermediate values
    final_results = results[-1]
    return final_results


def delta_coeff(A, theta):
    """
    compute delta coefficients given a sequence.
    :param A: input sequence in shape (time_step, number_of_features)
    :param theta: window size
    :return: delta coefficients for the input sequence
    """
    # transpose and repeat
    X = A.T
    Y = T.concatenate([T.extra_ops.repeat(X[:, 0], theta).reshape((X.shape[0], theta)),
                       X, T.extra_ops.repeat(X[:, -1], theta).reshape((X.shape[0], theta))], axis=1)
    delta, _ = theano.scan(delta_t, sequences=[T.arange(0, X.shape[1], dtype='int32')], non_sequences=[theta, Y])
    # transpose the results back to shape (time_step, number_of_features)
    delta = delta[:, :, -1].reshape(A.shape)
    return delta


def append_delta_coeff(A, theta):
    """
    append delta + acceleration coefficients given a sequence.
    :param A: input sequence in shape (time_step, number_of_features)
    :param theta: window size
    :return: delta + acceleration coefficients for the input sequence
    """
    # transpose and repeat
    X = A.T
    Y = T.concatenate([T.extra_ops.repeat(X[:, 0], theta).reshape((X.shape[0], theta)),
                       X, T.extra_ops.repeat(X[:, -1], theta).reshape((X.shape[0], theta))], axis=1)
    delta, _ = theano.scan(delta_t, sequences=[T.arange(0, X.shape[1], dtype='int32')], non_sequences=[theta, Y])
    # transpose the results back to shape (time_step, number_of_features)
    delta = delta[:, :, -1].reshape(A.shape)

    X = delta.T
    Y = T.concatenate([T.extra_ops.repeat(X[:, 0], theta).reshape((X.shape[0], theta)),
                       X, T.extra_ops.repeat(X[:, -1], theta).reshape((X.shape[0], theta))], axis=1)
    acc, _ = theano.scan(delta_t, sequences=[T.arange(0, X.shape[1], dtype='int32')], non_sequences=[theta, Y])
    acc = acc[:, :, -1].reshape(A.shape)
    res = T.concatenate([A, delta, acc], axis=1)
    return res


def main():
    """
    test runner, computes delta for an array of sequences
    :return: None
    """
    A = T.tensor3('A', dtype='float32')
    theta = T.iscalar('theta')

    # compute delta coefficients for multiple sequences
    results, updates = theano.scan(append_delta_coeff, sequences=A, non_sequences=theta)
    compute_deltas = theano.function([A, theta], outputs=results, updates=updates)

    seqs = np.array([[[1, 2, 3, 4, 5],
                      [10, 12, 13, 14, 15],
                      [300, 1, 23, 56, 22]],
                     [[1, 1, 1, 1, 1],
                      [1, 1, 100, 1, 1],
                      [1, 1, 1, 1, 1]]], dtype='float32')
    res = compute_deltas(seqs, 1)
    print(res)

if __name__ == '__main__':
    main()
