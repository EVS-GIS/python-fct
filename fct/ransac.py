# coding: utf-8

"""
Random Sample Consensus (RANSAC) Model Estimation
[1] https://fr.wikipedia.org/wiki/RANSAC
[2] https://scipy-cookbook.readthedocs.io/items/RANSAC.html
"""

import numpy as np

class LinearModel(object):

    def __init__(self, xcols, ycols):

        self.xcols = xcols
        self.ycols = ycols

    def fit(self, data):

        X = data[:, self.xcols]
        Y = data[:, self.ycols]
        A, sqerror, rank, eigen_values = np.linalg.lstsq(X, Y, rcond=None)
        return A, sqerror

    def residuals(self, parameters, data):

        X = data[:, self.xcols]
        Y = data[:, self.ycols]
        return Y - np.dot(X, parameters)

def random_partition(k, n):

    indexes = np.arange(n)
    np.random.shuffle(indexes)
    return indexes[:k], indexes[k:]

def ransac(x, model, n, k, t, d):
    """
    Fit `model` on data `x` using RANSAC procedure

    [1] https://fr.wikipedia.org/wiki/RANSAC
    [2] https://scipy-cookbook.readthedocs.io/items/RANSAC.html

    Parameters
    ----------

    x: np-array
        input data
    model: object
        model to train
    n: int
        samples to use for each iteration
    k: int
        number of iterations
    t: float
        error threshold
    d: int
        minimum number of correctly modeled observations
        needed to accept iteration

    Returns
    -------

    parameters: tuple
    mse: float
        mean square error of best model
    nfitted: int
        number of observations
        used to fit the best model
    """

    bestfit = None
    besterror = float('inf')
    bestlen = float('-inf')

    for _ in range(k):

        train_idx, test_idx = random_partition(n, x.shape[0])
        train = np.take(x, train_idx, axis=0)
        test = np.take(x, test_idx, axis=0)
        trained, sqerror = model.fit(train)
        residuals = np.square(model.residuals(trained, test))
        accepted_idx = test_idx[residuals[:, 0] < np.square(t)]
        accepted = np.take(x, accepted_idx, axis=0)

        if len(accepted) > d:

            data = np.concatenate((train, accepted))
            fit, sqerror = model.fit(data)
            mse = sqerror / data.shape[0]
            if mse < besterror:
                bestfit = fit
                besterror = mse
                bestlen = data.shape[0]

        elif bestfit is None:
            if len(accepted) > bestlen:
                bestlen = len(accepted)

    if bestfit is None:

        if bestlen > float('-inf'):

            raise RuntimeError('Could not meet acceptance criteria, ' + \
                'required length = %d ' % d + \
                'max. consensus length = %d' % bestlen)

        else:

            raise RuntimeError('Could not meet acceptance criteria')

    return bestfit, besterror, bestlen
