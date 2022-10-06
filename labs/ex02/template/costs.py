# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np




def compute_loss(y, tx, w, loss='MSE'):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    N = y.shape[0]
    error = y - np.dot(tx, w)
    if loss == 'MSE':
        loss = 1 / (2 * N) * np.dot(error.T, error)
        return loss
    elif loss == 'MAE':
        loss = 1/N * np.absolute(error)
        return loss
