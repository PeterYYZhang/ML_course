# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

from labs.ex02.template.costs import compute_loss


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w_optimum = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_loss(y, tx, w_optimum, loss='MSE')
    return mse, w_optimum
