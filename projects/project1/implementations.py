# -*- coding: utf-8 -*-
import numpy as np
import sys

eps = sys.float_info.epsilon # a small number for numerical stability
loss_threshold = 1e-6
weight_threshold = 1e-6
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient Descent with MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: numpy arrays of shape (2, ), final weight
        loss: loss value after the optimization
    """
    # initial_w = kwargs.get('initial_w')
    # max_iters = kwargs.get('max_iters')
    # gamma = kwargs.get('gamma')

    def compute_loss(y, tx, w):
        data_num = y.shape[0]
        e_mat = y - np.dot(tx, w)
        loss = 0.5 * np.dot(e_mat.T, e_mat) / data_num
        return loss[0][0]

    def compute_gradient(y, tx, w):
        data_num = y.shape[0]
        e_mat = y - np.dot(tx, w)
        grad = -np.dot(tx.T, e_mat) / data_num
        return grad
    
    w = initial_w.copy()
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):
        prev_loss = loss
        grad_w = compute_gradient(y, tx, w)
        w = w - gamma * grad_w
    
        loss = compute_loss(y, tx, w)
        if(abs(prev_loss - loss) < loss_threshold):
            break
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent with MSE, batch size is 1.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: numpy arrays of shape (2, ), final weight
        loss: loss value after the optimization
    """
    # initial_w = kwargs.get('initial_w')
    # max_iters = kwargs.get('max_iters')
    # gamma = kwargs.get('gamma')

    def compute_loss(y, tx, w):
        data_num = y.shape[0]
        e_mat = y - np.dot(tx, w)
        loss = 0.5 * np.dot(e_mat.T, e_mat) / data_num
        return loss[0][0]

    def compute_gradient(y, tx, w):
        data_num = y.shape[0]
        e_mat = y - np.dot(tx, w)
        grad = -np.dot(tx.T, e_mat) / data_num
        return grad
    
    w = initial_w.copy()
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):
        
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            prev_loss = loss
            grad_w = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad_w
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            if(abs(prev_loss - loss) < loss_threshold):
                break
    
    return w, loss

def least_squares(y, tx):
    """Normal Equation solving for LR.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        
    Returns:
        w: numpy arrays of shape (2, ), final weight
        loss: loss value after the optimization
    """
    def compute_loss(y, tx, w):
        data_num = y.shape[0]
        e_mat = y - np.dot(tx, w)
        loss = 0.5 * np.dot(e_mat.T, e_mat) / data_num
        if(len(loss.shape) == 0):
            return loss
        else:
            return loss[0][0]
    
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Normal equation solving with regularization term for ridge LR.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        
    Returns:
        w: numpy arrays of shape (2, ), final weight
        loss: loss value after the optimization
    """
    # lambda_ = kwargs.get('lambda_')

    def compute_loss(y, tx, w):
        data_num = y.shape[0]
        e_mat = y - np.dot(tx, w)
        loss = 0.5 * np.dot(e_mat.T, e_mat) / data_num
        if(len(loss.shape) == 0):
            return loss
        else:
            return loss[0][0]
    
    D = tx.shape[1]
    N = tx.shape[0]
    w = np.linalg.solve(np.dot(tx.T, tx) + 2*N*lambda_ * np.eye(D), np.dot(tx.T, y))
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression for classification.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: numpy arrays of shape (2, ), final weight
        loss: loss value after the optimization
    """
    # initial_w = kwargs.get('initial_w')
    # max_iters = kwargs.get('max_iters')
    # gamma = kwargs.get('gamma')
    sgd = False

    def compute_loss(y, tx, w):
        data_num = y.shape[0]
        g = sigmoid(np.dot(tx, w))
        loss = -np.dot(y.T, np.log(g)) - np.dot((1 - y).T, np.log(1 - g))
        if(len(loss.shape) == 0):
            return loss / data_num
        else:
            return loss[0][0] / data_num

    def compute_gradient(y, tx, w):
        data_num = y.shape[0]
        g = sigmoid(np.dot(tx, w))
        grad = np.dot(tx.T, g - y) / data_num
        return grad
    
    w = initial_w.copy()
    loss = compute_loss(y, tx, w)

    if(sgd == True):
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
                loss = compute_loss(minibatch_y, minibatch_tx, w)
                grad_w = compute_gradient(minibatch_y, minibatch_tx, w)
                w = w - gamma * grad_w
    elif(sgd == False):
        for n_iter in range(max_iters):
            prev_loss = loss
            prev_w = w.copy()
            grad_w = compute_gradient(y, tx, w)
            w = w - gamma * grad_w
            loss = compute_loss(y, tx, w)
            if(np.max(abs(prev_w - w)) < weight_threshold):
                break
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression for classification.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: numpy arrays of shape (2, ), final weight
        loss: loss value after the optimization
    """
    # initial_w = kwargs.get('initial_w')
    # max_iters = kwargs.get('max_iters')
    # gamma = kwargs.get('gamma')
    # sgd = kwargs.get('sgd', False)
    # lambda_ = kwargs.get('lambda_')
    sgd=False

    def compute_loss(y, tx, w):
        data_num = y.shape[0]
        g = sigmoid(np.dot(tx, w))
        loss = -np.dot(y.T, np.log(g)) - np.dot((1 - y).T, np.log(1 - g))
        if(len(loss.shape) == 0):
            return loss / data_num
        else:
            return loss[0][0] / data_num

    def compute_gradient(y, tx, w, lambda_):
        data_num = y.shape[0]
        g = sigmoid(np.dot(tx, w))
        grad = np.dot(tx.T, g - y)  / data_num + 2.0 * lambda_ * w
        return grad
    
    w = initial_w.copy()
    loss = compute_loss(y, tx, w)

    if(sgd == True):
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
                loss = compute_loss(minibatch_y, minibatch_tx, w)
                grad_w = compute_gradient(minibatch_y, minibatch_tx, w, lambda_)
                w = w - gamma * grad_w
    elif(sgd == False):
        for n_iter in range(max_iters):
            prev_loss = loss
            prev_w = w.copy()
            grad_w = compute_gradient(y, tx, w, lambda_)
            w = w - gamma * grad_w
            loss = compute_loss(y, tx, w)
            if(np.max(abs(prev_w - w)) < weight_threshold):
                break
    
    return w, loss


def compute_statistics_all(y, tx, w, func_type = 'linear', return_result = False):
    """Compute Loss and Accuracy after training.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy arrays of shape (2, ), final weight
        func_type: type of model used, linear for linear regression and logistic for logistic regression.
        
    Returns:
        loss: scalar
        acc: classification accuracy
    """
    data_num = tx.shape[0]
    
    if(func_type == 'linear'):
        model_output = np.dot(tx, w)
        e_mat = y - model_output
        loss = 0.5 * np.dot(e_mat.T, e_mat) / data_num
        model_output[np.where(model_output > 0)] = 1
        model_output[np.where(model_output <= 0)] = -1
    elif(func_type == 'logistic'):
        model_output = sigmoid(np.dot(tx, w))
        loss = -np.dot(y, np.log(model_output + eps)) - np.dot((1 - y).T, np.log(1 - model_output + eps))
        model_output[np.where(model_output > 0.5)] = 1
        model_output[np.where(model_output <= 0.5)] = 0

    if(return_result):
        return model_output

    acc = np.sum(model_output == y) / data_num
    return loss, acc
    



