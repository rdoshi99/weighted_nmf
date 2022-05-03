#!/usr/bin/env python

"""
Weighted Non-negative Matrix Factorization

Heteroskedastic Matrix Factorization is great but many problems are bound to only nonnegative weight values. 
This module provides Nonnegative Matrix Factorization, NMF, an iterative method for factoring matrices.

Given data[nobs, nvar] and weights[nobs, nvar],
    m = run(data, weights, options...)

Returns a model object m from which you can inspect the vector factors, reconstructed model, and original inputs, e.g.,
    pylab.plot(m.vectors[0])
    pylab.plot(m.model[0])
    pylab.plot(m.data[0])

Note that missing data is simply the limit of weight=0.

For ease of understanding the implementation, an unrolled, non-vectorized version of NMF is also implemented.

Ruhi Doshi (@rdoshi99), Spring 2022
"""

import numpy as np


class model(object):
    """
    `model` is a wrapper class to save the original data, factored vectors, and coefficients
    
    Returned by nmf(...) function. It stores the following variables:
      Inputs: 
        - data   (n_samples, n_features)
        - weights(n_samples, n_features)
        - vectors(n_features, n_components)
        - coeffs (n_components, n_samples)
      
      Computed:
        - model  (n_samples, n_features) - reconstruction of data using vectors,coeffs
    
    """
    def __init__(self, data, weights, vectors, coeffs, track_losses=False, loss_arr=[]):
        """
        Returns a `model` object with the given vectors, data, and weights
        
        Dimensions:
        - data   (n_samples, n_features)
        - weights(n_samples, n_features)
        - vectors(n_features, n_components)
        - coeffs (n_components, n_samples)
        
        Options:
        - track_losses : a boolean indicating if losses should be returned as part of the model attributes
        - loss_arr : the losses to be returned when track_losses = 1
        """
        self.vectors = vectors
        self.coeffs = coeffs
        
        self.data = data
        self.weights = weights

        self.n_samples = data.shape[0]
        self.n_features = data.shape[1]
        self.n_components = vectors.shape[0]
        
        self.model = np.matmul(self.vectors.T, self.coeffs.T).T
        
        if track_losses:
            self.losses = loss_arr
        
        
def objective(X, S, W, H):
    """
    Computes sum((model-data)^2 / weights)
    
    Args:
    - X : numpy matrix of shape [nvar, nobs] representing the data matrix transposed
    - S : numpy matrix of shape [nvar, nobs] representing the inverse variances matrix transposed
    - W : numpy matrix of shape [nvar, nvec] representing the vector factors
    - H : numpy matrix of shape [nvec, nobs] representing the fitted coefficients
    
    Returns a scalar representing the weighted chi-squared loss function for the given model
    """
    V = W@H
    return np.sum((X-V)**2*S)
    
    
def _update_W_nonvectorized(X, S, W, H):
    """
    This is a non-vectorized version of _update_W. See that for the interface.    
    """
    # one iteration of W
    W_new = np.zeros(W.shape)
    for k in range(W.shape[0]):
        X_k = X[k]
        W_k = W[k]
        S_k = S[k]
        W_k_new = []
        for i in range(W.shape[1]):            
            old_val = W_k[i]
            H_i = H[i]
            numer = np.sum(X_k * H_i * S_k)
            denom = np.sum(W_k @ H * H_i * S_k)
            new_val = old_val*numer/denom
            # W_k_new.append(new_val)
            W_new[k][i] = new_val
    return W_new


def _update_H_nonvectorized(X, S, W, H):
    """
    This is a non-vectorized version of _update_H. See that for the interface.
    """
    # one iteration of H
    H_new = np.zeros(H.shape)
    for i in range(H.shape[0]):
        H_i = H[i]
        Wt_i = W[:,i]
        for n in range(H.shape[1]):
            St_n = S[:,n]
            old_val = H_i[n]
            numer = np.sum(Wt_i * X[:,n] * St_n)
            denom = np.sum(H[:,n] @ W.T * Wt_i * St_n)
            new_val = old_val*numer/denom
            H_new[i][n] = new_val
    return H_new

    
def _update_W(X, S, W, H):
    """
    Performs an update of W in 1 iteration of the NMF algorithm
    
    Args:
    - X : numpy matrix of shape [nvar, nobs] representing the data matrix transposed
    - S : numpy matrix of shape [nvar, nobs] representing the inverse variances matrix transposed
    - W : numpy matrix of shape [nvar, nvec] representing the vector factors
    - H : numpy matrix of shape [nvec, nobs] representing the fitted coefficients
    
    Returns an updated W numpy matrix of shape [nvar, nvec]
    """
    # one iteration of W
    numerator = (X*S) @ H.T
    denominator = (W@H * S) @ H.T

    return W*numerator/denominator


def _update_H(X, S, W, H):
    """
    Performs an update of H in 1 iteration of the NMF algorithm
    
    Args:
    - X : numpy matrix of shape [nvar, nobs] representing the data matrix transposed
    - S : numpy matrix of shape [nvar, nobs] representing the inverse variances matrix transposed
    - W : numpy matrix of shape [nvar, nvec] representing the vector factors
    - H : numpy matrix of shape [nvec, nobs] representing the fitted coefficients
    
    Returns an updated H numpy matrix of shape [nvec, nobs]
    """
    # one iteration of H
    numerator = W.T @ (X * S)
    denominator = ((W@H * S).T @ W).T

    return H*numerator/denominator


def _nmf_iterate(X, S, W, H, tol=1e-4, max_iters=100, verbose=False):
    """
    Runs the main NMF algorithm by iterating over W and H updates
    
    Args:
    - X : numpy matrix of shape (n_features, n_samples) representing the data matrix transposed
    - S : numpy matrix of shape (n_features, n_samples) representing the inverse variances matrix transposed
    - W : numpy matrix of shape (n_features, n_components) representing the vector factors transposed
    - H : numpy matrix of shape (n_components, n_samples) representing the fitted coefficients transposed
    Options:
    - tol : (default=1e-4) float tolerance of the stopping condition
    - max_iters : (default=100) integer maximum number of NMF iterations (both W and H will be updated num_iters times)
    - verbose : (default=False) prints out the loss function at each iteration if True
    
    Returns updated W and H matrices, numpy array of loss function at each iteration
    """
    print('Running NMF algorithm')
    
    # iterate over W, H updates
    losses = np.zeros((max_iters,))    
    for i in range(max_iters):
        
        W = _update_W(X, S, W, H)
        H = _update_H(X, S, W, H)
        
        losses[i] = objective(X, S, W, H)
        if verbose:
            print('iter {}: {}'.format(i, losses[i]))
        if losses[i] <= tol:
            print('Stopping condition reached early. Terminating iterations.')
            break
    
    print(f'Completed {i} iterations, final cost: {losses[-1]}')
    return W.T, H.T, i, losses


def weighted_nmf_factorization(X, S=None, W=None, H=None, n_components=10, random_state=0, tol=1e-4, max_iters=100, verbose=False, track_losses=False):
    """
    Main function used to run the module and extract the stored values
    
    Args:
    - X : array-like (n_samples, n_features) representing the data matrix
    Options:
    - S : (default=None) array-like (n_samples, n_features) representing inverse variances matrix, initialized to uniform weights if None passed
    - W : (default=None) array-like (n_components, n_features) representing vector solution matrix, initialized randomly if None passed
    - H : (default=None) array-like (n_samples, n_components) representing coefficients solution matrix, initialized randomly if None passed
    - n_components : (default=10) integer number of component vectors to factorize data into
    - tol : (default=1e-4) float tolerance of the stopping condition
    - max_iters : (default=100) integer maximum number of iterations to run NMF
    - verbose : (default=False) prints out the loss function at each iteration if True
    - track_losses : (default=False) stores the losses after each update (W, H) in the model object
    
    Returns a model object of the NMF-fitted model
    """
    X = np.array(X)
    n_samples, n_features = X.shape
    
    # initialize weights, coeffs, and vectors
    np.random.seed(random_state)
    if not isinstance(S, np.ndarray):
        S = np.ones(X.shape)
    if not isinstance(W, np.ndarray):
        W = np.random.rand(n_components, n_features)
    if not isinstance(H, np.ndarray):
        H = np.random.rand(n_samples, n_components)
    # enforce array-like constraint
    S, W, H = np.array(S), np.array(W), np.array(H)
        
    # run NMF algorithm
    W_new, H_new, n_iters, losses = _nmf_iterate(X.T, S.T, W.T, H.T, tol=tol, max_iters=max_iters, verbose=verbose) # transpose inputs to match the shapes for the algorithm
    m = model(X, S, W_new, H_new, track_losses=track_losses, loss_arr=losses) # construct the model object to return
    
    return m
