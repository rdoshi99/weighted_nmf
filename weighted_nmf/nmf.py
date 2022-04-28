#!/usr/bin/env python

"""
Nonnegative Matrix Factorization
Heteroskedastic Matrix Factorization is great but many problems are bound to only nonnegative weight values. This module provides Nonnegative Matrix Factorization, NMF, an iterative method for factoring matrices.
Given data[nobs, nvar] and weights[nobs, nvar],
    m = nmf(data, weights, options...)
Returns an NMF object m from which you can inspect the vector factors, coefficients, reconstructed model, and original inputs, e.g.,
    pylab.plot( m.vectors[0] )
    pylab.plot( m.coeff[0] )
    pylab.plot( m.model[0] )
    pylab.plot( m.data[0] )

Note that missing data is simply the limit of weight=0.

For ease of understanding the implementation, an unrolled, non-vectorized version of NMF is also implemented.

Ruhi Doshi, Spring 2022
"""

import numpy as np
from tqdm import tqdm


class model(object):
    """
    `model` is a wrapper class to save the original data, factored vectors, and coefficients
    
    Returned by nmf(...) function. It stores the following variables:
      Inputs: 
        - data   [nobs, nvar]
        - weights[nobs, nvar]
        - vectors[nvar, nvec]
        - coeffs [nvec, nobs]
      
      Computed:
        - model  [nobs, nvar] - reconstruction of data using vectors,coeffs
    
    """
    def __init__(self, data, weights, vectors, coeffs, track_losses=False, loss_arr=[]):
        """
        Returns a `model` object with the given vectors, data, and weights
        
        Dimensions:
          - vectors[nvec, nvar]
          - coeffs [nobs, nvec]
          - data   [nobs, nvar]
          - weights[nobs, nvar]
        
        Options:
        - track_losses : a boolean indicating if losses should be returned as part of the model attributes
        - loss_arr : the losses to be returned when track_losses = 1
        """
        self.vectors = vectors
        self.coeffs = coeffs
        
        self.data = data
        self.weights = weights

        self.nobs = data.shape[0]
        self.nvar = data.shape[1]
        self.nvec = vectors.shape[0]
        
        self.model = np.dot(self.vectors, self.coeffs).T
        
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


def _nmf_iterate(X, S, W, H, num_iters=100, verbose=False):
    """
    Runs the main NMF algorithm by iterating over W and H updates
    
    Args:
    - X : numpy matrix of shape [nvar, nobs] representing the data matrix transposed
    - S : numpy matrix of shape [nvar, nobs] representing the inverse variances matrix transposed
    - W : numpy matrix of shape [nvar, nvec] representing the vector factors
    - H : numpy matrix of shape [nvec, nobs] representing the fitted coefficients
    Options:
    - num_iters : (default=100) integer number of NMF iterations (both W and H will be updated num_iters times)
    - verbose : (default=False) prints out the loss function at each iteration if True
    
    Returns updated W and H matrices, numpy array of loss function at each iteration
    """
    print('Running NMF algorithm')
    # iterate over W, H updates
    losses = np.zeros((num_iters,))    
    for i in tqdm(range(num_iters)):
        
        W = _update_W(X, S, W, H)
        H = _update_H(X, S, W, H)
        
        losses[i] = objective(X, S, W, H)
        if verbose:
            print('iter {}: {}'.format(i, losses[i]))
    
    print(f'Completed {i} iterations, final cost: {losses[-1]}')
    return W, H, losses


def run(data, weights, nvec=10, num_iters=100, verbose=False, track_losses=False):
    """
    Main function used to run the module and extract the stored values
    
    Args:
    - data : numpy data matrix of shape [nobs, nvar]
    - weights : numpy inverse variances matrix of shape [nobs, nvar]
    Options:
    - nvec : (default=10) integer number of vectors to factor data into
    - num_iters : (default=100) integer number of iterations to run NMF
    - verbose : (default=False) prints out the loss function at each iteration if True
    - track_losses : (default=False) stores the losses after each update (W, H) in the model object
    
    Returns a model object of the NMF-fitted model
    """
    nobs, nvar = data.shape
    X, S = data.T, weights.T # transpose the inputs to match the shapes for the algorithm
    
    # initialize random W, H
    W = np.random.rand(nvar, nvec)
    H = np.random.rand(nvec, nobs)
    
    # run NMF algorithm
    W_new, H_new, losses = _nmf_iterate(X, S, W, H, num_iters=num_iters, verbose=verbose)
    m = model(data, weights, W_new, H_new, track_losses=track_losses, loss_arr=losses) # construct the model object to return
    
    return m

