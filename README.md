# weighted_nmf

## Weighted Non-negative Matrix Factorization

Heteroskedastic Matrix Factorization is great but many problems are bound to only non-negative weight values (e.g., astronomy light curves).
This module provides a straightforward implementation of weighted non-negative matrix factorization, or weighted NMF. Weighted NMF is a matrix factorization method similar to principal components analysis (PCA) except that it enforces a non-negativity constraint on the produced basis vectors (rather than orthogonality) and it is naturally suited to handle non-homogenous uncertainties and missing data. Our implementation provies a multiplicative weights iterative update technique which minimizes a Frobenius norm objective function.


Given data `X` and weights `S` of shape `(n_samples, n_features)`,

   ```m = nmf_factorization(X, S, options...)```

Returns a model object `m` from which you can inspect the vector factors, coefficients, reconstructed model, and original inputs, e.g.,
    
```
pylab.plot(m.vectors[0])
    
pylab.plot(m.model[0])
    
pylab.plot(m.data[0])
```

Note that missing data is simply the limit of weight=0.

For ease of understanding the implementation, an unrolled, non-vectorized version of weighted NMF is also implemented.

Ruhi Doshi (@rdoshi99), Spring 2022


## Installation
To install this package, you can run the following commands in your terminal.

First, clone into this repository.

```git clone https://github.com/rdoshi99/weighted_nmf.git```

And move inside the current directory.

```cd weighted_nmf```

Then run the `setup.py` file.

```python setup.py install```

That completes the installation process for this package and its dependencies.

#### Requirements
Only requires `numpy` (as of version v1.0)

## Background

The work is based on this paper by Tsalmantza & Hogg, 2012 (1),  which outlines the underlying mathematical basis. It is available here: https://arxiv.org/pdf/1201.3370.pdf. Additionally, the algorithm specified by Blanton & Roweis (2) provides clearer explicit update rules for each iteration (available at https://arxiv.org/pdf/astro-ph/0606170.pdf).


1) Tsalmantza, P. and Hogg, D., 2012. A DATA-DRIVEN MODEL FOR SPECTRA: FINDING DOUBLE REDSHIFTS IN THE SLOAN DIGITAL SKY SURVEY. The Astrophysical Journal, [online] 753(2), p.122. Available at: <https://arxiv.org/pdf/1201.3370.pdf>.

2) M. Blanton and S. Roweis., (2007). K-corrections and filter transformations in the ultraviolet, optical, and near infrared. The Astronomical Journal 133, [online]734-754(2). Available at https://arxiv.org/abs/astro-ph/0606170.

#### Acknowledgement
Thank you to Stephen Bailey (@sbailey) for the significant guidance and support in improving the original algorithms and iterating throughout the research process.
