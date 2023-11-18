import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from scipy.stats import norm
from torch.distributions import Normal
import math
import numpy as np

from Eunomia.preferences import *
from Eunomia.additive_functions import *
from Eunomia.alternatives import *
from Eunomia.sampling import *

# Standard imports
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch import nn
from torch.distributions import constraints

import pyro.optim as optim

pyro.set_rng_seed(1)

def posterior_sampling_model(R, **kwargs):
    """
    Define a probabilistic model for posterior sampling using Pyro.

    Parameters
    ----------
    R : torch.Tensor
        Matrix of preference vectors, with shape (n, m).
    **kwargs : dict
        Keyword arguments to configure priors and hyperpriors.
        - sigma_w: Standard deviation of the normal prior for weights w. Default is 1e-2.
        - sigma_p: Rate parameter for the exponential hyperprior for sigma. Default is 1.

    Returns
    -------
    model : function
        The Pyro probabilistic model.
    """
    sigma_w = 1e-2 if "sigma_w" not in kwargs else kwargs["sigma_w"]
    sigma_p = float(1) if "sigma_p" not in kwargs else float(kwargs["sigma_p"])
    def model(R):
        num_features = R.size(1)
        w = pyro.sample("w", dist.Normal(torch.zeros(num_features), sigma_w*torch.ones(num_features)))
        sigma = pyro.sample("sigma", dist.Exponential(torch.tensor(sigma_p)))
        dot_product = torch.einsum('ij,j->i', R, w) / sigma
        cdf_values = 0.5 * (1 + torch.erf(dot_product / math.sqrt(2.0)))
        with pyro.plate("data", len(R)):
            pyro.sample("obs", pyro.distributions.Bernoulli(cdf_values), obs=torch.ones(R.shape[0]))
    return model

def sample_model(m, R, *args, **kwargs):
    """
    Perform MCMC sampling using NUTS on a given Pyro model.

    Parameters
    ----------
    m : function
        The Pyro probabilistic model.
    R : torch.Tensor
        Matrix of preference vectors, with shape (n, m).
    args : tuple
        The names of the parameters for which to return samples.
    **kwargs : dict
        Keyword arguments for MCMC sampling.
        - num_samples: Number of samples to draw. Default is 1000.
        - warmup_steps: Number of warmup steps. Default is 200.

    Returns
    -------
    tuple
        A tuple containing the posterior samples for the variables specified in args.
    """
    num_samples = 1000 if "num_samples" not in kwargs else kwargs["num_samples"]
    warmup_steps = 200 if "warmup_steps" not in kwargs else kwargs["warmup_steps"]
    nuts_kernel = NUTS(m)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(R)

    posterior_samples = mcmc.get_samples()
    if "return_diag" in kwargs and kwargs["return_diag"] == True:
        diag = mcmc.diagnostics()
        return  tuple([diag] + [posterior_samples[k] for k in args])
    return tuple([posterior_samples[k] for k in args])

def predict(R, w, sigma):
    """
    Generate predictions using the posterior samples.

    Parameters
    ----------
    R : torch.Tensor
        Matrix of preference vectors, with shape (n, m).
    w : torch.Tensor
        Sampled weights from the posterior, with shape (m,).
    sigma : float
        Sampled sigma value from the posterior.

    Returns
    -------
    torch.Tensor
        Predicted probabilities based on the Probit model.
    """
    dot_product = torch.einsum('ij,j->i', R, w) / sigma
    cdf_values = 0.5 * (1 + torch.erf(dot_product / math.sqrt(2.0)))
    return cdf_values

def get_acc_distribution(R, w_samples, sigma_samples=None):
    """
    Calculate the distribution of accuracies for the posterior samples.

    Parameters
    ----------
    R : torch.Tensor
        Matrix of preference vectors, with shape (n, m).
    w_samples : numpy array
        Posterior samples for the weights w.
    sigma_samples : numpy array, optional
        Posterior samples for sigma. Default is None, which assumes sigma=1 for all samples.

    Returns
    -------
    numpy array
        Array containing the accuracy for each set of posterior samples.
    """
    if sigma_samples is None:
        sigma_samples = np.ones(w_samples.shape[0])
    accs = []
    for w_sample, sigma in zip(w_samples, sigma_samples):
        pred = predict(R, w_sample, sigma)
        binary_predictions = (pred > 0.5).float()
        acc = binary_predictions.sum() / binary_predictions.shape[0]
        accs.append(np.array(acc))
    return np.array(accs)