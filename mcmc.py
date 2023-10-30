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
    Create a probabilistic model suitable for posterior sampling.
    
    Parameters:
    ------------
    R : tensor
        A matrix containing preference vectors.
    
    Optional Keyword Arguments:
    ---------------------------
    sigma_w : float
        Specifies the standard deviation of the Gaussian prior distribution for the weights `w`.
    sigma_p : float
        Specifies the rate parameter for the Exponential distribution used as a hyperprior for `sigma`.
    
    Returns:
    --------
    A Pyro probabilistic model.
    """
    sigma_w = 1e-2 if 'sigma_w' not in kwargs else kwargs['sigma_w']
    sigma_p = float(1) if 'sigma_p' not in kwargs else float(kwargs['sigma_p'])
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
    Perform MCMC sampling on a given Pyro model.
    
    Parameters:
    ------------
    m : Pyro model
        The model from which to generate samples.
    R : tensor
        A matrix containing preference vectors.
    args : tuple
        Named samples to return, containing the specific keys in which we are interested.
        
    Optional Keyword Arguments:
    ---------------------------
    num_samples : int
        The number of MCMC samples to generate.
    warmup_steps : int
        The number of warmup steps before generating samples.
        
    Returns:
    --------
    Posterior samples for the named variables.
    """
    num_samples = 1000 if 'num_samples' not in kwargs else kwargs['num_samples']
    warmup_steps = 200 if 'warmup_steps' not in kwargs else kwargs['warmup_steps']
    nuts_kernel = NUTS(m)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(R)
    posterior_samples = mcmc.get_samples()
    return tuple([posterior_samples[k] for k in args])
