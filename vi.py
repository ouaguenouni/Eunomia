from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.nn import PyroModule, PyroSample
from pyro import distributions as dist
import torch
from torch import nn
import pyro

class BayesianNet(PyroModule):
    """
    A Bayesian Neural Network using Pyro.

    Args:
    - in_features (int): The number of input features.
    - sigma_p (float, optional): The parameter of the exponential distribution from which sigma is sampled.
                                 Default is 1.0.
    - sigma_w (float, optional): Standard deviation of the normal priors on the weights. Default is 1.0.

    Attributes:
    - linear (PyroModule): The linear layer of the network.
    - sigma (float or PyroSample): The scale parameter for the likelihood.
    """
    def __init__(self, in_features, sigma_p=1.0, sigma_w=1.0):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, 1)
        self.linear.weight = PyroSample(dist.Normal(0., sigma_w).expand([1, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., sigma_w).expand([1]).to_event(1))
        if sigma_p is None:
            self.sigma = 1
        else:
            self.sigma = PyroSample(dist.Exponential(torch.tensor(sigma_p)).expand([1]).to_event(1))
            
    def forward(self, x, y=None):
        """Computes the forward pass of the network.
        
        Args:
        - x (Tensor): The input features.
        - y (Tensor, optional): The observed labels.
        
        Returns:
        - logits (Tensor): The output logits.
        """
        logits = self.linear(x).flatten() / self.sigma
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
        return logits


def fit_model_guide(model, guide, data):
    """Fits the model using SVI.
    
    Args:
    - model (callable): The Pyro model.
    - guide (callable): The Pyro guide.
    - data (Tensor): The data tensor.
    """
    adam = pyro.optim.Adam({"lr": 0.05})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for j in range(1000):
        loss = svi.step(data, torch.ones(data.shape[0]))
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

def sample_model_guide(model, guide, data):
    """Samples from the posterior distribution of the model parameters.
    
    Args:
    - model (callable): The Pyro model.
    - guide (callable): The Pyro guide.
    - data (Tensor): The data tensor.
    
    Returns:
    - weights (Tensor): Samples of the posterior weights.
    """
    guide.requires_grad_(False)
    predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("linear.weight", "obs", "_RETURN"))
    weights = predictive(data)["linear.weight"]
    weights = weights.reshape((weights.shape[0], weights.shape[-1]))
    return weights
