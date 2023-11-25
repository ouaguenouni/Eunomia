import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Eunomia.preferences import *
from Eunomia.additive_functions import *
from Eunomia.alternatives import *
from Eunomia.sampling import *
from Eunomia.mcmc import *
#from Eunomia.degree import *
from Eunomia.experiments import *
import torch
import torch.nn as nn
import torch.optim as optim


class PhiFunction(nn.Module):
    """
    A PyTorch module to implement the Phi function for the Symmetric Skew Bilinear (SSB) model.

    This module calculates the interaction utility between pairs of attributes in alternatives X and Y,
    based on the formula:
    
    f(X, Y) = [ \sum_{A \in \theta} w_A*I_X(A) - w_B*I_Y(B) ] + \sum_{A \in \theta} \sum_{B \in \theta} M[A, B, I_X(A), I_Y(B)]
    
    where M[A, B, I_X(A), I_Y(B)] is the utility of having A in X and B in Y based on their presence (1) or absence (0).

    Attributes:
        theta (list): A list of attributes or interactions considered by the model.
        phi_values (dict): A dictionary to store the parameters for each unique pair of attributes and their presence/absence states.

    Methods:
        forward(sparse_X, sparse_Y): Computes the sum of interaction utilities for given alternatives X and Y.
        logprior(): Returns the sum of the absolute values of the phi parameters, used for regularization.
    """

    def __init__(self, theta):
        """
        Initializes the PhiFunction module.

        Parameters:
            theta (list): A list of attributes or interactions considered by the model.
        """
        super(PhiFunction, self).__init__()
        n = len(theta)
        self.theta = theta
        self.phi_values = {}

        # Iterate over all pairs of attributes in theta to initialize parameters
        for i_A in range(n):
            A = theta[i_A]
            for i_B in range(i_A + 1, n):
                B = theta[i_B]
                # Initialize parameters for each state combination
                for X in [0, 1]:
                    for Y in [0, 1]:
                        self.phi_values[(tuple(A), tuple(B), X, Y)] = nn.Parameter(torch.tensor(0.0))

    def forward(self, sparse_X, sparse_Y):
        """
        Computes the sum of phi values for given sparse representations of alternatives X and Y.

        Parameters:
            sparse_X (list): A sparse representation of alternative X.
            sparse_Y (list): A sparse representation of alternative Y.

        Returns:
            float: The sum of interaction utilities for X and Y, computed as:
            
            \sum_{A \in \theta} \sum_{B \in \theta} M[A, B, I_X(A), I_Y(B)]
            
            where M[A, B, I_X(A), I_Y(B)] represents the utility parameter for the presence/absence states of A in X and B in Y.
        """
        phi_sum = 0
        for i_A in range(len(self.theta)):
            A = self.theta[i_A]
            for i_B in range(i_A + 1, len(self.theta)):
                B = self.theta[i_B]
                v = self.phi_values[(tuple(A), tuple(B),
                                     int(all(i in sparse_X for i in A)),
                                     int(all(i in sparse_Y for i in B)))]
                phi_sum += v
        return phi_sum

    def logprior(self):
        """
        Computes a regularization term as the sum of the absolute values of the phi parameters.

        Returns:
            float: The regularization term, computed as:
            
            \sum_{k} |M_k|
            
            where M_k represents each utility parameter in the model.
        """
        phis = [torch.abs(self.phi_values[k]) for k in self.phi_values]
        return sum(phis)


class SSBModel(nn.Module):
    """
    A PyTorch module implementing the Symmetric Skew Bilinear (SSB) model.

    This model combines additive and non-transitive interactions to assess preferences between alternatives.

    Attributes:
        theta (list): A list of attributes or interactions considered by the model.
        weights (torch.nn.Parameter): A parameter vector for the additive part of the model.
        phi_function (PhiFunction): An instance of PhiFunction to handle non-transitive interactions.

    Methods:
        logprior(lambda_1, lambda_2): Computes the regularization term for the model.
        likelihood(R): Computes the likelihood of the preference data.
        forward(sparse_X, sparse_Y): Computes the utility score for a pair of alternatives.
        accuracy(s): Computes the accuracy of the model on a dataset.
        display_model(): Displays the model parameters.
    """

    def __init__(self, theta):
        """
        Initializes the SSBModel module.

        Parameters:
            theta (list): A list of attributes or interactions considered by the model.
        """
        super(SSBModel, self).__init__()
        self.theta = theta
        self.weights = nn.Parameter(torch.randn(len(theta)))
        self.phi_function = PhiFunction(theta)
        
        
    def logprior(self, lambda_1, lambda_2):
        """
        Computes the regularization term for the model, combining the L2 norm of the weights and the regularization of the phi function parameters.
    
        The regularization term is given by:
        \lambda_1 \cdot \sum \text{weights}^2 + \lambda_2 \cdot \text{phi_function.logprior()}
    
        Here, \lambda_1 and \lambda_2 are regularization parameters that control the extent of regularization for the weights and phi function parameters, respectively.
    
        Parameters:
            lambda_1 (float): Regularization parameter for the weights.
            lambda_2 (float): Regularization parameter for the phi function.
    
        Returns:
            torch.tensor: The combined regularization term.
        """
        return lambda_1 * torch.sum(torch.pow(self.weights, 2)) + lambda_2 * self.phi_function.logprior()



    def likelihood(self, R):
        """
        Computes the likelihood of the preference data using a sigmoid function.
    
        For each pair of alternatives (X, Y) in R, this method computes the utility score f(X, Y) using the forward method. 
        The likelihood is then calculated as the sum of the sigmoid of these utility scores.
    
        The sigmoid function is used to transform the utility score into a probability between 0 and 1, reflecting 
        the probability of preferring X over Y.
    
        The formula for likelihood is given by:
        L(R) = \sum_{(X, Y) \in R} \sigma(f(X, Y))
        where \sigma is the sigmoid function, and f(X, Y) is the utility score for the pair (X, Y).
    
        Parameters:
            R (list): A list of pairs of alternatives.
    
        Returns:
            torch.tensor: The likelihood value.
        """
        l = torch.tensor(0).Float()
        for sparse_X, sparse_Y in R:
            f = self.forward(sparse_X, sparse_Y)
            l += torch.sigmoid(f)
        return l

    def forward(self, sparse_X, sparse_Y):
        """
        Computes the utility score for a pair of alternatives X and Y.

        Parameters:
            sparse_X (list): A sparse representation of alternative X.
            sparse_Y (list): A sparse representation of alternative Y.

        Returns:
            torch.tensor: The utility score for the pair (X, Y).
        """
        f = 0
        for S in self.theta:
            if all(i in sparse_X for i in S):
                f += self.weights[self.theta.index(S)]
            if all(i in sparse_Y for i in S):
                f -= self.weights[self.theta.index(S)]
        f += self.phi_function(sparse_X, sparse_Y)
        return f

    def accuracy(self, s):
        """
        Computes the accuracy of the model on a dataset.

        Parameters:
            s (list): A list of pairs of alternatives along with their actual preference.

        Returns:
            float: The accuracy of the model.
        """
        L = []
        for x, y in s:
            f = self.forward(x, y)
            L.append(1 if f > 0 else 0)
        return np.mean(L)

    def display_model(self):
        """
        Displays the model parameters.
        """
        for i in range(self.weights.shape[0]):
            if self.weights[i] > 1e-2:
                print(f"w[{i}] = {self.weights[i]}")
        for (i, j, A, B) in self.phi_function.phi_values: 
            if np.abs(self.phi_function.phi_values[(i, j, A, B)].item()) > 1e-2:
                print(f"C[{i}, {j}, {A}, {B}] = {self.phi_function.phi_values[(i, j, A, B)].item()}", end=" \n")
import torch.optim as optim

def fit_M_mle(model, R, lr=0.05, num_epochs=200):
    """
    Fit the phi function parameters of the model using Maximum Likelihood Estimation (MLE).

    Parameters:
    model: The SSB model instance.
    R: Preference data.
    lr (float): Learning rate for the optimizer. Defaults to 0.05.
    num_epochs (int): Number of training epochs. Defaults to 200.

    Returns:
    list: A list of accuracy values at each epoch.

    Description:
    The function optimizes the phi function parameters of the model to maximize the likelihood
    of the observed preference data R. It uses the Adam optimizer for the gradient descent.
    The likelihood is calculated as:
    $$
    L(R) = \prod_{(X, Y) \in R} \sigma(f(X, Y))
    $$
    The loss is the negative log-likelihood.
    """
    phi_params = [model.phi_function.phi_values[k] for k in model.phi_function.phi_values]
    optimizer = optim.Adam(phi_params, lr=lr)
    losses = []
    losses.append(model.accuracy(R))
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        likelihood = model.likelihood(R).double()
        loss = -torch.log(likelihood)
        loss.backward()
        optimizer.step()
        losses.append(model.accuracy(R))
    return losses

def fit_w_map(model, R, lr=0.05, num_epochs=500, lambda_1=1e-5):
    """
    Fit the weights of the model using Maximum a Posteriori (MAP) estimation.

    Parameters:
    model: The SSB model instance.
    R: Preference data.
    lr (float): Learning rate. Defaults to 0.05.
    num_epochs (int): Number of training epochs. Defaults to 500.
    lambda_1 (float): Regularization parameter for the weights. Defaults to 1e-5.

    Returns:
    list: A list of accuracy values at each epoch.

    Description:
    The function optimizes the weights of the model to maximize the posterior likelihood
    considering a regularization term for the weights. The loss function is:
    $$
    -\log(L(R)) + \lambda_1 \sum w^2
    $$
    where $L(R)$ is the likelihood of the preference data R.
    """
    optimizer = optim.Adam([model.weights], lr=lr)
    losses = []
    losses.append(model.accuracy(R))
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        likelihood = model.likelihood(R)
        loss = -torch.log(likelihood) + lambda_1 * torch.sum(torch.pow(model.weights, 2))
        loss.backward()
        optimizer.step()
        losses.append(model.accuracy(R))
    return losses

def fit_M_map(model, R, lr=0.05, num_epochs=500, lambda_2=1e-5):
    """
    Fit the phi function parameters of the model using MAP estimation.

    Parameters:
    model: The SSB model instance.
    R: Preference data.
    lr (float): Learning rate. Defaults to 0.05.
    num_epochs (int): Number of training epochs. Defaults to 500.
    lambda_2 (float): Regularization parameter for the phi function. Defaults to 1e-5.

    Returns:
    list: A list of accuracy values at each epoch.

    Description:
    The function optimizes the phi function parameters of the model to maximize the posterior likelihood
    with a regularization term for the phi function parameters. The loss function is:
    $$
    -\log(L(R)) + \lambda_2 \sum |phi\_params|
    $$
    where $L(R)$ is the likelihood of the preference data R.
    """
    phi_params = [model.phi_function.phi_values[k] for k in model.phi_function.phi_values]
    model.weights.requires_grad = False
    optimizer = optim.Adam(phi_params, lr=lr)
    losses = []
    losses.append(model.accuracy(R))
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        likelihood = model.likelihood(R)
        loss = -torch.log(likelihood) + lambda_2 * sum([torch.abs(i) for i in phi_params])
        loss.backward()
        optimizer.step()
        losses.append(model.accuracy(R))
    return losses
