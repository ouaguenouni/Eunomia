import numpy as np
import random
import itertools
from Eunomia.alternatives import *
from scipy.special import comb

def compute_ws_ranks(alt_matrix, theta, w):
    """
    Computes the additive ranks for a given set of alternatives and weights.

    Parameters:
        alt_matrix (numpy.ndarray): Matrix of alternatives of shape (k, n)
        theta (list): Set of feature sets to project alternatives onto
        w (numpy.ndarray): Vector of weights of shape (m, )

    Returns:
        ranks (numpy.ndarray): Vector of ranks of shape (k, )
    """
    # Step 1: Project the alternatives onto the new feature space defined by theta
    projected_alternatives = project_alternatives(alt_matrix, theta)

    # Step 2: Compute the ranks by taking the dot product along axis 1 (columns)
    ranks = np.dot(projected_alternatives, w)

    return ranks


def compute_semivalues(n, theta, weights, probability_function):
    """
    Compute the semivalues of each feature based on the provided parameters.

    Parameters:
    - n (int): The number of features.
    - theta (list of lists): A list containing subsets represented as lists of feature indices.
    - weights (list): A vector of weights for each element in theta.
    - probability_function (function): A function that takes a subset and returns a probability.

    Returns:
    - numpy.ndarray: A vector of semivalues, where the component i corresponds to the semivalue of feature i.
    """

    semivalues = np.zeros(n)
    #For each feature we want to compute the semivalue.
    for i in range(n):
        #We loop through all the parameters implying i
        for S in [s for s in theta if i in s]:
            #For each parameter we count the number of coalitions that contains it
            subset_size = len(S)
            for k in range(subset_size, n):
                binomial_coeff = comb(n - subset_size, k - subset_size)
                semivalues[i] += binomial_coeff * probability(k)
            semivalues[i] = semivalues[i] * w_S            
    return semivalues


def compute_semivalues(n, theta, weights, probability_function):
    """
    Compute the semivalues of each feature based on the provided parameters.

    Parameters:
    - n (int): The number of features.
    - theta (list of lists): A list containing subsets represented as lists of feature indices.
    - weights (list): A vector of weights for each element in theta.
    - probability_function (function): A function that takes a subset and returns a probability.

    Returns:
    - numpy.ndarray: A vector of semivalues, where the component i corresponds to the semivalue of feature i.
    
    Example: F = {0,1,2};  suppose P(x) = 1 (for the sake of example)
    θ = {0,1,2,01,02,12}; w =  [1, 2, 3, 4, 5, 6, 7]
    To compute the semivalue of 0 for instance recall that : w_0 = w[0] = 1; w_01 = w[3] = 4; w_02 = w[4] = 5  
             p(1)*[Parameters involving 0 contained in the coalitions of size 1 with their coefficients] = p(1) * w[0] = p(1)*1
             p(2)*[Parameters involving 0 contained in the coalitions of size 2 with their coefficients] = p(2) * (2*w[0] + w[3] + w[4]) = p(2)*11
             p(3)*[Parameters involving 0 contained in the coalitions of size 3 with their coefficients] = p(3) * (w[0] + w[3] + w[4]) = p(3)*10
    if we take p(x) = 1; then the semivalue of 0 is : 22.
    """

    semivalues = np.zeros(n)
    #For each feature we want to compute the semivalue.
    for i in range(n):
        #print(f"Computing the index of {i} : {[s for s in theta if i in s]}")
        #We loop through all the parameters implying i
        for S in [s for s in theta if i in s]:
            #For each parameter we count the number of coalitions that contains it
            subset_size = len(S)
            cpt = 0
            for k in range(subset_size, n+1):
                binomial_coeff = comb(n - subset_size, k - subset_size)
                cpt += binomial_coeff * probability_function(k)
            #print(f"Number of coalitions with {S} in them: ", cpt)
            semivalues[i] += cpt * weights[theta.index(S)]            
    return semivalues
