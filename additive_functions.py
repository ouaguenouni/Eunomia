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
    
    Example: F = {0,1,2};  suppose P(x) = 1 (for the sake of example)
    θ = {0,1,2,01,02,12}; w =  [1, 2, 3, 4, 5, 6, 7]
    To compute the semivalue of 0 for instance recall that : w_0 = w[0] = 1; w_01 = w[3] = 4; w_02 = w[4] = 5  
             p(1)*[Parameters involving 0 contained in the coalitions of size 1 with their coefficients] = p(1) * w[0] = p(1)*1
             p(2)*[Parameters involving 0 contained in the coalitions of size 2 with their coefficients] = p(2) * (2*w[0] + w[3] + w[4]) = p(2)*11
             p(3)*[Parameters involving 0 contained in the coalitions of size 3 with their coefficients] = p(3) * (w[0] + w[3] + w[4]) = p(3)*10
             Total : pb(1)*weights[0] + pb(2)*(weights[0] + weights[3]) + pb(2)*(weights[0] + weights[4]) + pb(3) * (weights[0] + weights[3] + weights[4])
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

def kendall_tau_distance(seq1, seq2):
    """
    Compute the Kendall Tau distance between two sorted sequences of indexes.

    Parameters:
    - seq1 (numpy.ndarray or list): The first sorted sequence of indexes.
    - seq2 (numpy.ndarray or list): The second sorted sequence of indexes.

    Returns:
    - int: The Kendall Tau distance.

    Example:
    To compute the Kendall Tau distance between two sequences:

    seq1 = [1, 3, 0, 2]
    seq2 = [1, 0, 3, 2]
    distance = kendall_tau_distance(seq1, seq2)
    print(distance)  # Output: 2
    """

    # Initialize the count of discordant pairs
    discordant_pairs = 0

    # Iterate through pairs of elements in both sequences
    for i in range(len(seq1)):
        for j in range(i + 1, len(seq1)):
            if (seq1[i] < seq1[j] and seq2[i] > seq2[j]) or (seq1[i] > seq1[j] and seq2[i] < seq2[j]):
                discordant_pairs += 1

    return discordant_pairs

def get_kt_distribution(rankings, gt):
    """
    Compute the Kendall Tau distance between each ranking in a set and a ground truth ranking.

    This function calculates the Kendall Tau distance for each ranking in the provided list 
    of rankings as compared to a given ground truth ranking. The Kendall Tau distance is 
    a measure of the difference between two rankings.

    Parameters:
    rankings (list of lists): A list where each element is a ranking (list) to be compared 
                              against the ground truth. Each ranking is a list of elements 
                              ordered according to some criterion.
    gt (list): The ground truth ranking. This is the reference ranking against which all 
               other rankings in `rankings` are compared.

    Returns:
    numpy.ndarray: An array of Kendall Tau distances, each representing the distance between 
                   the ground truth ranking and a ranking in the `rankings` list.
    """
    L = []
    for ranking in rankings:
        k = kendall_tau_distance(ranking, gt)
        L.append(k)
    
    L = np.array(L) 
    return L