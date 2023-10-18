import numpy as np
import random
import itertools
from Eunomia.alternatives import *

generate_normal_weights = lambda theta,sigma:np.random.normal(0, sigma, size = (len(theta),))


def generate_random_theta(n, m):
    """
    Generate a random theta set of size m from the set of features F which has n features.
    The elements in the returned theta set are unique and sorted by their size.

    Parameters:
        n (int): The size of the feature set F.
        m (int): The size of the theta set to be returned.

    Returns:
        list: List of unique sets sorted by their size, representing theta.
    """
    theta = set()
    F = set(range(n))

    while len(theta) < m:
        k = random.randint(1, len(F))
        combination = tuple(sorted(random.sample(F, k)))

        if combination not in theta:
            theta.add(combination)
    
    return sorted(list(theta), key=len)


def generate_additive_theta(n, k):
    """
    Generate a theta set containing all subsets of F that are of size <= k.

    Parameters:
        n (int): The size of the feature set F.
        k (int): The maximum size for each subset in the theta set.
                Must be less than n.

    Returns:
        theta (list of sets): A list of sets, each set being a combination of features from F.
    """
    
    if k >= n:
        raise ValueError("k should be less than n.")
        
    theta = []
    F = set(range(n))

    for size in range(1, k + 1):
        for combination in itertools.combinations(F, size):
            theta.append(set(combination))

    return theta


def generate_random_weights(m):
    """
    Generate a random vector of weights of size m.

    Parameters:
        m (int): Size of the weight vector to be returned.

    Returns:
        numpy.ndarray: A numpy array containing m random weights.
    """
    return np.random.rand(m)



def generate_random_alternative(n):
    """
    Generate a random alternative with n features.
    Each feature is binary (either 0 or 1).
    
    Parameters:
        n (int): Number of features.
    
    Returns:
        numpy.ndarray: Randomly generated alternative.
    """
    return np.random.randint(2, size=n)


def generate_random_alternatives_matrix(m, n):
    """
    Generate a matrix of m distinct random alternatives, each with n features.

    Parameters:
        m (int): Number of alternatives.
        n (int): Number of features per alternative.

    Returns:
        numpy.ndarray: Matrix of m distinct random alternatives.
    """
    alternatives_matrix = []
    while len(alternatives_matrix) < m:
        alt = generate_random_alternative(n)
        if not any((alt == arr).all() for arr in alternatives_matrix):
            alternatives_matrix.append(alt)
    return np.array(alternatives_matrix)


def compute_set_theta(n, alpha,  p):
    """
    Compute a set θ by adding all singletons first and then αP subsets, where P is the number of subsets of F of size ≥ 2.

    Parameters:
    - alpha (float): The scaling factor for the number of subsets to add.
    - p (float): The probability of exiting the addition of a subset.
    - n (int): The size of the feature set F.

    Returns:
    - set of frozensets: A set θ containing unique frozensets representing the logical combinations of features.

    Example:
    Consider a scenario where n=10 (size of F), and we want to explore the effect of different alpha values on θ size:
    
    - For alpha=0.01, θ will have a size of approximately 6.
    - For alpha=0.05, θ will have a size of approximately 30.
    - For alpha=0.1, θ will have a size of approximately 60.
    - For alpha=0.15, θ will have a size of approximately 90.
    - For alpha=0.2, θ will have a size of approximately 120.
    """
    
    # Initialize a set to store θ as frozensets
    theta = set()

    # Step 1: Add all singletons to θ
    for i in range(n):
        theta.add(frozenset([i]))

    # Step 2: Compute the number of subsets of F of size ≥ 2
    num_subsets = 0
    for r in range(2, n + 1):
        num_subsets += len(list(itertools.combinations(range(n), r)))

    # Step 3: Add αP subsets sequentially
    while len(theta) < int(alpha * num_subsets):
        subset = set()
        while random.random() < p:
            element = random.randint(0, n - 1)
            subset.add(element)
        if len(subset) >= 2:  # Add only if the subset size is greater than or equal to 2
            theta.add(frozenset(subset))
    return sorted([sorted(i) for i in theta], key = len)



