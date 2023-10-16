import numpy as np
import random
import itertools


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


def compute_additive_ranks(alt_matrix, theta, w):
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


