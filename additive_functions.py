import numpy as np
import random
import itertools


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


