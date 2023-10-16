import numpy as np
import random
import itertools


def sparse_to_alt(sparse_representation, n):
    """
    Convert a sparse tuple representation of an alternative back to its binary representation.

    Parameters:
    -----------
    sparse_representation : tuple
        A tuple containing the indices of features that are set to 1.
        
    n : int
        The total number of features in the binary representation.

    Returns:
    --------
    np.ndarray
        A binary numpy array representing the alternative.
    """
    alternative = np.zeros(n, dtype=int)
    for index in sparse_representation:
        alternative[index] = 1
    return alternative


def alt_to_sparse(alternative):
    """
    Convert a binary representation of an alternative to its sparse tuple representation.

    Parameters:
    -----------
    alternative : np.ndarray or list
        A binary array or list representing the alternative. Each element is either 0 or 1, 
        indicating the absence or presence of a feature, respectively.

    Returns:
    --------
    tuple
        A tuple containing the indices where the feature is set to 1, sorted in ascending order.
    """
    return tuple(i for i, v in enumerate(alternative) if v == 1)


def project_alternatives(alternatives, theta):
    """
    Project the alternatives to a new feature space defined by theta.

    Parameters:
        alternatives (numpy.ndarray): Matrix of shape (k, n) representing k alternatives.
        theta (list of sets): Each set contains indices representing features for projection.

    Returns:
        numpy.ndarray: Projected alternatives in new feature space.
    """
    projected_alternatives = []

    for alt in alternatives:
        new_alt = []
        for feature_set in theta:
            value = all(alt[i] for i in feature_set)
            new_alt.append(value)
        projected_alternatives.append(new_alt)

    return np.array(projected_alternatives)


