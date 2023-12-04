import numpy as np
import random
import itertools
from Eunomia.alternatives import *



import numpy as np

class PreferenceModel:
    """
    A class to represent a preference model that contains a matrix of alternatives
    and a vector of ranks for those alternatives.
    
    Attributes:
    -----------
    alternatives : np.ndarray
        A 2D NumPy array representing the matrix of alternatives. Shape: (k, n).
        
    ranks : np.ndarray
        A 1D NumPy array representing the rank of each alternative. Shape: (k,).
        
    preference_matrix : np.ndarray
        A 2D NumPy array representing the preference matrix. Shape: Dynamic based on preferences.
        
    theta : list
        A list containing the columns on which the alternatives are to be projected.
        
    sparse_preference_set : set
        A set containing the sparse representations of the preference set.
        
    Methods:
    --------
    __init__(self, alternatives, ranks):
        Initializes a new PreferenceModel object.
        
    generate_preference_matrix(self, theta=None):
        Generates and returns a matrix representing the preferences.
        
    generate_sparse_preference_set(self):
        Generates and returns the sparse representation of the preference set.
        
    __str__(self):
        Returns the string representation of the preference set.
        
    __repr__(self):
        Returns the machine-readable string representation of the PreferenceModel object.
    """
    
    def __init__(self, alternatives, ranks, theta = None):
        """
        Initializes a new PreferenceModel object.
        
        Parameters:
        -----------
        alternatives : np.ndarray
            A 2D NumPy array representing the matrix of alternatives. Shape: (k, n).
            
        ranks : np.ndarray
            A 1D NumPy array representing the rank of each alternative. Shape: (k,).
        """
        self.alternatives = alternatives
        self.ranks = ranks
        self.preference_matrix = None
        if not theta:
            theta = [{i} for i in range(alternatives.shape[1])]  # Default to the set of all original features
        self.theta = theta
        self.sparse_preference_set = None

    def generate_preference_matrix(self, theta=None):
        """Generates and returns a matrix representing the preferences."""
        if theta is not None:
            self.theta = theta
        projected_alternatives = project_alternatives(self.alternatives, self.theta)
        
        k, _ = projected_alternatives.shape
        P = []
        
        for i in range(k):
            for j in range(k):
                if self.ranks[i] > self.ranks[j]:
                    preference_vector = projected_alternatives[i] - projected_alternatives[j]
                    P.append(preference_vector)

        self.preference_matrix = np.array(P)
        return self.preference_matrix

    def generate_sparse_preference_set(self):
        """Generates and returns the sparse representation of the preference set."""
        if self.preference_matrix is None:
            self.generate_preference_matrix()

        self.sparse_preference_set = set()

        k, _ = self.alternatives.shape
        for i in range(k):
            for j in range(k):
                if self.ranks[i] > self.ranks[j]:
                    sparse_A = alt_to_sparse(self.alternatives[i])
                    sparse_B = alt_to_sparse(self.alternatives[j])
                    self.sparse_preference_set.add((sparse_A, sparse_B))

        return self.sparse_preference_set

    def __str__(self):
        """Returns the string representation of the preference set."""
        if self.sparse_preference_set is None:
            self.generate_sparse_preference_set()

        return "\n".join([f"{str(A)} > {str(B)}" for A, B in self.sparse_preference_set])
    
    def __repr__(self):
        """Returns the machine-readable string representation of the PreferenceModel object."""
        return self.__str__()    

