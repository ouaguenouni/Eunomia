import cvxopt
import scipy.special
import numpy as np

def test_degree_prf(prf, k):
    """
    Test the degree of additivity of a set of pairwise preferences.

    This function determines whether there exists a k-additive function that 
    can represent the given set of pairwise preferences. It sets up and solves 
    a quadratic programming problem where the decision variables may correspond 
    to the weights or coefficients of the k-additive function.

    Parameters:
    prf (list of tuples): A list of pairwise preferences. Each element in `prf`
                          is a tuple `(A, B)`, where `A` and `B` are subsets of 
                          elements, indicating a preference of `A` over `B`.
    k (int): The degree of additivity to be tested. This function checks if 
             there exists at least one k-additive function that represents 
             the preferences in `prf`.

    Returns:
    bool: Returns True if there exists at least one k-additive function that 
          can represent the preferences, False otherwise.
    """

    n = len(prf)

    # Create P matrix
    P = np.zeros((n, n))
    for i, x in enumerate(prf):
        A, B = x
        for j, y in enumerate(prf):
            C, D = y
            coef = degree_kernel(A, C, k) - degree_kernel(A, D, k) \
                   - degree_kernel(B, C, k) + degree_kernel(B, D, k)
            P[i, j] = coef

    # Create q vector
    q = -np.ones((n,))

    # Create G matrix and h vector for inequality constraints
    G = -np.eye(n)
    h = np.zeros((n,))

    # Convert to cvxopt matrices
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # Solving the quadratic program
    solution = cvxopt.solvers.qp(P, q, G, h)

    # Check if a solution exists
    return solution['status'] == 'optimal'


def compute_additivity_degree(prf, n):
    """
    Compute the smallest degree of additivity k for a given set of pairwise preferences.

    This function iteratively tests different degrees of additivity (k) from 1 to n
    and returns the smallest k for which there exists at least one k-additive function
    that can represent the preferences in `prf`.

    Parameters:
    prf (list of tuples): A list of pairwise preferences, where each element is a tuple
                          `(A, B)`, indicating a preference of subset `A` over subset `B`.
    n (int): The upper limit for testing the degree of additivity. The function will test
             values of k from 1 up to n.

    Returns:
    int: The smallest degree of additivity k that can represent the preferences. If no such 
         k is found up to n, returns None.
    """

    for k in range(1, n + 1):
        if test_degree_prf(prf, k):
            return k
    
    return None