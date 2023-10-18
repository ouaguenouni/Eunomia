def mcmc_posterior(R):
    """
    Compute the posterior distribution of model parameters using Markov Chain Monte Carlo (MCMC).

    Parameters:
    - R (PreferenceMatrix): An instance of a PreferenceMatrix containing preference data.

    Returns:
    - pm.backends.base.MultiTrace: A MultiTrace object containing samples from the posterior distribution.

    """

    X = R.generate_preference_matrix()

    # Create labels (y) for the preference data
    y = np.ones((X.shape[0], 1))

    with pm.Model() as probit_model:
        # Define priors for weights and bias
        weights = pm.Normal('weights', mu=0, sigma=1, shape=X.shape[1])
        bias = pm.Normal('bias', mu=0, sigma=1)

        # Probit link function
        mu = pm.math.dot(X, weights) + bias
        phi = pm.math.invprobit(mu)  # Inverse probit link function
        y_obs = pm.Bernoulli('y_obs', p=phi, observed=np.ones(X.shape[0]))

        # Prior formula for weights and bias:
        # P(weights) ~ Normal(0, 1)
        # P(bias) ~ Normal(0, 1)

        # Sample from the posterior using MCMC
        trace = pm.sample(2000, tune=500, chains=2, target_accept=0.90)

    return trace


def trace_to_numpy(trace):
    """
    Convert a PyMC3 trace to a NumPy array format for weights and bias parameters.

    Parameters:
    - trace (pm.backends.base.MultiTrace): A PyMC3 MultiTrace object containing parameter samples.

    Returns:
    - numpy.ndarray: A NumPy array containing the samples of weights and bias parameters.

    Example:
    To convert a PyMC3 trace to a NumPy array:

    numpy_samples = trace_to_numpy(trace)
    """

    numpy_trace = {}

    # Iterate through variables in the posterior data
    for var in trace.posterior.data_vars.keys():
        numpy_trace[var] = trace.posterior[var].values

    # Concatenate the weights and bias samples along the last axis
    numpy_samples = np.concatenate([numpy_trace["weights"], numpy_trace["bias"].reshape((2, -1, 1))], axis=2)

    return numpy_samples





