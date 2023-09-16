import numpy as np
from elexmodel.models.BootstrapElectionModel import OLSRegression

def test_cross_validation(bootstrap_election_model, rng):
    """
    Tests cross validation for finding the optimal lambda. 
    TODO: figure out how to force it to find a different lambda also
    """
    x = rng.normal(loc=2, scale=5, size=(100, 5))
    beta = rng.integers(low=-100, high=100, size=(5, 1))
    y = x @ beta + rng.normal(loc=0, scale=1, size=(100, 1))
    lambdas = [0, 1, 100]
    res = bootstrap_election_model.cv_lambda(x, y, lambdas)
    assert res == 0

def test_estimate_epsilon(bootstrap_election_model):
    """
    Testing estimating the contest level effect.
    """
    # the contest level effect (epsilon) is estiamted as the average of the residuals in that contest
    # if a contest has less then 2 units reporting then the contest level effect is set to zero
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator)
    assert epsilon_hat[0] == (residuals[0] + residuals[1]) / 2
    assert epsilon_hat[1] == (residuals[2] + residuals[3]) / 2
    assert epsilon_hat[2] == 0 # since the aggrgate indicator for that row only has one 1 which is less than 2

def test_estimate_delta(bootstrap_election_model):
    """
    Testing estimating the unit level error
    """
    # the unit level error is the difference between the residu and the state level effect
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator) # [0.5, 0.55, 0]
    delta_hat = bootstrap_election_model._estimate_delta(residuals, epsilon_hat, aggregate_indicator)
    desired = np.asarray([0.0, 0.0, -0.25, 0.25, 0.5])
    np.testing.assert_array_almost_equal(desired, delta_hat)

def test_estimate_model_error(bootstrap_election_model, rng):
    x = rng.normal(loc=2, scale=5, size=(100, 5))
    beta = rng.integers(low=-100, high=100, size=(5, 1))
    y = x @ beta
    ols_regression = OLSRegression()
    ols_regression.fit(x, y)
    aggregate_indicator = rng.multivariate_hypergeometric([1] * 5, 1, size=100)
    residuals, epsilon_hat, delta_hat = bootstrap_election_model._estimate_model_errors(ols_regression, x, y, aggregate_indicator)
    y_hat = ols_regression.predict(x)
    assert np.testing.assert_array_almost_equal(ols_regression.residuals(y, y_hat, loo=True, center=True), residuals)