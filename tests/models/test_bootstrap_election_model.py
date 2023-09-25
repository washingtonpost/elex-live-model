import pytest

import numpy as np
import pandas as pd

from elexmodel.models.BootstrapElectionModel import OLSRegressionSolver


def test_cross_validation(bootstrap_election_model, rng):
    """
    Tests cross validation for finding the optimal lambda.
    TODO: figure out how to force it to find a different lambda also
    """
    x = rng.normal(loc=2, scale=5, size=(100, 5))
    x[:, 0] = 1
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
    assert epsilon_hat[2] == 0  # since the aggrgate indicator for that row only has one 1 which is less than 2


def test_estimate_delta(bootstrap_election_model):
    """
    Testing estimating the unit level error
    """
    # the unit level error is the difference between the residu and the state level effect
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator)  # [0.5, 0.55, 0]
    delta_hat = bootstrap_election_model._estimate_delta(residuals, epsilon_hat, aggregate_indicator)
    desired = np.asarray([0.0, 0.0, -0.25, 0.25, 0.5])
    np.testing.assert_array_almost_equal(desired, delta_hat)


def test_estimate_model_error(bootstrap_election_model, rng):
    x = rng.normal(loc=2, scale=5, size=(100, 5))
    x[:, 0] = 1
    beta = rng.integers(low=-100, high=100, size=(5, 1))
    y = x @ beta
    ols_regression = OLSRegressionSolver()
    ols_regression.fit(x, y)
    aggregate_indicator = rng.multivariate_hypergeometric([1] * 5, 1, size=100)
    residuals, epsilon_hat, delta_hat = bootstrap_election_model._estimate_model_errors(
        ols_regression, x, y, aggregate_indicator
    )
    y_hat = ols_regression.predict(x)
    np.testing.assert_array_almost_equal(ols_regression.residuals(y, y_hat, loo=True, center=True), residuals)
    # epsilon_hat and delta_hat are tested above

def test_get_strata(bootstrap_election_model):
    reporting_units = pd.DataFrame([['a', True, True], ['b', True, True], ['c', True, True]], columns=['county_classification', 'reporting', 'expected'])
    nonreporting_units = pd.DataFrame([['c', False, True], ['d', False, True]], columns=['county_classification', 'reporting', 'expected'])
    x_train_strata, x_test_strata = bootstrap_election_model._get_strata(reporting_units, nonreporting_units)

    assert 'intercept' in x_train_strata.columns
    # a has been dropped
    assert 'county_classification_b' in x_train_strata.columns
    assert 'county_classification_c' in x_train_strata.columns
    assert 'county_classification_d' in x_train_strata.columns
    np.testing.assert_array_almost_equal(x_train_strata.values, np.asarray([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]]))

    assert 'intercept' in x_test_strata.columns
    assert 'county_classification_b' in x_test_strata.columns
    assert 'county_classification_c' in x_test_strata.columns
    assert 'county_classification_d' in x_test_strata.columns

    np.testing.assert_array_almost_equal(x_test_strata.values, np.asarray([[1, 0, 1, 0], [1, 0, 0, 1]]))


def test_estimate_strata_dist(bootstrap_election_model, rng):
    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    delta_hat = np.asarray([-0.3, 0.5, 0.2, 0.25, 0.28])
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub)

    # assert ppf[(1, 0, 0)](0.01) == pytest.approx(delta_hat[:2].min())
    # import pdb; pdb.set_trace()
    # assert ppf[(1, 0, 0)](0.49) == pytest.approx(delta_hat[:2].min())
    # assert ppf[(1, 0, 0)](0.51) == pytest.approx(delta_hat[:2].max())
    # assert ppf[(1, 0, 0)](0.99) == pytest.approx(delta_hat[:2].max())
    # import pdb; pdb.set_trace()
    # since all elements of the second strata are > 0.1 and < 0.3 we use the lb and ub for the first
    # quantiles
    # assert ppf[(1, 1, 0)](0.01) == lb
    # assert ppf[(1, 1, 0)](0.25) == lb
    # assert ppf[(1, 1, 0)](0.26) == pytest.approx(delta_hat[2:].min())
    # assert ppf[(1, 1, 0)](0.50) == pytest.approx(np.median(delta_hat[2:]))
    # assert ppf[(1, 1, 0)](0.51) == pytest.approx(delta_hat[2:].max())
    # assert ppf[(1, 1, 0)](0.74) == pytest.approx(delta_hat[2:].max())
    # assert ppf[(1, 1, 0)](0.75) == ub
    # assert ppf[(1, 1, 0)](0.99) == ub

def test_generate_nonreporting_bounds(bootstrap_election_model, rng):
    nonreporting_units = pd.DataFrame([[0.1, 1.2, 75], [0.8, 0.8, 24], [0.1, 0.01, 0], [-0.2, 0.8, 99], [-0.3, 0.9, 100]], columns=['results_normalized_margin', 'turnout_factor', 'percent_expected_vote'])
    
    # assumes that all outstanding vote will go to one of the two parties
    lower, upper = bootstrap_election_model._generate_nonreporting_bounds(nonreporting_units, 'results_normalized_margin')

    # hand checked in excel
    assert lower[0] == pytest.approx(-0.175)
    assert upper[0] == pytest.approx(0.325)
    
    assert lower[1] == pytest.approx(-0.568)
    assert upper[1] == pytest.approx(0.952)
    
    # if expected vote is close to 0 or 1 we set the bounds to be the extreme case
    assert lower[2] == bootstrap_election_model.y_unobserved_lower_bound
    assert upper[2] == bootstrap_election_model.y_unobserved_upper_bound

    assert lower[3] == pytest.approx(-0.208)
    assert upper[3] == pytest.approx(-0.188)

    assert lower[4] == bootstrap_election_model.y_unobserved_lower_bound
    assert upper[4] == bootstrap_election_model.y_unobserved_upper_bound

    lower, upper = bootstrap_election_model._generate_nonreporting_bounds(nonreporting_units, 'results_normalized_margin')
