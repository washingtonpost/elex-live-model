import numpy as np
import pandas as pd
import pytest
import scipy as sp

from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler
from elexmodel.models.BootstrapElectionModel import OLSRegressionSolver


def test_cross_validation(bootstrap_election_model, rng):
    """
    Tests cross validation for finding the optimal lambda.
    """
    x = rng.normal(loc=2, scale=5, size=(100, 75))
    x[:, 0] = 1
    beta = rng.normal(size=(75, 1))
    y = x @ beta + rng.normal(loc=0, scale=1, size=(100, 1))
    lambdas = [-0.1, 0, 0.01, 0.1, 1, 100]
    res = bootstrap_election_model.cv_lambda(x, y, lambdas)
    assert res == 0.01

    beta = rng.normal(scale=0.001, size=(75, 1))
    y = x @ beta + rng.normal(loc=0, scale=1, size=(100, 1))
    lambdas = [-0.1, 0, 0.01, 0.1, 1, 100]
    res = bootstrap_election_model.cv_lambda(x, y, lambdas)
    assert res == 100


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
    reporting_units = pd.DataFrame(
        [["a", True, True], ["b", True, True], ["c", True, True]],
        columns=["county_classification", "reporting", "expected"],
    )
    nonreporting_units = pd.DataFrame(
        [["c", False, True], ["d", False, True]], columns=["county_classification", "reporting", "expected"]
    )
    x_train_strata, x_test_strata = bootstrap_election_model._get_strata(reporting_units, nonreporting_units)

    assert "intercept" in x_train_strata.columns
    # a has been dropped
    assert "county_classification_b" in x_train_strata.columns
    assert "county_classification_c" in x_train_strata.columns
    assert "county_classification_d" in x_train_strata.columns
    np.testing.assert_array_almost_equal(x_train_strata.values, np.asarray([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]]))

    assert "intercept" in x_test_strata.columns
    assert "county_classification_b" in x_test_strata.columns
    assert "county_classification_c" in x_test_strata.columns
    assert "county_classification_d" in x_test_strata.columns

    np.testing.assert_array_almost_equal(x_test_strata.values, np.asarray([[1, 0, 1, 0], [1, 0, 0, 1]]))


def test_estimate_strata_dist(bootstrap_election_model, rng):
    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    delta_hat = np.asarray([-0.3, 0.5, 0.2, 0.25, 0.28])
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(
        x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub
    )
    assert ppf[(1, 0, 0)](0.01) == pytest.approx(delta_hat[:2].min())
    assert ppf[(1, 0, 0)](0.34) == pytest.approx(lb)
    assert ppf[(1, 0, 0)](0.66) == pytest.approx(ub)
    assert ppf[(1, 0, 0)](0.99) == pytest.approx(delta_hat[:2].max())

    # since all elements of the second strata are > 0.1 and < 0.3 we use the lb and ub for the first quantiles
    assert ppf[(1, 1, 0)](0.01) == pytest.approx(lb)
    assert ppf[(1, 1, 0)](0.25) == pytest.approx(lb)
    assert ppf[(1, 1, 0)](0.26) == pytest.approx(delta_hat[2:].min())
    assert ppf[(1, 1, 0)](0.50) == pytest.approx(np.median(delta_hat[2:]))
    assert ppf[(1, 1, 0)](0.51) == pytest.approx(delta_hat[2:].max())
    assert ppf[(1, 1, 0)](0.74) == pytest.approx(delta_hat[2:].max())
    assert ppf[(1, 1, 0)](0.75) == pytest.approx(ub)
    assert ppf[(1, 1, 0)](0.99) == pytest.approx(ub)

    assert ppf[(1, 0, 1)](0.49) == pytest.approx(lb)
    assert ppf[(1, 0, 1)](0.50) == pytest.approx(ub)

    # TODO: where is this -0.4 coming from?
    assert cdf[(1, 0, 0)](-0.4) == pytest.approx(0.01)
    assert cdf[(1, 0, 0)](-0.3) == pytest.approx(0.33)
    assert cdf[(1, 0, 0)](0.3) == pytest.approx(0.66)
    assert cdf[(1, 0, 0)](0.5) == pytest.approx(0.99)

    assert cdf[(1, 1, 0)](0.1) == pytest.approx(0.01)
    assert cdf[(1, 1, 0)](0.2) == pytest.approx(0.49)
    assert cdf[(1, 1, 0)](0.28) == pytest.approx(0.74)
    assert cdf[(1, 1, 0)](0.3) == pytest.approx(0.99)

    assert cdf[(1, 0, 1)](0.1) == pytest.approx(0.01)
    assert cdf[(1, 0, 1)](0.3) == pytest.approx(0.99)


def test_generate_nonreporting_bounds(bootstrap_election_model, rng):
    nonreporting_units = pd.DataFrame(
        [[0.1, 1.2, 75], [0.8, 0.8, 24], [0.1, 0.01, 0], [-0.2, 0.8, 99], [-0.3, 0.9, 100]],
        columns=["results_normalized_margin", "turnout_factor", "percent_expected_vote"],
    )

    # assumes that all outstanding vote will go to one of the two parties
    lower, upper = bootstrap_election_model._generate_nonreporting_bounds(
        nonreporting_units, "results_normalized_margin"
    )

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

    lower, upper = bootstrap_election_model._generate_nonreporting_bounds(nonreporting_units, "turnout_factor")

    assert lower[0] == pytest.approx(0.96)
    assert upper[0] == pytest.approx(4.8)

    assert lower[1] == pytest.approx(1.081081081)
    assert upper[1] == pytest.approx(80)  # this is 80 since we divide 0.8 / 0.01 (it's clipped)

    assert lower[2] == bootstrap_election_model.z_unobserved_lower_bound
    assert upper[2] == bootstrap_election_model.z_unobserved_upper_bound

    assert lower[3] == pytest.approx(0.536912752)
    assert upper[3] == pytest.approx(1.632653061)

    assert lower[4] == bootstrap_election_model.z_unobserved_lower_bound
    assert upper[4] == bootstrap_election_model.z_unobserved_upper_bound

    # changing parameters
    bootstrap_election_model.y_unobserved_lower_bound = -0.8
    bootstrap_election_model.y_unobserved_upper_bound = 0.8

    lower, upper = bootstrap_election_model._generate_nonreporting_bounds(
        nonreporting_units, "results_normalized_margin"
    )
    assert lower[0] == pytest.approx(-0.125)
    assert upper[0] == pytest.approx(0.275)

    assert lower[1] == pytest.approx(-0.416)
    assert upper[1] == pytest.approx(0.8)

    assert lower[2] == bootstrap_election_model.y_unobserved_lower_bound
    assert upper[2] == bootstrap_election_model.y_unobserved_upper_bound

    assert lower[3] == pytest.approx(-0.206)
    assert upper[3] == pytest.approx(-0.19)

    assert lower[4] == bootstrap_election_model.y_unobserved_lower_bound
    assert upper[4] == bootstrap_election_model.y_unobserved_upper_bound

    bootstrap_election_model.y_unobserved_lower_bound = 0.8
    bootstrap_election_model.y_unobserved_upper_bound = 1.2
    bootstrap_election_model.percent_expected_vote_error_bound = 0.1

    lower, upper = bootstrap_election_model._generate_nonreporting_bounds(nonreporting_units, "turnout_factor")
    assert lower[0] == pytest.approx(1.411764706)
    assert upper[0] == pytest.approx(1.846153846)

    assert lower[1] == pytest.approx(2.352941176)
    assert upper[1] == pytest.approx(5.714285714)

    assert lower[2] == bootstrap_election_model.z_unobserved_lower_bound
    assert upper[2] == bootstrap_election_model.z_unobserved_upper_bound

    assert lower[3] == pytest.approx(0.733944954)
    assert upper[3] == pytest.approx(0.898876404)

    assert lower[4] == bootstrap_election_model.z_unobserved_lower_bound
    assert upper[4] == bootstrap_election_model.z_unobserved_upper_bound


def test_strata_pit(bootstrap_election_model, rng):
    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    delta_hat = np.asarray([-0.3, 0.5, 0.2, 0.25, 0.28])
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(
        x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub
    )

    x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)
    uniforms = bootstrap_election_model._strata_pit(x_train_strata, x_train_strata_unique, delta_hat, cdf)

    assert uniforms[0] == pytest.approx(
        0.33
    )  # since -0.3 is the lowest of the three elements [-0.3, 0.5, 0.3] where 0.3 is the UB (LB is -0.3)
    assert uniforms[1] == pytest.approx(0.99)  # since 0.5 is the largest
    assert uniforms[2] == pytest.approx(0.49)  # [-0.3, 0.2, 0.25, 0.28, 0.3]
    assert uniforms[3] == pytest.approx(0.5)
    assert uniforms[4] == pytest.approx(0.74)

    uniforms = bootstrap_election_model._strata_pit(x_train_strata, x_train_strata_unique, delta_hat, cdf)


def test_bootstrap_deltas(bootstrap_election_model):
    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    delta_hat = np.asarray([-0.3, 0.5, 0.2, 0.25, 0.28])
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(
        x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub
    )
    x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)
    uniforms = bootstrap_election_model._strata_pit(x_train_strata, x_train_strata_unique, delta_hat, cdf)
    unifs = np.concatenate([uniforms, uniforms], axis=1)
    x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)
    bootstrap_deltas_y, bootstrap_deltas_z = bootstrap_election_model._bootstrap_deltas(
        unifs, x_train_strata, x_train_strata_unique, ppf, ppf
    )
    assert bootstrap_deltas_y.shape == (delta_hat.shape[0], bootstrap_election_model.B)
    assert bootstrap_deltas_z.shape == (delta_hat.shape[0], bootstrap_election_model.B)
    # testing that all elements of bootstrap delta is one of delta hat or lb or ub
    assert np.isclose(
        bootstrap_deltas_z.flatten().reshape(1, -1), np.concatenate([delta_hat, [0.1, 0.3]]).reshape(-1, 1), rtol=0.001
    ).any(0).mean() == pytest.approx(1)
    assert np.isclose(
        bootstrap_deltas_y.flatten().reshape(1, -1), np.concatenate([delta_hat, [0.1, 0.3]]).reshape(-1, 1), rtol=0.001
    ).any(0).mean() == pytest.approx(1)


def test_bootstrap_epsilons(bootstrap_election_model):
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator)
    delta_hat = bootstrap_election_model._estimate_delta(residuals, epsilon_hat, aggregate_indicator)

    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(
        x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub
    )

    bootstrap_epsilons_y, bootstrap_epsilons_z = bootstrap_election_model._bootstrap_epsilons(
        epsilon_hat, epsilon_hat, x_train_strata, x_train_strata_unique, ppf, ppf, aggregate_indicator
    )
    assert bootstrap_epsilons_y.shape == (epsilon_hat.shape[0], bootstrap_election_model.B)
    assert bootstrap_epsilons_z.shape == (epsilon_hat.shape[0], bootstrap_election_model.B)

    # the last epsilon only has one element, so epsilon_hat is zero so the bootstrapped versions should be zero also
    # TODO: IS THIS CORRECT?
    assert np.isclose(bootstrap_epsilons_y[-1], 0).mean() == pytest.approx(1)

    # the others have the correct mean
    assert bootstrap_epsilons_y[0].mean() == pytest.approx(epsilon_hat[0], rel=0.1)
    assert bootstrap_epsilons_y[1].mean() == pytest.approx(epsilon_hat[1], rel=0.1)


def test_bootstrap_errors(bootstrap_election_model):
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator)
    delta_hat = bootstrap_election_model._estimate_delta(residuals, epsilon_hat, aggregate_indicator)

    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(
        x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub
    )
    epsilon_B, delta_B = bootstrap_election_model._bootstrap_errors(
        epsilon_hat, epsilon_hat, delta_hat, delta_hat, x_train_strata, cdf, cdf, ppf, ppf, aggregate_indicator
    )
    epsilon_y_B, epsilon_z_B = epsilon_B
    delta_y_B, delta_z_B = delta_B

    assert epsilon_y_B.shape == (aggregate_indicator.shape[1], bootstrap_election_model.B)
    assert epsilon_z_B.shape == (aggregate_indicator.shape[1], bootstrap_election_model.B)
    assert delta_y_B.shape == (residuals.shape[0], bootstrap_election_model.B)
    assert delta_z_B.shape == (residuals.shape[0], bootstrap_election_model.B)
    # the values of bootstrap epsilon and delta are checked above


def test_sample_test_delta(bootstrap_election_model):
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator)
    delta_hat = bootstrap_election_model._estimate_delta(residuals, epsilon_hat, aggregate_indicator)

    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(
        x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub
    )

    delta_y, delta_z = bootstrap_election_model._sample_test_delta(x_test_strata, ppf, ppf)

    assert delta_y.shape == (x_test_strata.shape[0], bootstrap_election_model.B)
    assert delta_z.shape == (x_test_strata.shape[0], bootstrap_election_model.B)
    # TODO: THESE SHOULD NOT BE FAILING
    # assert np.isclose(delta_y.flatten().reshape(1, -1), np.concatenate([delta_hat, [0.1, 0.3]]).reshape(-1, 1), rtol=0.01).any(0).mean() == pytest.approx(1)
    # assert np.isclose(delta_z.flatten().reshape(1, -1), np.concatenate([delta_hat, [0.1, 0.3]]).reshape(-1, 1), rtol=0.001).any(0).mean() == pytest.approx(1)

    # np.where(np.isclose(delta_y.flatten().reshape(1, -1), np.concatenate([delta_hat, [0.1, 0.3]]).reshape(-1, 1), rtol=0.1).any(0) == False)


def test_sample_test_epsilon(bootstrap_election_model):
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator_train = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]).T
    aggregate_indicator_test = np.asarray([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator_train)

    epsilon_y, epsilon_z = bootstrap_election_model._sample_test_epsilon(
        residuals, residuals, epsilon_hat, epsilon_hat, aggregate_indicator_train, aggregate_indicator_test
    )
    # TODO: confirm that sample test epsilon function is doin the right thing!!

    # column 3 doesn't have any units that are reporting
    #


def test_sample_test_errors(bootstrap_election_model):
    residuals = np.asarray([0.5, 0.5, 0.3, 0.8, 0.5]).reshape(-1, 1)
    aggregate_indicator_train = np.asarray([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]).T
    # aggregate_indicator_test = np.asarray([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]]).T
    epsilon_hat = bootstrap_election_model._estimate_epsilon(residuals, aggregate_indicator_train)
    delta_hat = bootstrap_election_model._estimate_delta(residuals, epsilon_hat, aggregate_indicator_train)

    x_train, x_test = None, None
    x_train_strata = pd.DataFrame([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    x_test_strata = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 0, 1]])
    lb = 0.1
    ub = 0.3
    ppf, cdf = bootstrap_election_model._estimate_strata_dist(
        x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub
    )
    # TODO figure out why the aggregate_indicator things are doing weird stuff

    # test_error_y, test_error_z = bootstrap_election_model._sample_test_errors(residuals, residuals, epsilon_hat, epsilon_hat, x_test_strata, ppf, ppf, aggregate_indicator_train, aggregate_indicator_test)
    # assert test_error_y.shape == (aggregate_indicator_test.shape[0], 1)
    # assert test_error_z.shape == (aggregate_indicator_test.shape[0], 1)


def test_compute_bootstrap_errors(bootstrap_election_model, va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["margin"]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    current_data = data_handler.get_percent_fully_reported(50)

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office_id, geographic_unit_type, estimands, {"margin": "margin"}, data=va_governor_county_data
    )

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data, current_data, estimands, geographic_unit_type, handle_unreporting="drop"
    )

    reporting_units = combined_data_handler.get_reporting_units(percent_reporting_threshold, turnout_factor_upper=2.0)
    nonreporting_units = combined_data_handler.get_nonreporting_units(
        percent_reporting_threshold, turnout_factor_upper=2.0
    )
    unexpected_units = combined_data_handler.get_unexpected_units(percent_reporting_threshold, ["postal_code"])

    assert not bootstrap_election_model.ran_bootstrap
    bootstrap_election_model.B = 10
    bootstrap_election_model.compute_bootstrap_errors(reporting_units, nonreporting_units, unexpected_units)
    assert bootstrap_election_model.ran_bootstrap
    assert bootstrap_election_model.weighted_yz_test_pred.shape == (nonreporting_units.shape[0], 1)
    assert bootstrap_election_model.weighted_yz_test_pred.shape == bootstrap_election_model.weighted_z_test_pred.shape
    assert bootstrap_election_model.errors_B_1.shape == (nonreporting_units.shape[0], bootstrap_election_model.B)
    assert bootstrap_election_model.errors_B_2.shape == (nonreporting_units.shape[0], bootstrap_election_model.B)
    assert bootstrap_election_model.errors_B_3.shape == (nonreporting_units.shape[0], bootstrap_election_model.B)
    assert bootstrap_election_model.errors_B_4.shape == (nonreporting_units.shape[0], bootstrap_election_model.B)


def test_get_unit_predictions(bootstrap_election_model, va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["margin"]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    current_data = data_handler.get_percent_fully_reported(50)

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office_id, geographic_unit_type, estimands, {"margin": "margin"}, data=va_governor_county_data
    )

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data, current_data, estimands, geographic_unit_type, handle_unreporting="drop"
    )

    reporting_units = combined_data_handler.get_reporting_units(percent_reporting_threshold, turnout_factor_upper=2.0)
    nonreporting_units = combined_data_handler.get_nonreporting_units(
        percent_reporting_threshold, turnout_factor_upper=2.0
    )
    unexpected_units = combined_data_handler.get_unexpected_units(percent_reporting_threshold, ["postal_code"])

    bootstrap_election_model.B = 10
    assert not bootstrap_election_model.ran_bootstrap
    unit_predictions = bootstrap_election_model.get_unit_predictions(
        reporting_units, nonreporting_units, estimand="margin", unexpected_units=unexpected_units
    )
    assert bootstrap_election_model.ran_bootstrap
    assert unit_predictions.shape == (nonreporting_units.shape[0], 1)


def test_is_top_level_aggregate(bootstrap_election_model):
    assert bootstrap_election_model._is_top_level_aggregate(["postal_code"])
    assert bootstrap_election_model._is_top_level_aggregate(["postal_code", "district"])

    assert not bootstrap_election_model._is_top_level_aggregate(["postal_code", "county_fips"])
    assert not bootstrap_election_model._is_top_level_aggregate(["postal_code", "district", "county_classification"])
    assert not bootstrap_election_model._is_top_level_aggregate(["county_fips"])
    assert not bootstrap_election_model._is_top_level_aggregate([])


def test_aggregate_predictions(bootstrap_election_model):
    reporting_units = pd.DataFrame(
        [
            ["a", -3, 0.2, 1, 1, 1, 1, 3, 5, 8],
            ["a", 1, 0, 1, 1, 1, 1, 2, 1, 3],
            ["b", 5, -0.1, 1, 1, 1, 1, 8, 3, 11],
            ["c", 3, -0.2, 1, 1, 1, 1, 9, 1, 9],
            ["c", 3, 0.8, 1, 1, 1, 1, 2, 4, 6],
        ],
        columns=[
            "postal_code",
            "pred_margin",
            "results_margin",
            "results_weights",
            "baseline_weights",
            "turnout_factor",
            "reporting",
            "baseline_dem",
            "baseline_gop",
            "baseline_turnout",
        ],
    )
    nonreporting_units = pd.DataFrame(
        [
            ["a", -3, 0.2, 1, 1, 1, 0, 3, 5, 8],
            ["b", 1, 0, 1, 1, 1, 0, 2, 1, 3],
            ["d", 5, -0.1, 1, 1, 1, 0, 8, 3, 11],
            ["d", 3, 0.8, 1, 1, 1, 0, 2, 4, 6],
            ["e", 4, 0.1, 1, 1, 1, 0, 5, 1, 9],
            ["e", 4, 0.1, 1, 1, 1, 0, 5, 1, 9],
        ],
        columns=[
            "postal_code",
            "pred_margin",
            "results_margin",
            "results_weights",
            "baseline_weights",
            "turnout_factor",
            "reporting",
            "baseline_dem",
            "baseline_gop",
            "baseline_turnout",
        ],
    )
    unexpected_units = pd.DataFrame(
        [
            ["a", -3, 0.2, 1, 1, 1, 0, np.nan, np.nan, np.nan],
            ["d", 1, 0, 1, 1, 1, np.nan, np.nan, np.nan],
            ["f", 5, -0.1, 1, 1, 1, 0, np.nan, np.nan, np.nan],
            ["f", 3, 0.8, 1, 1, 1, 0, np.nan, np.nan, np.nan],
        ],
        columns=[
            "postal_code",
            "pred_margin",
            "results_margin",
            "results_weights",
            "baseline_weights",
            "turnout_factor",
            "reporting",
            "baseline_dem",
            "baseline_gop",
            "baseline_turnout",
        ],
    )

    bootstrap_election_model.weighted_z_test_pred = np.asarray([1, 1, 1, 1, 1, 1]).reshape(-1, 1)

    aggregate_predictions = bootstrap_election_model.get_aggregate_predictions(
        reporting_units, nonreporting_units, unexpected_units, ["postal_code"], "margin"
    )

    assert aggregate_predictions[aggregate_predictions.postal_code == "a"].pred_margin[0] == pytest.approx(
        -2.6 / 4
    )  # (-3 (pred) + 0.2 + 0 (reporting margin) + 0.2 (unexpected margin))/ 4
    assert aggregate_predictions[aggregate_predictions.postal_code == "b"].pred_margin[1] == pytest.approx(
        0.9 / 2
    )  # (-0.1 + 1) / 2
    assert aggregate_predictions[aggregate_predictions.postal_code == "c"].pred_margin[2] == pytest.approx(
        0.3
    )  # (0.3 + 0.3) / 2
    assert aggregate_predictions[aggregate_predictions.postal_code == "d"].pred_margin[3] == pytest.approx(
        8 / 3
    )  # 0.8 / 3
    assert aggregate_predictions[aggregate_predictions.postal_code == "e"].pred_margin[4] == pytest.approx(
        4
    )  # (4 + 4) / 2
    assert aggregate_predictions[aggregate_predictions.postal_code == "f"].pred_margin[5] == pytest.approx(0.7 / 2)

    assert aggregate_predictions[aggregate_predictions.postal_code == "a"].results_margin[0] == pytest.approx(0.6 / 4)
    assert aggregate_predictions[aggregate_predictions.postal_code == "b"].results_margin[1] == pytest.approx(-0.1 / 2)
    assert aggregate_predictions[aggregate_predictions.postal_code == "c"].results_margin[2] == pytest.approx(0.3)
    assert aggregate_predictions[aggregate_predictions.postal_code == "d"].results_margin[3] == pytest.approx(0.7 / 3)
    assert aggregate_predictions[aggregate_predictions.postal_code == "e"].results_margin[4] == pytest.approx(0.1)
    assert aggregate_predictions[aggregate_predictions.postal_code == "f"].results_margin[5] == pytest.approx(0.7 / 2)

    assert aggregate_predictions[aggregate_predictions.postal_code == "a"].reporting[0] == pytest.approx(2)
    assert aggregate_predictions[aggregate_predictions.postal_code == "b"].reporting[1] == pytest.approx(1)
    assert aggregate_predictions[aggregate_predictions.postal_code == "c"].reporting[2] == pytest.approx(2)
    assert aggregate_predictions[aggregate_predictions.postal_code == "d"].reporting[3] == pytest.approx(0)
    assert aggregate_predictions[aggregate_predictions.postal_code == "e"].reporting[4] == pytest.approx(0)  # 0.8 / 3
    assert aggregate_predictions[aggregate_predictions.postal_code == "f"].reporting[5] == pytest.approx(0)  # 0.8 / 3

    # TODO: add test where weighted_z_test_pred is more complex
    # TODO: add test where aggregate is more complicated (eg. postal code, district)


def test_get_quantile(bootstrap_election_model):
    bootstrap_election_model.B = 1000
    alpha = 0.95
    lower_q, upper_q = bootstrap_election_model._get_quantiles(alpha)
    assert lower_q == pytest.approx(0.025)
    assert upper_q == pytest.approx(0.975)


def test_get_unit_prediction_intervals(bootstrap_election_model, rng):
    reporting_units, nonreporting_units = None, None
    n = 10
    B = 10000
    alpha = 0.95

    s = 2
    bootstrap_election_model.weighted_yz_test_pred = rng.normal(scale=1, size=(n, 1))
    bootstrap_election_model.errors_B_1 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.errors_B_2 = rng.normal(scale=s, size=(n, B))
    lower, upper = bootstrap_election_model.get_unit_prediction_intervals(
        reporting_units, nonreporting_units, alpha, "margin"
    )

    lower_alpha = (1 - alpha) / 2
    upper_alpha = 1 - lower_alpha

    lower_q = np.floor(lower_alpha * (B + 1)) / B
    upper_q = np.ceil(upper_alpha * (B - 1)) / B

    assert lower.shape == (n, 1)
    assert upper.shape == (n, 1)
    assert all(upper > lower)
    assert all(upper > bootstrap_election_model.weighted_yz_test_pred)
    assert all(bootstrap_election_model.weighted_yz_test_pred > lower)

    # sqrt of variance of two errors above
    normal_lower_q = sp.stats.norm.ppf(lower_q, scale=np.sqrt(s**2 + s**2))
    normal_upper_q = sp.stats.norm.ppf(upper_q, scale=np.sqrt(s**2 + s**2))
    # arbitrarily one can be off because of sampling variability and rounding
    assert ((bootstrap_election_model.weighted_yz_test_pred - normal_upper_q).round() == lower).mean() >= 0.9
    assert ((bootstrap_election_model.weighted_yz_test_pred - normal_lower_q).round() == upper).mean() >= 0.9


def test_get_aggregate_prediction_intervals(bootstrap_election_model, rng):
    reporting_units = pd.DataFrame(
        [
            ["a", -3, 0.2, 1, 1, 1, 1, 3, 5, 8],
            ["a", 1, 0, 1, 1, 1, 1, 2, 1, 3],
            ["b", 5, -0.1, 1, 1, 1, 1, 8, 3, 11],
            ["c", 3, -0.2, 1, 1, 1, 1, 9, 1, 9],
            ["c", 3, 0.8, 1, 1, 1, 1, 2, 4, 6],
        ],
        columns=[
            "postal_code",
            "pred_margin",
            "results_margin",
            "results_weights",
            "baseline_weights",
            "turnout_factor",
            "reporting",
            "baseline_dem",
            "baseline_gop",
            "baseline_turnout",
        ],
    )
    reporting_units["results_normalized_margin"] = reporting_units.results_margin / reporting_units.results_weights
    nonreporting_units = pd.DataFrame(
        [
            ["a", -3, 0.2, 1, 1, 1, 0, 3, 5, 8],
            ["b", 1, 0, 1, 1, 1, 0, 2, 1, 3],
            ["d", 5, -0.1, 1, 1, 1, 0, 8, 3, 11],
            ["d", 3, 0.8, 1, 1, 1, 0, 2, 4, 6],
            ["e", 4, 0.1, 1, 1, 1, 0, 5, 1, 9],
            ["e", 4, 0.1, 1, 1, 1, 0, 5, 1, 9],
        ],
        columns=[
            "postal_code",
            "pred_margin",
            "results_margin",
            "results_weights",
            "baseline_weights",
            "turnout_factor",
            "reporting",
            "baseline_dem",
            "baseline_gop",
            "baseline_turnout",
        ],
    )
    nonreporting_units["results_normalized_margin"] = (
        nonreporting_units.results_margin / nonreporting_units.results_weights
    )
    unexpected_units = pd.DataFrame(
        [
            ["a", -3, 0.2, 1, 1, 1, 0, np.nan, np.nan, np.nan],
            ["d", 1, 0, 1, 1, 1, np.nan, np.nan, np.nan],
            ["f", 5, -0.1, 1, 1, 1, 0, np.nan, np.nan, np.nan],
            ["f", 3, 0.8, 1, 1, 1, 0, np.nan, np.nan, np.nan],
        ],
        columns=[
            "postal_code",
            "pred_margin",
            "results_margin",
            "results_weights",
            "baseline_weights",
            "turnout_factor",
            "reporting",
            "baseline_dem",
            "baseline_gop",
            "baseline_turnout",
        ],
    )
    unexpected_units["results_normalized_margin"] = unexpected_units.results_margin / unexpected_units.results_weights

    n = nonreporting_units.shape[0]
    B = 10
    s = 1.0
    bootstrap_election_model.B = B
    bootstrap_election_model.errors_B_1 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.errors_B_2 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.errors_B_3 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.errors_B_4 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.weighted_z_test_pred = rng.normal(scale=s, size=(n, 1))
    bootstrap_election_model.weighted_yz_test_pred = rng.normal(scale=s, size=(n, 1))

    lower, upper = bootstrap_election_model.get_aggregate_prediction_intervals(
        reporting_units, nonreporting_units, unexpected_units, ["postal_code"], 0.95, None, None
    )

    assert lower.shape == (6, 1)
    assert upper.shape == (6, 1)

    assert lower[2] == pytest.approx(upper[2])  # since c is fully reporting
    assert lower[5] == pytest.approx(upper[5])  # since all f units are unexpected
    assert all(lower <= upper)

    # TODO: add test where aggregate is more complicated (eg. postal code, district)


def test_get_national_summary_estimates(bootstrap_election_model, rng):
    n = 10
    s = 2.0
    B = 20
    bootstrap_election_model.B = B
    bootstrap_election_model.aggregate_error_B_1 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.aggregate_error_B_2 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.aggregate_error_B_3 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.aggregate_error_B_4 = rng.normal(scale=s, size=(n, B))
    bootstrap_election_model.aggregate_perc_margin_total = rng.normal(scale=s, size=(n, 1))

    nat_sum_estimates = bootstrap_election_model.get_national_summary_estimates(None, None, 0, 0.95)
    assert "margin" in nat_sum_estimates
    assert len(nat_sum_estimates["margin"]) == 3
    assert nat_sum_estimates["margin"][0] >= nat_sum_estimates["margin"][1]
    assert nat_sum_estimates["margin"][0] <= nat_sum_estimates["margin"][2]

    # testing adding to base
    base_to_add = rng.random()
    nat_sum_estimates_w_base = bootstrap_election_model.get_national_summary_estimates(None, None, base_to_add, 0.95)
    assert nat_sum_estimates_w_base["margin"][0] == pytest.approx(nat_sum_estimates["margin"][0] + base_to_add)
    assert nat_sum_estimates_w_base["margin"][1] == pytest.approx(nat_sum_estimates["margin"][1] + base_to_add)
    assert nat_sum_estimates_w_base["margin"][2] == pytest.approx(nat_sum_estimates["margin"][2] + base_to_add)

    # test calling races
    states_called = {i: 1 for i in range(n)}
    nat_sum_data_dict = {i: 3 for i in range(n)}
    nat_sum_data_dict[1] = 7
    nat_sum_estimates = bootstrap_election_model.get_national_summary_estimates(
        nat_sum_data_dict, states_called, 0, 0.95
    )
    assert nat_sum_estimates["margin"][0] == pytest.approx(34)
    assert nat_sum_estimates["margin"][1] == pytest.approx(34)
    assert nat_sum_estimates["margin"][2] == pytest.approx(34)

    states_called = {i: 0 for i in range(n)}
    nat_sum_estimates = bootstrap_election_model.get_national_summary_estimates(
        nat_sum_data_dict, states_called, 0, 0.95
    )
    assert nat_sum_estimates["margin"][0] == pytest.approx(0)
    assert nat_sum_estimates["margin"][1] == pytest.approx(0)
    assert nat_sum_estimates["margin"][2] == pytest.approx(0)

    states_called = {i: 0 for i in range(n)}
    states_called[1] = 1
    nat_sum_estimates = bootstrap_election_model.get_national_summary_estimates(
        nat_sum_data_dict, states_called, 0, 0.95
    )
    assert nat_sum_estimates["margin"][0] == pytest.approx(7)
    assert nat_sum_estimates["margin"][1] == pytest.approx(7)
    assert nat_sum_estimates["margin"][2] == pytest.approx(7)
