import numpy as np
import pandas as pd
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

from elexmodel.models import BaseElectionModel

TOL = 1e-3


def test_fit_model():
    """
    Test fitting the model.
    """
    model_settings = {}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    qr = QuantileRegressionSolver(solver="ECOS")

    df_X = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 1, 1, 2]})
    df_y = pd.DataFrame({"y": [3, 8, 9, 15]}).y
    weights = pd.DataFrame({"weights": [1, 1, 1, 1]}).weights
    model.fit_model(qr, df_X, df_y, 0.5, weights, True)

    assert all(np.abs(qr.predict(df_X) - [8, 8, 8, 15]) <= TOL)
    assert all(np.abs(qr.coefficients - [1, 7]) <= TOL)


def test_get_unit_predictions():
    model_settings = {"features": ["b"]}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    df_X = pd.DataFrame(
        {
            "residuals_a": [1, 2, 3, 4],
            "total_voters_a": [4, 2, 9, 5],
            "last_election_results_a": [5, 1, 4, 2],
            "results_a": [0, 0, 0, 1],
            "b": [2, 3, 4, 5],
        }
    )
    model.get_unit_predictions(df_X, df_X, estimand="a", lambda_=1)

    "intercept" in model.features_to_coefficients
    "b" in model.features_to_coefficients
    model.features_to_coefficients["intercept"] > 0


def test_aggregation_simple():
    """
    This test is a basic test for aggregating reporting votes. We have have two data frames, reporting votes and
    reporting unexpected votes (when we didn't have a geographic unit in the preprocessed data) and we want to make
    sure that we are summing them together correctly. So both dataframes have a `results` Series that need to be
    summed based on the aggregate.
    """
    model_settings = {}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    estimand = "turnout"

    df1 = pd.DataFrame(
        {
            "c1": ["a", "a", "b", "b", "c", "c"],
            f"results_{estimand}": [1, 3, 7, 9, 2, 4],
            "reporting": [1, 1, 1, 1, 1, 1],
        }
    )

    df2 = df1.copy()
    df3 = model._get_reporting_aggregate_votes(df1, df2, aggregate=["c1"], estimand=estimand)
    assert pd.DataFrame({"c1": ["a", "b", "c"], f"results_{estimand}": [8, 32, 12], "reporting": [4, 4, 4]}).equals(df3)

    df2 = pd.DataFrame(
        {
            "c1": ["a", "c", "c"],
            f"results_{estimand}": [7, 19, 4],
            "reporting": [1, 1, 1],
        }
    )
    df3 = model._get_reporting_aggregate_votes(df1, df2, aggregate=["c1"], estimand=estimand)
    assert pd.DataFrame(
        {"c1": ["a", "b", "c"], f"results_{estimand}": [11.0, 16.0, 29.0], "reporting": [3.0, 2.0, 4.0]}
    ).equals(df3)

    df2 = pd.DataFrame(
        {
            "c1": ["a", "d", "d"],
            f"results_{estimand}": [7, 19, 4],
            "reporting": [1, 1, 1],
        }
    )
    df3 = model._get_reporting_aggregate_votes(df1, df2, aggregate=["c1"], estimand=estimand)
    assert pd.DataFrame(
        {"c1": ["a", "b", "c", "d"], f"results_{estimand}": [11.0, 16.0, 6.0, 23.0], "reporting": [3.0, 2.0, 2.0, 2.0]}
    ).equals(df3)

    df1 = pd.DataFrame(
        {
            "c1": ["a", "a", "a", "b", "b", "b"],
            "c2": ["x", "x", "y", "y", "z", "z"],
            f"results_{estimand}": [1, 4, 9, 1, 9, 3],
            "reporting": [1, 1, 1, 1, 1, 1],
        }
    )

    df2 = pd.DataFrame(
        {
            "c1": ["a", "b", "d"],
            "c2": ["w", "z", "t"],
            f"results_{estimand}": [5, 3, 1],
            "reporting": [1, 1, 1],
        }
    )

    df3 = model._get_reporting_aggregate_votes(df1, df2, aggregate=["c1", "c2"], estimand=estimand)
    assert pd.DataFrame(
        {
            "c1": ["a", "a", "b", "b", "a", "d"],
            "c2": ["x", "y", "y", "z", "w", "t"],
            f"results_{estimand}": [5.0, 9.0, 1.0, 15.0, 5.0, 1.0],
            "reporting": [2.0, 1.0, 1.0, 3.0, 1.0, 1.0],
        }
    ).equals(df3)

    df1 = pd.DataFrame(
        {
            "c1": ["a", "a", "b", "b"],
            "county_classification": ["x", "x", "y", "z"],
            f"results_{estimand}": [5, 3, 1, 3],
            "reporting": [1, 1, 1, 1],
        }
    )

    """
    county classification is weird one. For reporting unexpected units we don't know county classification
    so we can't aggregate over that in this case. So this test makes sure that we ignore it.
    """
    df3 = model._get_reporting_aggregate_votes(df1, None, aggregate=["c1", "county_classification"], estimand=estimand)
    assert pd.DataFrame(
        {
            "c1": ["a", "b", "b"],
            "county_classification": ["x", "y", "z"],
            f"results_{estimand}": [8, 1, 3],
            "reporting": [2, 1, 1],
        }
    ).equals(df3)


def test_aggregation_nonreporting_simple():
    """
    This test is a basic test for aggregating votes from nonreporting units.These are votes
    that have been counted from units that haven't met our reporting threshold.
    """
    model_settings = {}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    estimand = "turnout"

    df1 = pd.DataFrame(
        {
            "c1": ["a", "a", "b", "b", "c", "c"],
            f"results_{estimand}": [1, 3, 7, 9, 2, 4],
            "reporting": [0, 0, 0, 0, 0, 0],
        }
    )

    df2 = model._get_nonreporting_aggregate_votes(df1, aggregate=["c1"])

    assert pd.DataFrame({"c1": ["a", "b", "c"], f"results_{estimand}": [4, 16, 6], "reporting": [0, 0, 0]}).equals(df2)


def test_aggregation(va_governor_precinct_data):
    """
    The same test as above, just using the 2017 Virginia governor precinct results as test data.
    The data is split into two dataframes, df1 being a standin for reporting data and df2 being
    a standin for reporting unexpected data.
    """
    model_settings = {}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    estimand = "turnout"

    df = va_governor_precinct_data[
        ["postal_code", "geographic_unit_fips", "county_classification", "county_fips", "results_turnout"]
    ]

    df1 = df[:1000].copy()
    df1["reporting"] = 1
    df2 = df[1000:].copy()
    df2["reporting"] = 0

    df3 = model._get_reporting_aggregate_votes(df1, df2, aggregate=["postal_code"], estimand=estimand)
    assert 2535685.0 == df3[f"results_{estimand}"].values[0]  # total based on summing csv

    df3 = model._get_reporting_aggregate_votes(df1, df2, aggregate=["county_fips"], estimand=estimand)
    assert 10664.0 == df3[f"results_{estimand}"].values[0]  # first county based on summing csv

    # same as above. since we ignoring reporting unexpected data in this case, we just sum from d1
    df3 = model._get_reporting_aggregate_votes(df1, df2, aggregate=["county_classification"], estimand=estimand)
    assert (
        df1.groupby("county_classification").sum().reset_index(drop=False)[f"results_{estimand}"].values[0]
        == df3[f"results_{estimand}"].values[0]
    )


def test_aggregation_nonreporting(va_governor_precinct_data):
    """
    Testing nonreporting aggregation using the 2017 Virginia governor precinct results as test data.
    We set "reporting" = 0 for all units for this test. The results of the test should be summing
    all the votes to the aggregate levels (this represents votes in units that have not yet met the
    reporting threshold)
    """
    model_settings = {}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    estimand = "turnout"

    df = va_governor_precinct_data[
        ["postal_code", "geographic_unit_fips", "county_classification", "county_fips", "results_turnout"]
    ]

    df1 = df.copy()
    df1["reporting"] = 0

    df2 = model._get_nonreporting_aggregate_votes(df1, aggregate=["postal_code"])
    assert 2535685.0 == df2[f"results_{estimand}"].values[0]  # total based on summing csv

    df3 = model._get_nonreporting_aggregate_votes(df1, aggregate=["county_fips"])
    assert 10664.0 == df3[f"results_{estimand}"].values[0]  # first county based on summing csv


def test_get_aggregate_predictions_simple():
    """
    This is a basic test for our prediction aggregation. We sum the results of the reporting and reporting
    unexpected units with the predicitions from the nonreporting units. We also sum the results of the reporting
    and reporting and unexpected units with the results from the nonreporting units
    (to get the current aggregate total). All summing is done grouped by aggregate.git st
    """
    model_settings = {}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    estimand = "turnout"

    df1 = pd.DataFrame(
        {
            "c1": ["a", "a", "b", "b", "c", "c"],
            f"results_{estimand}": [1, 5, 9, 8, 9, 1],
            "reporting": [1, 1, 1, 1, 1, 1],
        }
    )

    df2 = pd.DataFrame(
        {
            "c1": ["c", "a", "a", "d", "e"],
            f"pred_{estimand}": [5, 1, 9, 1, 4],
            f"results_{estimand}": [0, 1, 0, 2, 3],
            "reporting": [0, 0, 0, 0, 0],
        }
    )

    df3 = pd.DataFrame({"c1": ["d", "f"], f"results_{estimand}": [3, 9], "reporting": [1, 1]})

    df4 = model.get_aggregate_predictions(df1, df2, df3, ["c1"], estimand)

    assert pd.DataFrame(
        {
            "c1": ["a", "b", "c", "d", "e", "f"],
            f"pred_{estimand}": [16.0, 17.0, 15.0, 4.0, 4.0, 9.0],
            f"results_{estimand}": [7.0, 17.0, 10.0, 5.0, 3.0, 9.0],
            "reporting": [2.0, 2.0, 2.0, 1.0, 0.0, 1.0],
        }
    ).equals(df4)


def test_get_aggregate_predictions(va_governor_precinct_data):
    """
    Same test as above, except again we are using the 2017 Virginia governor precinct data.
    Here the first dataframe is a standin for reporting units, the second data frame is a standin for
    nonreporting units and the third dataframe is a standing for reporting unexpected units.
    """
    model_settings = {}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    estimand = "turnout"

    df = va_governor_precinct_data[
        ["postal_code", "geographic_unit_fips", "county_classification", "county_fips", f"results_{estimand}"]
    ]

    df1 = df[:1000].copy()
    df1["reporting"] = 1
    df2 = df[1000:2000].rename(columns={f"results_{estimand}": f"pred_{estimand}"}).copy()
    df2["reporting"] = 0
    df2[f"results_{estimand}"] = 0
    df3 = df[2000:].copy()
    df3["reporting"] = 1

    df4 = model.get_aggregate_predictions(df1, df2, df3, ["postal_code"], estimand)
    assert 2535685.0 == df4[f"pred_{estimand}"].values[0]  # total based on summing csv

    df4 = model.get_aggregate_predictions(df1, df2, df3, ["county_fips"], estimand)
    assert 10664.0 == df4[f"pred_{estimand}"].values[0]  # first county based on summing csv

    # again we need to leave out reporting unexpected units for county_classification aggregation
    df4 = model.get_aggregate_predictions(df1, df2, df3, ["county_classification"], estimand)
    assert (
        df1.groupby("county_classification").sum().reset_index(drop=False)[f"results_{estimand}"].values[0]
        + df2.groupby("county_classification").sum().reset_index(drop=False)[f"pred_{estimand}"].values[0]
        == df4[f"pred_{estimand}"].values[0]
    )


def test_compute_lambda_under_one():
    """
    Test/view computing lambda
    """
    lambda_ = [0.01, 0.05, 0.99, 0.56]
    model_settings = {"features": ["b"]}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    df_X = pd.DataFrame(
        {
            "residuals_a": [1, 2, 3, 4],
            "total_voters_a": [4, 2, 9, 5],
            "last_election_results_a": [5, 1, 4, 2],
            "results_a": [5, 2, 8, 0],
            "results_b": [0, 6, 2, 1],
            "baseline_a": [9, 2, 4, 5],
            "baseline_b": [9, 2, 4, 5],
            "a": [2, 2, 3, 7],
            "b": [2, 3, 4, 5],
        }
    )

    new_lambda, avg_MAPE = model.compute_lambda(df_X, lambda_, "a")

    assert new_lambda == 0.01
    assert avg_MAPE == 0.625  # value checked by hand


def test_compute_lambda_over_one():
    """
    Test/view computing lambda
    """
    lambda_ = [4, 7, 21, 3, 0, 1, 0.5]
    model_settings = {"features": ["b"]}
    model = BaseElectionModel.BaseElectionModel(model_settings)
    df_X = pd.DataFrame(
        {
            "residuals_a": [1, 2, 3, 4],
            "total_voters_a": [4, 2, 9, 5],
            "last_election_results_a": [5, 1, 4, 2],
            "results_a": [5, 2, 8, 0],
            "results_b": [0, 6, 2, 1],
            "baseline_a": [9, 2, 4, 5],
            "baseline_b": [9, 2, 4, 5],
            "a": [2, 2, 3, 7],
            "b": [2, 3, 4, 5],
        }
    )

    new_lambda, avg_MAPE = model.compute_lambda(df_X, lambda_, "a")

    assert new_lambda == 0
    assert avg_MAPE == 0.35714285714285715
