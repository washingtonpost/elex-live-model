import numpy as np
import pandas as pd
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

from elexmodel.models import NonparametricElectionModel
from elexmodel.models.ConformalElectionModel import PredictionIntervals

TOL = 1e-5


def test_instantiation():
    model_settings = {}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings=model_settings)

    assert not model.robust

    model_settings = {"robust": False}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings=model_settings)

    assert not model.robust

    model_settings = {"robust": True}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings=model_settings)

    assert model.robust


def test_compute_conf_frac():
    model_settings = {"robust": False}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)

    n_reporting_units = 200
    alpha = 0.95
    conf_frac = model._compute_conf_frac(n_reporting_units, alpha)
    assert conf_frac == 0.81

    n_reporting_units = 1000
    conf_frac = model._compute_conf_frac(n_reporting_units, alpha)
    assert conf_frac == 0.90


def test_get_minimum_reporting_units():
    model_settings = {}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
    n_min = model.get_minimum_reporting_units(0.7)

    assert n_min == 6

    n_min = model.get_minimum_reporting_units(0.9)

    assert n_min == 20


def test_aggregate_prediction_intervals_simple():
    """
    This is a basic test for the non parametric aggregation of prediction intervals.
    Which is just summing the prediction intervals. This means adding the results in reporting
    and reporting unexpected units to lower and upper prediction intervals for nonreporting units.
    """
    model_settings = {"robust": False}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
    alpha = 0.7
    estimand = "turnout"

    df1 = pd.DataFrame(
        {
            "c1": ["a", "a", "a", "b", "b", "c"],
            f"results_{estimand}": [5, 3, 1, 9, 2, 1],
            "reporting": [1, 1, 1, 1, 1, 1],
        }
    )  # a:9, b: 11, c: 1

    df2 = pd.DataFrame(
        {
            "c1": ["b", "d", "d"],
            f"results_{estimand}": [8, 4, 2],
            "reporting": [1, 1, 1],
        }
    )  # b: 8, d: 6

    df3 = pd.DataFrame(
        {
            "c1": ["a", "a", "c", "c", "e", "e"],
            f"lower_{alpha}_{estimand}": [2, 1, 3, 4, 1, 2],  # a: 3, c: 7, e: 3
            f"upper_{alpha}_{estimand}": [9, 8, 7, 9, 5, 4],  # a: 17, c: 16, e: 9
        }
    )

    prediction_intervals = PredictionIntervals([], [], [])
    intervals = model.get_aggregate_prediction_intervals(df1, df3, df2, ["c1"], alpha, prediction_intervals, estimand)

    assert np.array_equal(np.asarray([12, 19, 8, 6, 3]), intervals.lower)
    assert np.array_equal(np.asarray([26, 19, 17, 6, 9]), intervals.upper)


def test_aggregate_prediction_intervals(va_governor_precinct_data):
    """
    Same as above, just using 2017 Virginia governor precinct data.
    The first dataframe is reporting data, the second dataframe is nonreporting data and the third
    dataframe is reporting unexpected data.
    """
    model_settings = {"robust": False}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
    alpha = 0.9
    estimand = "turnout"

    df = va_governor_precinct_data[
        ["postal_code", "geographic_unit_fips", "county_classification", "county_fips", "results_turnout"]
    ]

    df1 = df[:1000].copy()
    df1["reporting"] = 1
    df2 = df[1000:2000].copy()
    df2["reporting"] = 0
    df2[f"lower_{alpha}_{estimand}"] = df2[f"results_{estimand}"]
    df2[f"upper_{alpha}_{estimand}"] = df2[f"results_{estimand}"]
    df3 = df[2000:].copy()
    df3["reporting"] = 1

    prediction_intervals = PredictionIntervals([], [], [])

    intervals = model.get_aggregate_prediction_intervals(
        df1, df2, df3, ["postal_code"], alpha, prediction_intervals, estimand
    )
    assert 2535685.0 == intervals.lower[0]  # total based on summing csv
    assert 2535685.0 == intervals.upper[0]  # total based on summing csv

    intervals = model.get_aggregate_prediction_intervals(
        df1, df2, df3, ["county_fips"], alpha, prediction_intervals, estimand
    )
    assert 10664.0 == intervals.lower[0]  # first county based on summing csv
    assert 10664.0 == intervals.upper[0]  # first county based on summing csv


####
# The functions that are tested below are implemented in BaseElectionModel
# which is an abstract base class
####


def test_aggregation_simple():
    """
    This test is a basic test for aggregating reporting votes. We have have two data frames, reporting votes and
    reporting unexpected votes (when we didn't have a geographic unit in the preprocessed data) and we want to make
    sure that we are summing them together correctly. So both dataframes have a `results` Series that need to be
    summed based on the aggregate.
    """
    model_settings = {}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
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
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
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
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
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
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
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
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
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
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
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


####
# The functions that are tested below are implemented in ConformalElectionModel
# which is an abstract base class
####


def test_fit_model():
    """
    Test fitting the model.
    """
    model_settings = {}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
    qr = QuantileRegressionSolver()

    df_X = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 1, 1, 2]})

    df_y = pd.DataFrame({"y": [3, 8, 9, 15]}).y
    weights = pd.DataFrame({"weights": [1, 1, 1, 1]}).weights
    model.fit_model(qr, df_X, df_y, 0.5, weights, True)

    np.testing.assert_allclose(qr.predict(df_X), [[8, 8, 8, 15]], rtol=TOL)
    np.testing.assert_allclose(qr.coefficients, [[1, 7]], rtol=TOL)


def test_get_unit_predictions():
    model_settings = {"lambda_": 1, "features": ["b"]}
    model = NonparametricElectionModel.NonparametricElectionModel(model_settings)
    df_X = pd.DataFrame(
        {
            "residuals_a": [1, 2, 3, 4],
            "total_voters_a": [4, 2, 9, 5],
            "last_election_results_a": [5, 1, 4, 2],
            "results_a": [0, 0, 0, 1],
            "b": [2, 3, 4, 5],
        }
    )
    model.get_unit_predictions(df_X, df_X, estimand="a")

    "intercept" in model.features_to_coefficients
    "b" in model.features_to_coefficients
    model.features_to_coefficients["intercept"] > 0
