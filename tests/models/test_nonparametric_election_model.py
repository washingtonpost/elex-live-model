import numpy as np
import pandas as pd

from elexmodel.models import NonparametricElectionModel

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
    model = NonparametricElectionModel.NonparametricElectionModel()
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

    intervals = model.get_aggregate_prediction_intervals(df1, df3, df2, ["c1"], alpha, None, estimand, model_settings)

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

    intervals = model.get_aggregate_prediction_intervals(
        df1, df2, df3, ["postal_code"], alpha, None, estimand, model_settings
    )
    assert 2535685.0 == intervals.lower[0]  # total based on summing csv
    assert 2535685.0 == intervals.upper[0]  # total based on summing csv

    intervals = model.get_aggregate_prediction_intervals(
        df1, df2, df3, ["county_fips"], alpha, None, estimand, model_settings
    )
    assert 10664.0 == intervals.lower[0]  # first county based on summing csv
    assert 10664.0 == intervals.upper[0]  # first county based on summing csv
