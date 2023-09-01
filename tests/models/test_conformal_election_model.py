import numpy as np
import pandas as pd
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

from elexmodel.models import ConformalElectionModel

TOL = 1e-3


def test_fit_model():
    """
    Test fitting the model.
    """
    model_settings = {}
    model = ConformalElectionModel.ConformalElectionModel(model_settings)
    qr = QuantileRegressionSolver(solver="ECOS")

    df_X = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 1, 1, 2]})

    df_y = pd.DataFrame({"y": [3, 8, 9, 15]}).y
    weights = pd.DataFrame({"weights": [1, 1, 1, 1]}).weights
    model.fit_model(qr, df_X, df_y, 0.5, weights, True)

    assert all(np.abs(qr.predict(df_X) - [8, 8, 8, 15]) <= TOL)
    assert all(np.abs(qr.coefficients - [1, 7]) <= TOL)


def test_get_unit_predictions():
    model_settings = {"lambda_": 1, "features": ["b"]}
    model = ConformalElectionModel.ConformalElectionModel(model_settings)
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
