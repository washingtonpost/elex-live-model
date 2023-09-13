import pandas as pd
import pytest

TOL = 1e-3


def test_get_minimal_reporting_units(base_election_model):
    minimal_units = base_election_model.get_minimum_reporting_units(0.1)
    assert minimal_units == 10


def test_get_unit_predictions(base_election_model):
    with pytest.raises(NotImplementedError):
        base_election_model.get_unit_predictions(pd.DataFrame(), pd.DataFrame(), "test")


def test_get_unit_prediction_intervals(base_election_model):
    with pytest.raises(NotImplementedError):
        base_election_model.get_unit_prediction_intervals(pd.DataFrame(), pd.DataFrame(), 0.1, "test")


def test_get_aggregate_prediction_intervals(base_election_model):
    with pytest.raises(NotImplementedError):
        base_election_model.get_aggregate_prediction_intervals(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], 0.1)


def test_get_coefficients(base_election_model):
    coefficients = base_election_model.get_coefficients()
    assert isinstance(coefficients, dict)
    assert len(coefficients) == 0
