import logging

import numpy as np
import pandas as pd
import pytest

from elexmodel.client import ModelNotEnoughSubunitsException
from elexmodel.handlers.config import ConfigHandler
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.logging import initialize_logging
from elexmodel.utils import math_utils

initialize_logging()

# A set of valid model parameters
office = "G"
estimands = ["turnout", "dem"]
geographic_unit_type = "precinct"
features = ["gender_f", "median_household_income"]
aggregates = ["postal_code", "county_fips"]
fixed_effects = []
pi_method = "gaussian"
model_parameters = {"beta": 3, "winsorize": False, "robust": True, "lambda_": 0, "y_LB": 1.0, "y_UB": 4}
handle_unreporting = "drop"


def test_check_input_parameters(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    assert model_client._check_input_parameters(
        config_handler,
        office,
        estimands,
        geographic_unit_type,
        features,
        aggregates,
        fixed_effects,
        pi_method,
        model_parameters,
        handle_unreporting,
    )


def test_check_input_parameters_bootstrap(model_client, va_config):
    # this is to test y_LB and y_UB
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    assert model_client._check_input_parameters(
        config_handler,
        office,
        estimands,
        geographic_unit_type,
        features,
        aggregates,
        fixed_effects,
        "bootstrap",
        model_parameters,
        handle_unreporting,
    )


def test_check_input_parameters_office(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            "O",
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            pi_method,
            model_parameters,
            handle_unreporting,
        )


def test_check_input_parameters_pi_method(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            "bad_pi_method",
            model_parameters,
            handle_unreporting,
        )


def test_check_input_parameters_estimand(caplog, model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with caplog.at_level(logging.INFO):
        model_client._check_input_parameters(
            config_handler,
            office,
            ["foo"],
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            pi_method,
            model_parameters,
            handle_unreporting,
        )

    assert "Found additional estimands " in caplog.text


def test_check_input_parameters_geographic_unit_type(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            "geographic_unit_type",
            features,
            aggregates,
            fixed_effects,
            pi_method,
            model_parameters,
            handle_unreporting,
        )


def test_check_input_parameters_features(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            ["gender_f", "bad_feature1", "bad_feature2"],
            aggregates,
            fixed_effects,
            pi_method,
            model_parameters,
            handle_unreporting,
        )


def test_check_input_parameters_aggregates(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            ["bad_aggregate_1", "postal_code", "bad_aggregate2"],
            fixed_effects,
            pi_method,
            model_parameters,
            handle_unreporting,
        )


def test_check_input_parameters_fixed_effect_list(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            ["bad_fixed_effect"],
            pi_method,
            model_parameters,
            handle_unreporting,
        )


def test_check_input_parameters_fixed_effect_dict(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            {"bad_fixed_effect": ["a", "b"]},
            pi_method,
            model_parameters,
            handle_unreporting,
        )


def test_check_input_parameters_beta(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            pi_method,
            {
                "beta": "bad_beta",
                "winsorize": False,
                "robust": True,
                "lambda_": 0,
            },
            handle_unreporting,
        )


def test_check_input_parameters_winsorize(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            pi_method,
            {
                "beta": 3,
                "winsorize": "bad_winsorize",
                "robust": True,
                "lambda_": 0,
            },
            handle_unreporting,
        )


def test_check_input_parameters_robust(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            "nonparametric",
            {
                "beta": 3,
                "winsorize": False,
                "robust": "bad_robust",
                "lambda_": 0,
            },
            handle_unreporting,
        )


def test_check_input_parameters_lambda_(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            pi_method,
            {
                "beta": 3,
                "winsorize": False,
                "robust": True,
                "lambda_": -1,
            },
            handle_unreporting,
        )


def test_check_input_parameters_y_UB_LB(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            "bootstrap",
            {"beta": 3, "winsorize": False, "robust": True, "lambda_": -1, "y_LB": "break", "y_UB": 1},
            handle_unreporting,
        )


def test_check_input_parameters_handle_unreporting(model_client, va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    with pytest.raises(ValueError):
        model_client._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            pi_method,
            model_parameters,
            "bad_handle_unreporting",
        )


def test_get_aggregate_list(model_client):
    assert model_client.get_aggregate_list("P", "county_fips") == ["postal_code", "county_fips"]
    assert model_client.get_aggregate_list("P", "postal_code") == ["postal_code"]
    assert model_client.get_aggregate_list("P", "district") == ["postal_code", "district"]

    assert model_client.get_aggregate_list("H", "county_fips") == ["postal_code", "district", "county_fips"]
    assert model_client.get_aggregate_list("H", "postal_code") == ["postal_code", "district"]
    assert model_client.get_aggregate_list("H", "district") == ["postal_code", "district"]


def test_compute_evaluation(historical_model_client):
    random_number_generator = np.random.RandomState(42)
    raw_results = random_number_generator.randint(low=0, high=10, size=6)
    estimand = "turnout"
    pred = random_number_generator.randint(low=0, high=10, size=6)
    lower = random_number_generator.randint(low=0, high=10, size=6)
    upper = random_number_generator.randint(low=0, high=10, size=6)

    estimates = pd.DataFrame(
        {
            "c1": ["a", "a", "b", "b", "c", "c"],
            "c2": ["x", "y", "y", "z", "x", "z"],
            f"pred_{estimand}": pred.tolist(),
            f"lower_0.7_{estimand}": lower.tolist(),
            f"upper_0.7_{estimand}": upper.tolist(),
            f"lower_0.9_{estimand}": (lower - 1).tolist(),
            f"upper_0.9_{estimand}": (upper + 1).tolist(),
        }
    )
    results = pd.DataFrame(
        {
            "c1": ["a", "a", "b", "b", "c", "c"],
            "c2": ["x", "y", "y", "z", "x", "z"],
            f"raw_results_{estimand}": raw_results.tolist(),
        }
    )

    eval_ = historical_model_client.compute_evaluation(estimates, results, ["c1", "c2"], ["c1"], [0.7, 0.9], estimand)

    # assumes individual functions work. Tests for them are above.
    assert eval_["a"][f"mae_{estimand}"] == math_utils.compute_error(raw_results[:2], pred[:2], type_="mae")
    assert eval_["b"][f"mae_{estimand}"] == math_utils.compute_error(raw_results[2:4], pred[2:4], type_="mae")
    assert eval_["c"][f"mae_{estimand}"] == math_utils.compute_error(raw_results[4:], pred[4:], type_="mae")

    assert eval_["a"][f"mape_{estimand}"] == math_utils.compute_error(raw_results[:2], pred[:2], type_="mape")
    assert eval_["b"][f"mape_{estimand}"] == math_utils.compute_error(raw_results[2:4], pred[2:4], type_="mape")
    assert eval_["c"][f"mape_{estimand}"] == math_utils.compute_error(raw_results[4:], pred[4:], type_="mape")

    assert eval_["a"][f"frac_within_pi_0.7_{estimand}"] == math_utils.compute_frac_within_pi(
        lower[:2], upper[:2], pred[:2]
    )
    assert eval_["b"][f"frac_within_pi_0.7_{estimand}"] == math_utils.compute_frac_within_pi(
        lower[2:4], upper[2:4], pred[2:4]
    )
    assert eval_["c"][f"frac_within_pi_0.7_{estimand}"] == math_utils.compute_frac_within_pi(
        lower[4:], upper[4:], pred[4:]
    )

    assert eval_["a"][f"mean_pi_length_0.7_{estimand}"] == math_utils.compute_mean_pi_length(
        lower[:2], upper[:2], raw_results[:2]
    )
    assert eval_["b"][f"mean_pi_length_0.7_{estimand}"] == math_utils.compute_mean_pi_length(
        lower[2:4], upper[2:4], raw_results[2:4]
    )
    assert eval_["c"][f"mean_pi_length_0.7_{estimand}"] == math_utils.compute_mean_pi_length(
        lower[4:], upper[4:], raw_results[4:]
    )

    eval_ = historical_model_client.compute_evaluation(
        estimates, results, ["c1", "c2"], lambda x: True, [0.7, 0.9], estimand
    )

    assert eval_[True][f"mae_{estimand}"] == math_utils.compute_error(raw_results, pred, type_="mae")
    assert eval_[True][f"mape_{estimand}"] == math_utils.compute_error(raw_results, pred, type_="mape")
    assert eval_[True][f"frac_within_pi_0.7_{estimand}"] == math_utils.compute_frac_within_pi(lower, upper, pred)
    assert eval_[True][f"mean_pi_length_0.7_{estimand}"] == math_utils.compute_mean_pi_length(lower, upper, raw_results)


def test_get_estimates_fully_reporting(model_client, va_governor_county_data, va_config):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(100)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    result = model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        save_output=[],
    )

    assert result["state_data"].shape == (1, 6)
    assert result["unit_data"].shape == (133, 8)

    assert list(result["state_data"].columns.values) == [
        "postal_code",
        "pred_turnout",
        "results_turnout",
        "reporting",
        "lower_0.9_turnout",
        "upper_0.9_turnout",
    ]
    assert list(result["unit_data"].columns.values) == [
        "postal_code",
        "geographic_unit_fips",
        "pred_turnout",
        "reporting",
        "unit_category",
        "lower_0.9_turnout",
        "upper_0.9_turnout",
        "results_turnout",
    ]

    assert result["state_data"]["postal_code"][0] == "VA"
    assert result["state_data"]["pred_turnout"][0] == 2614065.0
    assert result["state_data"]["results_turnout"][0] == 2614065.0
    assert result["state_data"]["reporting"][0] == 133.0
    assert result["state_data"]["lower_0.9_turnout"][0] == 2614065.0
    assert result["state_data"]["upper_0.9_turnout"][0] == 2614065.0


def test_not_including_unit_data(model_client, va_assembly_precinct_data, va_config):
    election_id = "2017-11-07_VA_G"
    office_id = "Y"
    geographic_unit_type = "precinct-district"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100
    aggregates = ["postal_code", "district"]

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_assembly_precinct_data
    )
    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(100)

    preprocessed_data = va_assembly_precinct_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    result = model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        aggregates=aggregates,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        save_output=[],
    )
    assert "unit_data" not in result.keys()


def test_unexpected_units_no_new_units(model_client, va_governor_precinct_data, va_config):
    # verifies that adding unexpected units doesn't add any extra units when not expected to
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "precinct"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100
    aggregates = ["county_fips"]

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_precinct_data, unexpected_units=5
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(100)

    preprocessed_data = va_governor_precinct_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    result = model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        aggregates=aggregates,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        save_output=[],
    )
    va_counties_count = va_governor_precinct_data[["county_fips"]].drop_duplicates().shape[0]
    assert result["county_data"].shape[0] == va_counties_count


def test_unexpected_units_new_units(model_client, va_governor_county_data, va_config):
    # verifies that adding unexpected units DOES add any extra units when expected to
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100
    aggregates = ["county_fips"]
    unexpected_units = 5

    data_handler = MockLiveDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimands,
        data=va_governor_county_data,
        unexpected_units=unexpected_units,
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(100)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    result = model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        aggregates=aggregates,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        save_output=[],
    )
    va_counties_count = va_governor_county_data[["county_fips"]].drop_duplicates().shape[0]
    assert result["county_data"].shape[0] == va_counties_count + unexpected_units


def test_get_estimates_some_reporting(model_client, va_governor_county_data, va_config):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle(seed=5)
    data = data_handler.get_percent_fully_reported(70)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    result = model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        save_output=[],
    )
    assert result["state_data"].shape == (1, 6)
    assert result["unit_data"].shape == (133, 8)

    assert list(result["state_data"].columns.values) == [
        "postal_code",
        "pred_turnout",
        "results_turnout",
        "reporting",
        "lower_0.9_turnout",
        "upper_0.9_turnout",
    ]
    assert list(result["unit_data"].columns.values) == [
        "postal_code",
        "geographic_unit_fips",
        "pred_turnout",
        "reporting",
        "unit_category",
        "lower_0.9_turnout",
        "upper_0.9_turnout",
        "results_turnout",
    ]
    assert result["state_data"]["postal_code"][0] == "VA"
    assert result["state_data"]["pred_turnout"][0] == 2587563.0
    assert result["state_data"]["results_turnout"][0] == 1570077.0
    assert result["state_data"]["reporting"][0] == 94.0
    assert result["state_data"]["lower_0.9_turnout"][0] == 2443849.0
    assert result["state_data"]["upper_0.9_turnout"][0] == 2683348.0


def test_get_estimates_no_subunits_reporting(model_client, va_governor_county_data, va_config):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(0)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    with pytest.raises(ModelNotEnoughSubunitsException, match="Currently 0 reporting, need at least 20"):
        model_client.get_estimates(
            data,
            election_id,
            office_id,
            estimands,
            prediction_intervals,
            percent_reporting_threshold,
            geographic_unit_type,
            raw_config=va_config,
            preprocessed_data=preprocessed_data,
            save_output=[],
        )


def test_get_estimates_not_enough_subunits_reporting(model_client, va_governor_county_data, va_config):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(10)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    with pytest.raises(ModelNotEnoughSubunitsException, match="Currently 14 reporting, need at least 20"):
        model_client.get_estimates(
            data,
            election_id,
            office_id,
            estimands,
            prediction_intervals,
            percent_reporting_threshold,
            geographic_unit_type,
            raw_config=va_config,
            preprocessed_data=preprocessed_data,
            save_output=[],
        )


def test_conformalization_data(model_client, va_governor_county_data, va_config):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(70)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        pi_method="gaussian",
        save_output=[],
    )

    conform_unit = model_client.all_conformalization_data_unit_dict
    conform_agg = model_client.all_conformalization_data_agg_dict

    assert len(conform_unit) == 1
    assert len(conform_agg) == 1
    assert len(conform_unit[0.9]) == 1
    assert len(conform_agg[0.9]) == 1
    assert list(conform_unit[0.9].keys()) == ["turnout"]
    assert list(conform_agg[0.9].keys()) == ["turnout"]
    assert isinstance(conform_unit[0.9]["turnout"][0], pd.DataFrame)
    assert isinstance(conform_unit[0.9]["turnout"][1], pd.DataFrame)
    assert isinstance(conform_agg[0.9]["turnout"][0], pd.DataFrame)
    assert isinstance(conform_agg[0.9]["turnout"][1], pd.DataFrame)

    model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        pi_method="nonparametric",
        save_output=[],
    )

    conform_unit = model_client.all_conformalization_data_unit_dict
    conform_agg = model_client.all_conformalization_data_agg_dict

    assert len(conform_unit) == 1
    assert len(conform_agg) == 1
    assert len(conform_unit[0.9]) == 1
    assert len(conform_agg[0.9]) == 1
    assert list(conform_unit[0.9].keys()) == ["turnout"]
    assert list(conform_agg[0.9].keys()) == ["turnout"]
    assert conform_unit[0.9]["turnout"][0] is None
    assert isinstance(conform_unit[0.9]["turnout"][1], pd.DataFrame)
    assert conform_agg[0.9]["turnout"][0] is None
    assert isinstance(conform_agg[0.9]["turnout"][1], pd.DataFrame)


def test_estimandizer_input(model_client, va_governor_county_data, va_config):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout", "party_vote_share_dem"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(100)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1
    try:
        model_client.get_estimates(
            data,
            election_id,
            office_id,
            estimands,
            prediction_intervals,
            percent_reporting_threshold,
            geographic_unit_type,
            raw_config=va_config,
            preprocessed_data=preprocessed_data,
            save_output=[],
        )
    except KeyError:
        pytest.raises("Error with client input for estimandizer")


def test_get_national_summary_votes_estimates(model_client, va_governor_county_data, va_config):
    expected = {"margin": [1.0, 1.0, 1.0]}
    expected_df = pd.DataFrame.from_dict(expected, orient="index", columns=["agg_pred", "lower_0.99", "upper_0.99"])
    expected_df.index.name = "estimand"
    expected_df = expected_df.reset_index()

    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["margin"]
    prediction_intervals = [0.9]
    percent_reporting_threshold = 100
    kwargs = {"pi_method": "bootstrap", "features": ["baseline_normalized_margin"], "national_summary": True}

    data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(100)

    preprocessed_data = va_governor_county_data.copy()
    preprocessed_data["last_election_results_turnout"] = preprocessed_data["baseline_turnout"].copy() + 1

    model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        raw_config=va_config,
        preprocessed_data=preprocessed_data,
        save_output=[],
        **kwargs,
    )

    current = model_client.get_national_summary_votes_estimates(None, 0, [0.99])

    pd.testing.assert_frame_equal(current, model_client.results_handler.final_results["nat_sum_data"])
    pd.testing.assert_frame_equal(expected_df, model_client.results_handler.final_results["nat_sum_data"])
