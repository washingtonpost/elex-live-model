import numpy as np
import pandas as pd

from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def test_load(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    estimand_baselines = {"turnout": "turnout"}
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands=["turnout"], data=va_governor_county_data
    )
    current_data = live_data_handler.data
    preprocessed_data_handler = PreprocessedDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimand_baselines=estimand_baselines,
        estimands=estimands,
        data=va_governor_county_data,
    )
    preprocessed_data = preprocessed_data_handler.data

    combined_data_handler = CombinedDataHandler(
        preprocessed_data, current_data, estimands, "county", handle_unreporting="drop"
    )

    assert combined_data_handler.data.shape == (133, 33)


def test_zero_unreporting_missing_single_estimand_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimand_baselines = {"turnout": "turnout", "dem": "dem"}
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimand_baselines=estimand_baselines,
        estimands=estimands,
        data=va_governor_county_data,
    )
    preprocessed_data = preprocessed_data_handler.data

    combined_data_handler = CombinedDataHandler(
        preprocessed_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0  # value with na result has been set to zero
    assert combined_data_handler.data["results_turnout"].iloc[0] != 0  # has not been set to zero
    assert (
        combined_data_handler.data["percent_expected_vote"].iloc[0] == 0
    )  # percent expected vote with na result has been set to zero
    assert combined_data_handler.data.shape == (133, 36)  # didn't drop any
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_missing_multiple_estimands_value(va_governor_county_data):
    """
    Set the value for multiple estimands (dem, turnout) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimand_baselines = {"turnout": "turnout", "dem": "dem"}
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan
    current_data.loc[0, "results_turnout"] = np.nan

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimand_baselines=estimand_baselines,
        estimands=estimands,
        data=va_governor_county_data,
    )
    preprocessed_data = preprocessed_data_handler.data

    combined_data_handler = CombinedDataHandler(
        preprocessed_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0
    assert combined_data_handler.data["results_turnout"].iloc[0] == 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 36)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero
    assert combined_data_handler.data["results_turnout"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_missing_percent_expected_vote_value(va_governor_county_data):
    """
    Set the value and percent reporting for one estimand (dem) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimand_baselines = {"turnout": "turnout", "dem": "dem"}
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "percent_expected_vote"] = np.nan
    current_data.loc[0, "results_dem"] = np.nan

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimand_baselines=estimand_baselines,
        estimands=estimands,
        data=va_governor_county_data,
    )
    preprocessed_data = preprocessed_data_handler.data

    combined_data_handler = CombinedDataHandler(
        preprocessed_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 36)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_random_percent_expected_vote_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimand_baselines = {"turnout": "turnout", "dem": "dem"}
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = np.random.randint(1, 100, current_data.shape[0])
    current_data.loc[0, "results_dem"] = np.nan

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimand_baselines=estimand_baselines,
        estimands=estimands,
        data=va_governor_county_data,
    )
    preprocessed_data = preprocessed_data_handler.data

    combined_data_handler = CombinedDataHandler(
        preprocessed_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0  # all values set to 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 36)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_drop_unreporting_missing_single_estimand_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "drop"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimand_baselines = {"turnout": "turnout", "dem": "dem"}
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimand_baselines=estimand_baselines,
        estimands=estimands,
        data=va_governor_county_data,
    )
    preprocessed_data = preprocessed_data_handler.data

    combined_data_handler = CombinedDataHandler(
        preprocessed_data, current_data, estimands, "county", handle_unreporting="drop"
    )
    assert combined_data_handler.data.shape == (132, 36)  # dropped one
    assert combined_data_handler.data["results_dem"].iloc[0] != 0  # didn't accidentally set other to zero


def test_get_reporting_data(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    estimand_baseline = {"turnout": "turnout"}

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=20)
    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_governor_county_data
    )

    # no fixed effects
    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data, current_data, estimands, geographic_unit_type
    )
    observed_data = combined_data_handler.get_reporting_units(100)
    assert observed_data.shape[0] == 20
    assert observed_data.reporting.iloc[0] == 1
    assert observed_data.reporting.sum() == 20


def test_get_unexpected_units_county_district(va_assembly_county_data):
    election_id = "2017-11-07_VA_G"
    office = "Y"
    geographic_unit_type = "county-district"
    estimands = ["turnout"]
    unexpected_units = 5
    estimand_baseline = {"turnout": "turnout"}

    live_data_handler = MockLiveDataHandler(
        election_id,
        office,
        geographic_unit_type,
        estimands,
        data=va_assembly_county_data,
        unexpected_units=unexpected_units,
    )
    current_data = live_data_handler.get_n_fully_reported(n=20)
    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_assembly_county_data
    )

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data, current_data, estimands, geographic_unit_type
    )
    unexpected_data = combined_data_handler.get_unexpected_units(100, ["county_fips", "district"])
    assert unexpected_data.shape[0] == unexpected_units
    assert unexpected_data[unexpected_data.county_fips == ""].shape[0] == 0
    assert unexpected_data["county_fips"].map(lambda x: len(x) == 6).all()
    assert unexpected_data[unexpected_data.district == ""].shape[0] == 0
    assert unexpected_data["district"].map(lambda x: len(x) < 6).all()


def test_get_unexpected_units_county(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    reporting_unexpected_units = 5
    estimand_baseline = {"turnout": "turnout"}

    live_data_handler = MockLiveDataHandler(
        election_id,
        office,
        geographic_unit_type,
        estimands,
        data=va_governor_county_data,
        unexpected_units=reporting_unexpected_units,
    )
    current_data = live_data_handler.get_n_fully_reported(n=20)
    # Add an additional row of a nonreporting unexpected unit that we test below
    extra_row = current_data[current_data["geographic_unit_fips"].map(lambda x: len(x)) == 5].head(1)
    extra_row["geographic_unit_fips"] = extra_row["geographic_unit_fips"] + "1"
    extra_row["percent_expected_vote"] = 50
    current_data = pd.concat([current_data, extra_row])

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_governor_county_data
    )

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data, current_data, estimands, geographic_unit_type
    )
    unexpected_data = combined_data_handler.get_unexpected_units(100, ["county_fips"])
    assert unexpected_data.shape[0] == reporting_unexpected_units + 1
    assert unexpected_data[unexpected_data.county_fips == ""].shape[0] == 0
    assert unexpected_data["county_fips"].map(lambda x: len(x) == 6).all()
    # test that nonreporting unexpected unit is captured here
    assert unexpected_data[unexpected_data.percent_expected_vote == 50].shape[0] == 1
