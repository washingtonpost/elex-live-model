import numpy as np
import pandas as pd

from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def test_load(va_governor_county_data):
    estimands = ["turnout"]
    live_data_handler = MockLiveDataHandler(
        "2017-11-07_VA_G", "G", "county", estimands=["turnout"], data=va_governor_county_data
    )
    current_data = live_data_handler.data

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", fixed_effects=[], handle_unreporting="drop"
    )
    assert combined_data_handler.data.shape == (133, 29)


def test_zero_unreporting_missing_single_estimand_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "zero"
    """
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler("2017-11-07_VA_G", "G", "county", estimands, data=va_governor_county_data)
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0  # value with na result has been set to zero
    assert combined_data_handler.data["results_turnout"].iloc[0] != 0  # has not been set to zero
    assert (
        combined_data_handler.data["percent_expected_vote"].iloc[0] == 0
    )  # percent expected vote with na result has been set to zero
    assert combined_data_handler.data.shape == (133, 31)  # didn't drop any
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_missing_multiple_estimands_value(va_governor_county_data):
    """
    Set the value for multiple estimands (dem, turnout) as na to test unreporting = "zero"
    """
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler("2017-11-07_VA_G", "G", "county", estimands, data=va_governor_county_data)
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan
    current_data.loc[0, "results_turnout"] = np.nan

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0
    assert combined_data_handler.data["results_turnout"].iloc[0] == 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 31)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero
    assert combined_data_handler.data["results_turnout"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_missing_percent_expected_vote_value(va_governor_county_data):
    """
    Set the value and percent reporting for one estimand (dem) as na to test unreporting = "zero"
    """
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler("2017-11-07_VA_G", "G", "county", estimands, data=va_governor_county_data)
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "percent_expected_vote"] = np.nan
    current_data.loc[0, "results_dem"] = np.nan

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 31)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_random_percent_expected_vote_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "zero"
    """
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler("2017-11-07_VA_G", "G", "county", estimands, data=va_governor_county_data)
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = np.random.randint(1, 100, current_data.shape[0])
    current_data.loc[0, "results_dem"] = np.nan

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0  # all values set to 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 31)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_drop_unreporting_missing_single_estimand_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "drop"
    """
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler("2017-11-07_VA_G", "G", "county", estimands, data=va_governor_county_data)
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="drop"
    )
    assert combined_data_handler.data.shape == (132, 31)  # dropped one
    assert combined_data_handler.data["results_dem"].iloc[0] != 0  # didn't accidentally set other to zero


def test_generate_fixed_effects(va_governor_county_data):
    """
    We test adding one or two fixed effects. This test assumes that all units are reporting,
    and therefore all fixed effect categories will exist in reporting_units. That means that
    all fixed effect categories get added to nonreporting_units.
    """
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    estimand_baseline = {"turnout": "turnout"}
    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=va_governor_county_data.shape[0])

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_governor_county_data
    )

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data,
        current_data,
        estimands,
        "county",
        fixed_effects=["county_classification"],
        handle_unreporting="drop",
    )

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)

    assert combined_data_handler.data.shape == (133, 33)

    n_expected_columns = combined_data_handler.data.shape[1] + 3  # residual intercept and reporting
    n_expected_columns += 5  # 6 - 1 the fixed effects with one dropped
    assert reporting_data.shape == (133, n_expected_columns)
    assert nonreporting_data.shape == (0, n_expected_columns)

    assert "county_classification_nova" in reporting_data.columns
    assert "county_classification_nova" in nonreporting_data.columns

    assert "county_classification" in combined_data_handler.fixed_effects
    assert len(combined_data_handler.expanded_fixed_effects) == 5  # 6 - 1

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data,
        current_data,
        estimands,
        "county",
        fixed_effects=["county_classification", "county_fips"],
        handle_unreporting="drop",
    )

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)

    assert combined_data_handler.data.shape == (133, 33)

    n_expected_columns = combined_data_handler.data.shape[1] + 3  # residual intercept and reporting
    n_expected_columns += 6 + 133 - 2  # subtracting two dropped columns
    assert reporting_data.shape == (133, n_expected_columns)
    assert nonreporting_data.shape == (0, n_expected_columns)

    assert "county_classification_nova" in reporting_data.columns
    assert "county_classification_nova" in nonreporting_data.columns

    assert "county_fips_51790" in reporting_data.columns
    assert "county_fips_51790" in nonreporting_data.columns

    assert "county_classification" in combined_data_handler.fixed_effects
    assert "county_fips" in combined_data_handler.fixed_effects
    assert len(combined_data_handler.expanded_fixed_effects) == 137  # 6 + 133 - 2


def test_generate_fixed_effects_not_all_reporting(va_governor_county_data):
    """
    This tests adding fixed effects when not all units are reporting and therefore
    only a subset of the fixed effect categories are added as columns to the reporting data
    """
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    estimand_baseline = {"turnout": "turnout"}
    n = 10
    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=n)

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_governor_county_data
    )

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data,
        current_data,
        estimands,
        "county",
        fixed_effects=["county_fips"],
        handle_unreporting="drop",
    )

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)

    assert combined_data_handler.data.shape == (133, 33)

    n_expected_columns = combined_data_handler.data.shape[1] + 3  # residual intercept and reporting
    n_expected_columns += n - 1  # for dropped
    assert reporting_data.shape == (n, n_expected_columns)
    assert nonreporting_data.shape == (133 - n, n_expected_columns + (133 - n))

    assert "county_fips_51001" not in reporting_data.columns  # dropped fromg get_dummies because first
    assert "county_fips_51001" not in nonreporting_data.columns  # not added manually nor in nonreporting data

    assert "county_fips_51003" in reporting_data.columns  # in here because get_dummies
    assert "county_fips_51003" in nonreporting_data.columns  # in here because manaully added

    assert "county_fips_51790" not in reporting_data.columns  # not in here because not reporting
    assert "county_fips_51790" in nonreporting_data.columns  # in here because get_dummies

    assert "county_fips" in combined_data_handler.fixed_effects
    assert len(combined_data_handler.expanded_fixed_effects) == n - 1


def test_generate_fixed_effects_mixed_reporting(va_governor_precinct_data):
    """
    This tests adding fixed effects when not all units are reporting but units from the fixed
    effects appear in both the reporting and the nonreporting set.
    """
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "precinct"
    estimands = ["turnout"]
    estimand_baseline = {"turnout": "turnout"}
    n = 100
    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_precinct_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=n)

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_governor_precinct_data
    )

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data,
        current_data,
        estimands,
        "county",
        fixed_effects=["county_fips"],
        handle_unreporting="drop",
    )

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)
    assert combined_data_handler.data.shape == (2360, 33)

    n_expected_columns = combined_data_handler.data.shape[1] + 3  # residual intercept and reporting
    n_expected_columns += 7 - 1  # when n = 100 we get to county 51013
    assert reporting_data.shape == (n, n_expected_columns)
    assert nonreporting_data.shape == (2360 - n, n_expected_columns + (133 - 7))

    assert "county_fips_51001" not in reporting_data.columns  # dropped fromg get_dummies because first
    assert "county_fips_51001" not in nonreporting_data.columns  # not added manually nor in nonreporting data

    assert "county_fips_51003" in reporting_data.columns  # in here because get_dummies
    assert "county_fips_51003" in nonreporting_data.columns  # in here because manaully added

    assert "county_fips_51013" in reporting_data.columns  # in here because get_dummies
    assert "county_fips_51013" in nonreporting_data.columns  # in here because get_dummies and drop_first=False

    assert "county_fips_51790" not in reporting_data.columns  # not in here because not reporting
    assert "county_fips_51790" in nonreporting_data.columns  # in here because get_dummies

    assert "county_fips" in combined_data_handler.fixed_effects
    assert len(combined_data_handler.expanded_fixed_effects) == 7 - 1


def test_expanding_fixed_effects_basic():
    df = pd.DataFrame({"c1": ["a", "b", "b", "c"], "c2": ["w", "x", "y", "z"], "c3": [2, 4, 1, 9]})
    expanded = CombinedDataHandler._expand_fixed_effects(df, ["c1"], drop_first=True)
    pd.testing.assert_frame_equal(
        expanded,
        pd.DataFrame(
            {
                "c2": ["w", "x", "y", "z"],
                "c3": [2, 4, 1, 9],
                "c1_b": [0, 1, 1, 0],
                "c1_c": [0, 0, 0, 1],
                "c1": ["a", "b", "b", "c"],
            }
        ),
    )

    df = pd.DataFrame({"c1": ["a", "b", "b", "c"], "c2": ["w", "x", "y", "z"], "c3": [2, 4, 1, 9]})
    expanded = CombinedDataHandler._expand_fixed_effects(df, ["c1"], drop_first=False)
    pd.testing.assert_frame_equal(
        expanded,
        pd.DataFrame(
            {
                "c2": ["w", "x", "y", "z"],
                "c3": [2, 4, 1, 9],
                "c1_a": [1, 0, 0, 0],
                "c1_b": [0, 1, 1, 0],
                "c1_c": [0, 0, 0, 1],
                "c1": ["a", "b", "b", "c"],
            }
        ),
    )

    expanded = CombinedDataHandler._expand_fixed_effects(df, ["c1", "c2"], drop_first=True)
    pd.testing.assert_frame_equal(
        expanded,
        pd.DataFrame(
            {
                "c3": [2, 4, 1, 9],
                "c1_b": [0, 1, 1, 0],
                "c1_c": [0, 0, 0, 1],
                "c2_x": [0, 1, 0, 0],
                "c2_y": [0, 0, 1, 0],
                "c2_z": [0, 0, 0, 1],
                "c1": ["a", "b", "b", "c"],
                "c2": ["w", "x", "y", "z"],
            }
        ),
    )


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
    assert observed_data.intercept.iloc[0] == 1
    assert observed_data.intercept.sum() == 20


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
