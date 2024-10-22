import numpy as np
import pandas as pd

from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.LiveData import MockLiveDataHandler


def test_load(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands=["turnout"], data=va_governor_county_data
    )
    current_data = live_data_handler.data
    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="drop"
    )

    assert combined_data_handler.data.shape == (133, 32)


def test_zero_unreporting_missing_single_estimand_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0  # value with na result has been set to zero
    assert combined_data_handler.data["results_turnout"].iloc[0] != 0  # has not been set to zero
    assert (
        combined_data_handler.data["percent_expected_vote"].iloc[0] == 0
    )  # percent expected vote with na result has been set to zero
    assert combined_data_handler.data.shape == (133, 34)  # didn't drop any
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_missing_multiple_estimands_value(va_governor_county_data):
    """
    Set the value for multiple estimands (dem, turnout) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan
    current_data.loc[0, "results_turnout"] = np.nan

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0
    assert combined_data_handler.data["results_turnout"].iloc[0] == 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 34)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero
    assert combined_data_handler.data["results_turnout"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_missing_percent_expected_vote_value(va_governor_county_data):
    """
    Set the value and percent reporting for one estimand (dem) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "percent_expected_vote"] = np.nan
    current_data.loc[0, "results_dem"] = np.nan

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 34)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_zero_unreporting_random_percent_expected_vote_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "zero"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = np.random.randint(1, 100, current_data.shape[0])
    current_data.loc[0, "results_dem"] = np.nan

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="zero"
    )
    assert combined_data_handler.data["results_dem"].iloc[0] == 0.0  # all values set to 0.0
    assert combined_data_handler.data["percent_expected_vote"].iloc[0] == 0.0
    assert combined_data_handler.data.shape == (133, 34)
    assert combined_data_handler.data["results_dem"].iloc[1] != 0  # didn't accidentally set other to zero


def test_drop_unreporting_missing_single_estimand_value(va_governor_county_data):
    """
    Set the value for one estimand (dem) as na to test unreporting = "drop"
    """
    election_id = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout", "dem"]
    live_data_handler = MockLiveDataHandler(
        election_id, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.data
    current_data["percent_expected_vote"] = 100
    current_data.loc[0, "results_dem"] = np.nan

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data, current_data, estimands, "county", handle_unreporting="drop"
    )
    assert combined_data_handler.data.shape == (132, 34)  # dropped one
    assert combined_data_handler.data["results_dem"].iloc[0] != 0  # didn't accidentally set other to zero


def test_get_reporting_data(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=20)

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout
    va_governor_county_data["last_election_results_turnout"] = va_governor_county_data.baseline_turnout + 1

    # no fixed effects
    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)
    turnout_factor_lower = 0.5
    turnout_factor_upper = 1.5
    margin_change_threshold = 0.4
    (observed_data, _, _) = combined_data_handler.get_units(
        100, turnout_factor_lower, turnout_factor_upper, margin_change_threshold, [], [], []
    )
    assert observed_data.shape[0] == 20
    assert observed_data.reporting.iloc[0] == 1
    assert observed_data.reporting.sum() == 20


def test_get_reporting_data_dropping_with_turnout_factor(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=20)

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout
    va_governor_county_data["last_election_results_turnout"] = va_governor_county_data.baseline_turnout + 1

    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)

    turnout_factor_lower = 0.95
    turnout_factor_upper = 1.2
    reporting_units_above_turnout_factor_threshold = combined_data_handler.data[
        combined_data_handler.data.turnout_factor > turnout_factor_upper
    ].shape[0]
    reporting_units_below_turnout_factor_threshold = combined_data_handler.data[
        (combined_data_handler.data.percent_expected_vote == 100)
        & (combined_data_handler.data.turnout_factor < turnout_factor_lower)
    ].shape[0]

    (observed_data, _, _) = combined_data_handler.get_units(
        100, turnout_factor_lower, turnout_factor_upper, 0.4, [], [], []
    )

    # 20 units should be reporting,
    # but the additional ones are dropped to unexpected because they are above/below threshold
    # and so are subtracted from the reporting ones
    assert observed_data.shape[0] == 20 - (
        reporting_units_above_turnout_factor_threshold + reporting_units_below_turnout_factor_threshold
    )


def test_get_unexpected_units_county_district(va_assembly_county_data):
    election_id = "2017-11-07_VA_G"
    office = "Y"
    geographic_unit_type = "county-district"
    estimands = ["turnout"]
    unexpected_units = 5

    live_data_handler = MockLiveDataHandler(
        election_id,
        office,
        geographic_unit_type,
        estimands,
        data=va_assembly_county_data,
        unexpected_units=unexpected_units,
    )
    current_data = live_data_handler.get_n_fully_reported(n=20)

    va_assembly_county_data["baseline_weights"] = va_assembly_county_data.baseline_turnout
    va_assembly_county_data["last_election_results_turnout"] = va_assembly_county_data.baseline_turnout + 1

    combined_data_handler = CombinedDataHandler(va_assembly_county_data, current_data, estimands, geographic_unit_type)
    turnout_factor_lower = 0  # set to extreme values so as not to add any more "unexpected"
    turnout_factor_upper = 100
    (_, _, unexpected_data) = combined_data_handler.get_units(
        100, turnout_factor_lower, turnout_factor_upper, 0.4, [], [], ["county_fips", "district"]
    )

    assert unexpected_data.shape[0] == unexpected_units
    assert unexpected_data[unexpected_data.county_fips == ""].shape[0] == 0
    assert unexpected_data["county_fips"].map(lambda x: len(x) == 6).all()
    assert unexpected_data[unexpected_data.district == ""].shape[0] == 0
    assert unexpected_data["district"].map(lambda x: len(x) < 6).all()
    # no non-predictive units here
    assert len(unexpected_data[unexpected_data["unit_category"] == "unexpected"]) == unexpected_units


def test_get_unexpected_units_county(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    reporting_unexpected_units = 5

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

    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout
    va_governor_county_data["last_election_results_turnout"] = va_governor_county_data.baseline_turnout + 1

    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)
    turnout_factor_lower = 0.5
    turnout_factor_upper = 1.5
    (_, _, unexpected_data) = combined_data_handler.get_units(
        100, turnout_factor_lower, turnout_factor_upper, 0.4, [], [], ["county_fips"]
    )
    assert unexpected_data.shape[0] == reporting_unexpected_units + 1
    assert unexpected_data[unexpected_data.county_fips == ""].shape[0] == 0
    assert unexpected_data["county_fips"].map(lambda x: len(x) == 6).all()
    # test that nonreporting unexpected unit is captured here
    assert unexpected_data[unexpected_data.percent_expected_vote == 50].shape[0] == 1
    # no non-predictive units here
    assert len(unexpected_data[unexpected_data["unit_category"] == "unexpected"]) == reporting_unexpected_units + 1


def test_zero_baseline_turnout_as_unexpected(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=20)
    # Add an additional row of a nonreporting unexpected unit that we test below
    va_governor_county_data.loc[0, "baseline_turnout"] = 0
    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout
    va_governor_county_data["last_election_results_turnout"] = va_governor_county_data.baseline_turnout + 1
    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)
    turnout_factor_lower = 0.5
    turnout_factor_upper = 1.5

    (reporting_units, nonreporting_units, unexpected_data) = combined_data_handler.get_units(
        100, turnout_factor_lower, turnout_factor_upper, 0.4, [], [], ["county_fips"]
    )

    assert va_governor_county_data.loc[0].geographic_unit_fips in unexpected_data.geographic_unit_fips.tolist()
    assert len(unexpected_data) == 1
    assert len(unexpected_data[unexpected_data["unit_category"] == "non-modeled"]) == 1

    assert len(reporting_units) == 20 - 1
    assert va_governor_county_data.loc[0].geographic_unit_fips not in reporting_units.geographic_unit_fips.tolist()

    assert va_governor_county_data.loc[0].geographic_unit_fips not in nonreporting_units.geographic_unit_fips.tolist()


def test_turnout_factor_as_non_predictive(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=133)
    # Add an additional row of a nonreporting unexpected unit that we test below
    va_governor_county_data.loc[0, "baseline_turnout"] = 0
    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout
    va_governor_county_data["last_election_results_turnout"] = va_governor_county_data.baseline_turnout + 1
    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)
    turnout_factor_lower = 0.95
    turnout_factor_upper = 1.2
    (_, _, unexpected_data) = combined_data_handler.get_units(
        100, turnout_factor_lower, turnout_factor_upper, 0.4, [], [], ["county_fips"]
    )
    over = combined_data_handler.data[combined_data_handler.data.turnout_factor >= turnout_factor_upper].shape[0]
    under = combined_data_handler.data[combined_data_handler.data.turnout_factor < turnout_factor_lower].shape[0]
    assert unexpected_data.shape[0] == over + under
    assert len(unexpected_data[unexpected_data["unit_category"] == "non-modeled"]) == over + under


def test_margin_change_threshold_as_non_predictive(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["margin"]

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=133)
    va_governor_county_data["baseline_weights"] = (
        va_governor_county_data.baseline_dem + va_governor_county_data.baseline_gop
    )
    va_governor_county_data["baseline_margin"] = (
        va_governor_county_data.baseline_dem - va_governor_county_data.baseline_gop
    )
    va_governor_county_data["baseline_normalized_margin"] = (
        va_governor_county_data.baseline_margin / va_governor_county_data.baseline_weights
    )
    va_governor_county_data["last_election_results_margin"] = va_governor_county_data.baseline_margin

    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)

    margin_change_threshold = 0.1
    (reporting_data, _, unexpected_data) = combined_data_handler.get_units(
        100, 0.2, 2.5, margin_change_threshold, [], [], ["county_fips"]
    )
    over = combined_data_handler.data[
        combined_data_handler.data.normalized_margin_change > margin_change_threshold
    ].shape[0]
    assert unexpected_data.shape[0] == over
    assert len(unexpected_data[unexpected_data["unit_category"] == "non-modeled"]) == over
    assert reporting_data.shape[0] == 133 - over


def test_unit_blocklist(va_governor_county_data, rng):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=100)
    # Add an additional row of a nonreporting unexpected unit that we test below
    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout
    va_governor_county_data["last_election_results_turnout"] = va_governor_county_data.baseline_turnout + 1
    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)

    number_to_blocklist = 10
    unit_blocklist = rng.choice(
        va_governor_county_data.geographic_unit_fips, number_to_blocklist, replace=False
    ).tolist()
    (reporting_data, nonreporting_data, unexpected_data) = combined_data_handler.get_units(
        100, 0.5, 1.5, 0.4, unit_blocklist, [], ["county_fips"]
    )
    number_blocklist_reporting = (
        current_data[current_data.percent_expected_vote == 100].geographic_unit_fips.isin(unit_blocklist).sum()
    )
    number_blocklist_nonreporting = (
        current_data[current_data.percent_expected_vote < 100].geographic_unit_fips.isin(unit_blocklist).sum()
    )

    assert unexpected_data.shape[0] == number_to_blocklist
    assert unexpected_data.geographic_unit_fips.isin(unit_blocklist).all()
    assert reporting_data.shape[0] == 100 - number_blocklist_reporting
    assert nonreporting_data.shape[0] == 33 - number_blocklist_nonreporting


def test_postal_code_blocklist(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )
    current_data = live_data_handler.get_n_fully_reported(n=100)
    # Add an additional row of a nonreporting unexpected unit that we test below
    va_governor_county_data["baseline_weights"] = va_governor_county_data.baseline_turnout
    va_governor_county_data["last_election_results_turnout"] = va_governor_county_data.baseline_turnout + 1
    combined_data_handler = CombinedDataHandler(va_governor_county_data, current_data, estimands, geographic_unit_type)

    postal_code_blocklist = ["VA"]
    (reporting_data, nonreporting_data, unexpected_data) = combined_data_handler.get_units(
        100, 0.5, 1.5, 0.4, [], postal_code_blocklist, ["county_fips"]
    )

    assert unexpected_data.shape[0] == 133
    assert reporting_data.shape[0] == 0
    assert nonreporting_data.shape[0] == 0
