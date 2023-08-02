from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def test_create_estimand_margin_preprocessed(va_governor_county_data):
    """
    Tests margin estimand generation (for preprocessed data only)
    """
    va_data_copy = va_governor_county_data
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = []
    estimand_baseline = {}

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_data_copy
    )

    new_estimands = ["margin"]

    estimandizer = Estimandizer(preprocessed_data_handler, new_estimands)
    new_data_handler = estimandizer.generate_estimands()

    assert "margin" in new_data_handler.data


def test_create_estimand_voter_turnout_rate(va_governor_county_data):
    """
    Tests voter turnout rate estimand generation on preprocessed data of the VA general
    """
    va_data_copy = va_governor_county_data
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = []
    estimand_baseline = {}

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_data_copy
    )

    new_estimands = ["voter_turnout_rate"]

    estimandizer = Estimandizer(preprocessed_data_handler, new_estimands)
    new_data_handler = estimandizer.generate_estimands()

    assert "voter_turnout_rate" in new_data_handler.data


def test_create_estimand_age_combined(va_governor_county_data):
    """
    Tests age bracket estimand generation on a combined data handler
    """
    va_data_copy = va_governor_county_data
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = []
    estimand_baseline = {}

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_data_copy
    )

    live_data_handler = MockLiveDataHandler(
        election_id, office, geographic_unit_type, estimands, data=va_governor_county_data
    )

    current_data = live_data_handler.get_n_fully_reported(n=va_governor_county_data.shape[0])

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data,
        current_data,
        estimands,
        "county",
        handle_unreporting="drop",
    )

    new_estimands = ["age_groups"]

    estimandizer = Estimandizer(combined_data_handler, new_estimands)
    new_data_handler = estimandizer.generate_estimands()

    assert "age_group_30_45" in new_data_handler.data


def test_candidate(tx_primary_governor_config):
    """
    Tests `{candidate_last_name}_{polID}` estimand generation on a preprocessed data handler for tx primaries

    {'2018-03-06_TX_R': [{
        'office': 'G',
        'states': ['TX'],
        'geographic_unit_types': ['county'],
        'baseline_results_year': 2014,
        'historical_election': [],
        'features': ['age_le_30', 'age_geq_30_le_45', 'age_geq_35_le_65', 'age_geq_65', 'ethnicity_east_and_south_asian', 'ethnicity_hispanic_and_portuguese', 'ethnicity_european', 'ethnicity_likely_african_american', 'ethnicity_other', 'ethnicity_unknown', 'median_household_income', 'percent_bachelor_or_higher'],
        'aggregates': ['postal_code', 'county_classification'], 'fixed_effect': [],
        'baseline_pointer': {'abbott_41404': 'abbott_41404', 'krueger_66077': 'abbott_41404', 'kilgore_57793': 'abbott_41404',
        'turnout': 'turnout'}}]}

    This function adds the combined values for each candidate (ex: all abbott_41404) to the main list under '2018-03-06_TX_R'
    """
    tx_data_copy = tx_primary_governor_config
    election_id = "2018-03-06_TX_R"
    office = "G"
    geographic_unit_type = "county"
    estimands = []
    estimand_baseline = {}

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=tx_data_copy
    )

    new_estimands = ["candidate"]

    estimandizer = Estimandizer(preprocessed_data_handler, new_estimands)
    new_data_handler = estimandizer.generate_estimands()

    assert "abbott_41404" in new_data_handler.data[new_data_handler.election_id][0]
