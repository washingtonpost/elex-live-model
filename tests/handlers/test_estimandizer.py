from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def test_share_preprocessed(va_governor_county_data):
    """
    Tests margin estimand generation (for preprocessed data only)

    Structure of a "G" election:
    (['postal_code', 'state_fips', 'county_fips', 'geographic_unit_name',
       'geographic_unit_fips', 'geographic_unit_type', 'county_classification',
       'results_turnout', 'results_dem', 'results_gop', 'baseline_turnout',
       'baseline_dem', 'baseline_gop', 'age_le_30', 'age_geq_30_le_45',
       'age_geq_45_le_65', 'age_geq_65', 'ethnicity_east_and_south_asian',
       'ethnicity_european', 'ethnicity_hispanic_and_portuguese',
       'ethnicity_likely_african_american', 'ethnicity_other',
       'ethnicity_unknown', 'gender_f', 'gender_m', 'gender_unknown',
       'median_household_income', 'percent_bachelor_or_higher',
       'total_age_voters', 'total_eth_voters', 'total_gen_voters'],
      dtype='object')
    """
    va_data_copy = va_governor_county_data.copy()
    election_id = "2017-11-07_VA_G"
    office = "G"
    election_type = election_id[-1]
    geographic_unit_type = "county"
    estimands = []
    estimand_baseline = {}

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_data_copy
    )
    estimand_fns = {
        "party_vote_share": None,
    }

    estimandizer = Estimandizer(preprocessed_data_handler, election_type, estimand_fns)
    new_data_handler = estimandizer.generate_estimands()

    assert "party_vote_share_dem" in new_data_handler.data.columns


def test_share_combined(va_governor_county_data):
    """
    Tests age bracket estimand generation on a combined data handler
    """
    va_data_copy = va_governor_county_data.copy()
    election_id = "2017-11-07_VA_G"
    office = "G"
    election_type = election_id[-1]
    geographic_unit_type = "county"
    estimands = ["dem"]
    estimand_baseline = {}

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_data_copy
    )

    live_data_handler = MockLiveDataHandler(election_id, office, geographic_unit_type, estimands, data=va_data_copy)

    current_data = live_data_handler.get_n_fully_reported(n=va_data_copy.shape[0])

    combined_data_handler = CombinedDataHandler(
        preprocessed_data_handler.data,
        current_data,
        estimands,
        "county",
        handle_unreporting="drop",
    )

    estimand_fns = {
        "party_vote_share": None,
    }

    estimandizer = Estimandizer(combined_data_handler, election_type, estimand_fns)
    new_data_handler = estimandizer.generate_estimands()

    assert "party_vote_share_dem" in new_data_handler.data.columns


def test_candidate(az_assembly_precinct_data):
    """
    Tests `{candidate_last_name}_{polID}` estimand generation on a preprocessed data handler for tx primaries
    """
    az_data_copy = az_assembly_precinct_data.copy()
    election_id = "2020-08-04_AZ_R"
    office = "S"
    election_type = election_id[-1]
    geographic_unit_type = "precinct"
    estimands = []
    estimand_baseline = {}

    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=az_data_copy
    )

    estimand_fns = {
        "candidate": None,
    }

    estimandizer = Estimandizer(preprocessed_data_handler, election_type, estimand_fns)
    new_data_handler = estimandizer.generate_estimands()

    assert "mccarthy_68879" in new_data_handler.data.columns
