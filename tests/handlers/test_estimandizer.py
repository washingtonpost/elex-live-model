import pandas as pd

from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def test_create_estimand_margin(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["margin"]
    estimand_baseline = {"margin": "margin"}
    # Modify data to include estimand columns
    va_governor_county_data["baseline_margin"] = (
        va_governor_county_data["results_dem"] - va_governor_county_data["results_gop"]
    )
    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_governor_county_data
    )
    estimandizer = Estimandizer(preprocessed_data_handler, estimands)
    new_data_handler = estimandizer.generate_estimands()
    expected_result = pd.DataFrame(
        {
            "results_dem": [100, 200, 150],
            "results_gop": [80, 150, 120],
            "total_gen_voters": [200, 350, 270],
            "margin": [20, 50, 30],
        }
    )
    assert "margin" in new_data_handler.data


def test_create_estimand_voter_turnout_rate(va_governor_county_data):
    election_id = "2017-11-07_VA_G"
    office = "G"
    geographic_unit_type = "county"
    estimands = ["voter_turnout_rate"]
    estimand_baseline = {"voter_turnout_rate": "voter_turnout_rate"}
    # Modify data to include estimand columns
    va_governor_county_data["baseline_voter_turnout_rate"] = (
        va_governor_county_data["results_turnout"] / va_governor_county_data["total_gen_voters"]
    )
    preprocessed_data_handler = PreprocessedDataHandler(
        election_id, office, geographic_unit_type, estimands, estimand_baseline, data=va_governor_county_data
    )
    estimandizer = Estimandizer(preprocessed_data_handler, estimands)
    new_data_handler = estimandizer.generate_estimands()
    expected_result = pd.DataFrame(
        {"results_turnout": [150, 200, 180], "total_gen_voters": [300, 400, 360], "voter_turnout_rate": [0.5, 0.5, 0.5]}
    )
    assert "voter_turnout_rate" in new_data_handler.data
