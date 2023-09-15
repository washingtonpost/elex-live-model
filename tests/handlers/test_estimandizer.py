from elexmodel.handlers.data.Estimandizer import Estimandizer


def test_add_estimand_results_not_historical(va_governor_county_data):
    """
    Tests the add_estimand_results() method.
    """

    va_data_copy = va_governor_county_data.copy()
    estimands = ["party_vote_share_dem"]

    estimandizer = Estimandizer()
    (output_df, result_columns) = estimandizer.add_estimand_results(va_data_copy, estimands, False)

    assert "results_party_vote_share_dem" in output_df.columns
    assert result_columns == ["results_party_vote_share_dem"]


def test_add_estimand_results_historical(va_governor_county_data):
    """
    Tests the add_estimand_results() method with historical elections.
    """
    va_data_copy = va_governor_county_data.copy()
    estimands = ["party_vote_share_dem"]

    estimandizer = Estimandizer()
    (output_df, result_columns) = estimandizer.add_estimand_results(va_data_copy, estimands, True)

    assert "results_party_vote_share_dem" in output_df.columns
    assert result_columns == ["results_party_vote_share_dem"]


def test_add_estimand_baselines_not_historical(va_governor_county_data):
    estimand_baselines = {"turnout": "turnout", "party_vote_share_dem": "party_vote_share_dem"}
    estimandizer = Estimandizer()
    output_df = estimandizer.add_estimand_baselines(va_governor_county_data.copy(), estimand_baselines, False)
    assert "baseline_party_vote_share_dem" in output_df.columns
    assert "last_election_results_party_vote_share_dem" in output_df.columns


def test_add_estimand_baselines_historical(va_governor_county_data):
    estimand_baselines = {"turnout": "turnout", "party_vote_share_dem": "party_vote_share_dem"}
    estimandizer = Estimandizer()
    output_df = estimandizer.add_estimand_baselines(
        va_governor_county_data.copy(), estimand_baselines, True, include_results_estimand=True
    )
    assert "baseline_party_vote_share_dem" in output_df.columns
    assert "results_party_vote_share_dem" in output_df.columns
    assert "last_election_results_party_vote_share_dem" not in output_df.columns
