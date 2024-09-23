import pytest

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
    assert "results_weights" in output_df.columns
    assert result_columns == ["results_party_vote_share_dem", "results_turnout"]


def test_add_estimand_results_historical(va_governor_county_data):
    """
    Tests the add_estimand_results() method with historical elections.
    """
    va_data_copy = va_governor_county_data.copy()
    estimands = ["party_vote_share_dem"]

    estimandizer = Estimandizer()
    (output_df, result_columns) = estimandizer.add_estimand_results(va_data_copy, estimands, True)

    assert "results_party_vote_share_dem" in output_df.columns
    assert result_columns == ["results_party_vote_share_dem", "results_turnout"]


def test_add_estimand_baselines_not_historical(va_governor_county_data):
    estimand_baselines = {"turnout": "turnout", "party_vote_share_dem": "party_vote_share_dem"}
    estimandizer = Estimandizer()
    output_df = estimandizer.add_estimand_baselines(va_governor_county_data.copy(), estimand_baselines, False)
    assert "baseline_weights" in output_df.columns
    assert "baseline_party_vote_share_dem" in output_df.columns
    assert "last_election_results_party_vote_share_dem" in output_df.columns


def test_add_estimand_baselines_historical(va_governor_county_data):
    estimand_baselines = {"turnout": "turnout", "party_vote_share_dem": "party_vote_share_dem"}
    estimandizer = Estimandizer()
    output_df = estimandizer.add_estimand_baselines(
        va_governor_county_data.copy(), estimand_baselines, True, include_results_estimand=True
    )
    assert "baseline_party_vote_share_dem" in output_df.columns
    assert "baseline_weights" in output_df.columns
    assert "results_party_vote_share_dem" in output_df.columns
    assert "last_election_results_party_vote_share_dem" not in output_df.columns


def test_add_turnout_factor(va_governor_county_data):
    estimands = ["party_vote_share_dem", "turnout"]
    estimand_baselines = {"turnout": "turnout", "party_vote_share_dem": "party_vote_share_dem"}
    estimandizer = Estimandizer()
    output_df = estimandizer.add_estimand_baselines(
        va_governor_county_data.copy(), estimand_baselines, False, include_results_estimand=False
    )
    output_df, __ = estimandizer.add_estimand_results(output_df, estimands, False)

    # check that nan turns into 0
    output_df.loc[0, "baseline_weights"] = 0.0
    output_df = estimandizer.add_turnout_factor(output_df)

    assert "turnout_factor" in output_df.columns
    assert 0 == pytest.approx(output_df.loc[0, "turnout_factor"])


def test_add_margin_estimand_zero_normalized_margin(va_governor_county_data):
    estimand_baselines = {"margin": None}
    estimandizer = Estimandizer()

    # test that we're handling zeros ok
    test_df = va_governor_county_data.copy()
    test_df.loc[1, "baseline_dem"] = 0
    test_df.loc[1, "baseline_gop"] = 0

    output_df = estimandizer.add_estimand_baselines(test_df, estimand_baselines, False, include_results_estimand=False)

    assert "baseline_normalized_margin" in output_df.columns
    assert test_df.loc[1, "baseline_normalized_margin"] == 0
