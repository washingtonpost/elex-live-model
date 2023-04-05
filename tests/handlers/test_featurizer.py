import numpy as np
import pandas as pd

from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.Featurizer import Featurizer
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def compute_testing_mean_for_centering():
    """ "
    Test whether computing the column mean for centering works.
    """
    features = ["a", "b", "c"]
    featurizer = Featurizer(features, [])

    # test with one dataframe
    df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 2, 2], "c": [3, 3, 3, 3], "d": [1, 2, 3, 4]})

    featurizer.compute_means_for_centering(df)

    assert featurizer.column_means.equals(pd.Series({"a": 1, "b": 2, "c": 3}, index=["a", "b", "c"], dtype=np.float64))

    # test with two dataframes
    df2 = pd.DataFrame({"a": [2, 2, 2, 2], "b": [3, 3, 3, 3], "c": [4, 4, 4, 4], "d": [1, 2, 3, 4]})

    featurizer.compute_means_for_centering(df, df2)
    assert featurizer.column_means.equals(
        pd.Series({"a": 1.5, "b": 2.5, "c": 3.5}, index=["a", "b", "c"], dtype=np.float64)
    )


def test_centering_features():
    """
    Test whether centering the features works
    """
    features = ["a", "b"]
    featurizer = Featurizer(features, [])

    # test with one dataframe
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 9]})

    featurizer.compute_means_for_centering(df)
    featurizer._center_features(df)
    assert df.equals(pd.DataFrame({"a": [-1.0, 0.0, 1.0], "b": [-3.0, -1.0, 4.0]}))


def test_adding_intercept():
    features = ["a", "b", "c"]
    featurizer = Featurizer(features, [])

    # test with one dataframe
    df = pd.DataFrame({"a": [2, 2, 2, 2], "b": [3, 3, 3, 3], "c": [1, 2, 3, 4]})

    featurizer._add_intercept(df)

    assert "intercept" in df.columns
    assert df.intercept.equals(pd.Series([1, 1, 1, 1]))


def test_expanding_fixed_effects_basic():
    fixed_effects = ["c1"]
    featurizer = Featurizer([], fixed_effects)
    df = pd.DataFrame({"c1": ["a", "b", "b", "c"], "c2": ["w", "x", "y", "z"], "c3": [2, 4, 1, 9]})
    expanded = featurizer._expand_fixed_effects(df, drop_first=True)
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
    expanded = featurizer._expand_fixed_effects(df, drop_first=False)
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

    fixed_effects = ["c1", "c2"]
    featurizer = Featurizer([], fixed_effects)
    expanded = featurizer._expand_fixed_effects(df, drop_first=True)
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
        handle_unreporting="drop",
    )

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)

    featurizer = Featurizer([], ["county_classification"])
    featurizer.compute_means_for_centering(reporting_data, nonreporting_data)

    reporting_data_features = featurizer.featurize_fitting_data(reporting_data)
    nonreporting_data_features = featurizer.featurize_heldout_data(nonreporting_data)

    assert combined_data_handler.data.shape == (133, 33)

    n_expected_columns = 6  # (6 - 1) fixed effects + 1 intercept
    assert reporting_data_features.shape == (133, n_expected_columns)
    assert nonreporting_data_features.shape == (0, n_expected_columns)

    assert "county_classification_nova" in reporting_data_features.columns
    assert "county_classification_nova" in nonreporting_data_features.columns

    assert "county_classification" in featurizer.fixed_effects
    assert len(featurizer.expanded_fixed_effects) == 5  # 6 - 1

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data,
        current_data,
        estimands,
        "county",
        handle_unreporting="drop",
    )

    featurizer = Featurizer([], ["county_classification", "county_fips"])
    featurizer.compute_means_for_centering(reporting_data, nonreporting_data)

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)

    reporting_data_features = featurizer.featurize_fitting_data(reporting_data)
    nonreporting_data_features = featurizer.featurize_heldout_data(nonreporting_data)

    assert combined_data_handler.data.shape == (133, 33)

    n_expected_columns = 138  # (6 - 1) + (133 - 1) fixed effects + 1 intercept
    assert reporting_data_features.shape == (133, n_expected_columns)
    assert nonreporting_data_features.shape == (0, n_expected_columns)

    assert "county_classification_nova" in reporting_data_features.columns
    assert "county_classification_nova" in nonreporting_data_features.columns

    assert "county_fips_51790" in reporting_data_features.columns
    assert "county_fips_51790" in nonreporting_data_features.columns

    assert "county_classification" in featurizer.fixed_effects
    assert "county_fips" in featurizer.fixed_effects
    assert len(featurizer.expanded_fixed_effects) == 137  # 6 + 133 - 2


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
        handle_unreporting="drop",
    )

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)

    featurizer = Featurizer([], ["county_fips"])
    featurizer.compute_means_for_centering(reporting_data, nonreporting_data)

    reporting_data_features = featurizer.featurize_fitting_data(reporting_data)
    nonreporting_data_features = featurizer.featurize_heldout_data(nonreporting_data)

    assert combined_data_handler.data.shape == (133, 33)

    n_expected_columns = (n - 1) + 1  # minus 1 for dropped fixed effect, plus 1 for intercept
    assert reporting_data_features.shape == (n, n_expected_columns)
    assert nonreporting_data_features.shape == (133 - n, n_expected_columns)

    assert "county_fips_51001" not in reporting_data_features.columns  # dropped from get_dummies because first
    assert "county_fips_51001" not in nonreporting_data_features.columns  # therefore not added manually

    assert "county_fips_51003" in reporting_data_features.columns  # in here because get_dummies
    assert (
        "county_fips_51003" in nonreporting_data_features.columns
    )  # in here because manaully added (since in reporting_units)

    assert "county_fips_51790" not in reporting_data_features.columns  # not in here because not reporting
    assert (
        "county_fips_51790" not in nonreporting_data_features.columns
    )  # not in here because not in featurizer.complete_features

    assert "county_fips" in featurizer.fixed_effects
    assert len(featurizer.expanded_fixed_effects) == n - 1

    assert not reporting_data_features["county_fips_51009"].isnull().any()


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
        handle_unreporting="drop",
    )

    reporting_data = combined_data_handler.get_reporting_units(99)
    nonreporting_data = combined_data_handler.get_nonreporting_units(99)

    featurizer = Featurizer([], ["county_fips"])
    featurizer.compute_means_for_centering(reporting_data, nonreporting_data)

    reporting_data_features = featurizer.featurize_fitting_data(reporting_data)
    nonreporting_data_features = featurizer.featurize_heldout_data(nonreporting_data)

    assert combined_data_handler.data.shape == (2360, 33)

    n_expected_columns = 7  # when n = 100 we get to county 51013 (minus dropped fixed effect, plus intercept)
    assert reporting_data_features.shape == (n, n_expected_columns)
    assert nonreporting_data_features.shape == (2360 - n, n_expected_columns)

    assert "county_fips_51001" not in reporting_data_features.columns  # dropped from get_dummies because first
    assert "county_fips_51001" not in nonreporting_data_features.columns  # therefore not added manually

    assert "county_fips_51003" in reporting_data_features.columns  # in here because get_dummies
    assert (
        "county_fips_51003" in nonreporting_data_features.columns
    )  # in here because manaully added (because in reporting_data)

    assert "county_fips_51013" in reporting_data_features.columns  # in here because get_dummies
    assert (
        "county_fips_51013" in nonreporting_data_features.columns
    )  # in here because get_dummies (and in featurizer.complete_features)

    assert "county_fips_51790" not in reporting_data_features.columns  # not in here because not reporting
    assert (
        "county_fips_51790" not in nonreporting_data_features.columns
    )  # not in here because not in featurizer.complete_features

    assert "county_fips" in featurizer.fixed_effects
    assert len(featurizer.expanded_fixed_effects) == 7 - 1
