import numpy as np
import pandas as pd

from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.Featurizer import Featurizer
from elexmodel.handlers.data.LiveData import MockLiveDataHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def test_centering_features():
    """
    Test whether centering the features works
    """
    features = ["a", "b", "c", "d"]
    featurizer = Featurizer(features, {})

    df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 2, 2, 2], "c": [3, 3, np.nan, 3], "d": [1, 2, 3, 4]})

    df_new = featurizer.prepare_data(df, center_features=True, scale_features=False, add_intercept=False)
    df_expected_result = pd.DataFrame(
        {
            "a": [0.0, 0.0, 0.0, 0.0],
            "b": [0.0, 0.0, 0.0, 0.0],
            "c": [0.0, 0.0, np.nan, 0.0],
            "d": [-1.5, -0.5, 0.5, 1.5],
        }
    )
    pd.testing.assert_frame_equal(df_new, df_expected_result)


def test_adding_intercept():
    """
    Test adding intercept
    """
    features = ["a", "b", "c"]
    featurizer = Featurizer(features, {})

    df = pd.DataFrame({"a": [2, 2, 2, 2], "b": [3, 3, 3, 3], "c": [1, 2, 3, 4]})

    df_new = featurizer.prepare_data(df, center_features=False, scale_features=False, add_intercept=True)

    assert "intercept" in df_new.columns
    assert "intercept" in featurizer.complete_features
    assert "intercept" in featurizer.active_features
    pd.testing.assert_series_equal(df_new.intercept, pd.Series([1, 1, 1, 1], name="intercept"))


def test_scaling_features():
    """
    Test whether scaling features works
    """
    features = ["a", "b", "c", "d"]
    featurizer = Featurizer(features, {})

    # standard deviations here are 0.5, 1, 2 and inf
    df = pd.DataFrame({"a": [1, 1, 1, 2], "b": [1, 1, 1, 3], "c": [1, 1, 1, 5], "d": [1, 1, 1, 1]})

    df_new = featurizer.prepare_data(df, center_features=False, scale_features=True, add_intercept=False)

    df_expected_result = pd.DataFrame(
        {
            "a": [2.0, 2.0, 2.0, 4.0],
            "b": [1.0, 1.0, 1.0, 3.0],
            "c": [0.5, 0.5, 0.5, 2.5],
            "d": [np.inf, np.inf, np.inf, np.inf],
        }
    )
    pd.testing.assert_frame_equal(df_new, df_expected_result)


def test_column_names():
    """
    This function tests to make sure that the featurizer returns the right columns
    """
    features = ["a", "b", "c"]
    fixed_effects = ["fe_a", "fe_b"]
    featurizer = Featurizer(features, fixed_effects)

    split_fitting_heldout = 4
    # fe_a: "c" exists in fitting but not in heldout, "d" exists in heldout but not in fitting
    # fe_b: "x", "7" and "y" exist in fitting but not in heldout, "z", "w" exist in heldout but not in fitting
    df = pd.DataFrame(
        {
            "a": [5, 3, 1, 5, 2, 2, 2, 2],
            "b": [2, 2, 2, 2, 3, 3, 3, 3],
            "c": [3, 3, 3, 3, 1, 2, 3, 4],
            "d": [1, 2, 3, 4, 5, 3, 1, 5],
            "fe_a": ["a", "a", "b", "c", "a", "a", "b", "d"],
            "fe_b": ["1", "x", "7", "y", "1", "z", "z", "w"],
            "reporting": [1, 1, 1, 1, 0, 0, 0, 0],
            "unit_category": ["expected"] * 8,
        }
    )
    df_new = featurizer.prepare_data(df, center_features=False, scale_features=False, add_intercept=True)

    df_fitting = featurizer.filter_to_active_features(df_new[:split_fitting_heldout])
    df_heldout = featurizer.generate_holdout_data(df_new[split_fitting_heldout:])
    assert (df_fitting.columns == df_heldout.columns).all()

    assert "a" in df_fitting.columns
    assert "a" in df_heldout.columns
    assert "a" in featurizer.features
    assert "a" in featurizer.active_features
    assert "a" in featurizer.complete_features

    assert "fe_a" in featurizer.fixed_effect_cols
    assert "fe_a" in featurizer.fixed_effect_params.keys()

    # a is in fitting and in heldout BUT it's the first and therefore dropped to avoid multicolinearity
    assert "fe_a_a" not in featurizer.expanded_fixed_effects
    assert "fe_a_a" not in featurizer.active_fixed_effects
    assert "fe_a_a" not in featurizer.active_features
    assert "fe_a_a" not in featurizer.complete_features
    assert "fe_a_a" not in df_fitting.columns
    assert "fe_a_a" not in df_heldout.columns

    # b is in fitting and in heldout
    assert "fe_a_b" in featurizer.expanded_fixed_effects
    assert "fe_a_b" in featurizer.active_fixed_effects
    assert "fe_a_b" in featurizer.active_features
    assert "fe_a_b" in featurizer.complete_features
    assert "fe_a_b" in df_fitting.columns
    assert "fe_a_b" in df_heldout.columns

    # c is in fitting but not in heldout
    assert "fe_a_c" in featurizer.expanded_fixed_effects
    assert "fe_a_c" in featurizer.active_fixed_effects
    assert "fe_a_c" in featurizer.active_features
    assert "fe_a_c" in featurizer.complete_features
    assert "fe_a_c" in df_fitting.columns
    assert "fe_a_c" in df_heldout.columns  # should still be in heldout since added manually

    # d is not in fitting but in heldout
    assert "fe_a_d" in featurizer.expanded_fixed_effects
    assert "fe_a_d" not in featurizer.active_fixed_effects
    assert "fe_a_d" not in featurizer.active_features
    assert "fe_a_d" in featurizer.complete_features
    assert "fe_a_d" not in df_fitting.columns
    assert "fe_a_d" not in df_heldout.columns


def test_generating_heldout_set():
    """
    This test makes sure the heldout set is as expected
    """
    features = ["a", "b", "c"]
    fixed_effects = ["fe_a", "fe_b"]
    featurizer = Featurizer(features, fixed_effects)

    split_fitting_heldout = 4
    # fe_a: "c" exists in fitting but not in heldout, "d" exists in heldout but not in fitting
    # fe_b: "x", "7" and "y" exist in fitting but not in heldout, "z", "w" exist in heldout but not in fitting
    df = pd.DataFrame(
        {
            "a": [5, 3, 1, 5, 2, 2, 2, 2],
            "b": [2, 2, 2, 2, 3, 3, 3, 3],
            "c": [3, 3, 3, 3, 1, 2, 3, 4],
            "d": [1, 2, 3, 4, 5, 3, 1, 5],
            "fe_a": ["a", "a", "b", "c", "a", "a", "b", "d"],
            "fe_b": ["1", "x", "7", "y", "1", "z", "z", "w"],
            "reporting": [1, 1, 1, 1, 0, 0, 0, 0],
            "unit_category": ["expected"] * 8,
        }
    )

    df_new = featurizer.prepare_data(df, center_features=False, scale_features=False, add_intercept=True)

    df_heldout = featurizer.generate_holdout_data(df_new[split_fitting_heldout:])

    "a" in df_heldout.columns
    "b" in df_heldout.columns
    "c" in df_heldout.columns
    "d" not in df_heldout.columns  # not specified in features

    "fe_a_a" not in df_heldout.columns  # dropped to avoid multicolinearity
    "fe_a_b" in df_heldout.columns
    "fe_a_c" in df_heldout.columns
    "fe_a_d" not in df_heldout.columns  # not an active fixed effect

    assert df_heldout.loc[6, "fe_a_b"] == 1  # since row 6 has an active fixed effect
    assert df_heldout.loc[7, "fe_a_b"] == 1 / 3  # since row 7 has an inactive fixed effect
    assert df_heldout.loc[7, "fe_a_c"] == 1 / 3  # since row 7 has an inactive fixed effect

    "fe_b_1" not in df_heldout.columns  # dropped to avoid multicolinearity
    "fe_b_x" in df_heldout.columns
    "fe_b_z" not in df_heldout.columns  # inactive

    assert df_heldout.loc[6, "fe_a_b"] == 1  # since row 6 has an active fixed effect
    assert df_heldout.loc[7, "fe_a_b"] == 1 / 3  # since row 7 has an inactive fixed effect
    assert df_heldout.loc[7, "fe_a_c"] == 1 / 3  # since row 7 has an inactive fixed effect

    # element 4 has the dropped fixed effect value in fe_b and so should only have an intercept
    assert df_heldout.loc[4, "intercept"] == 1
    assert df_heldout.loc[4, "fe_b_7"] == 0
    assert df_heldout.loc[4, "fe_b_x"] == 0
    assert df_heldout.loc[4, "fe_b_y"] == 0

    # row 5 has an inactive fixed effect
    assert df_heldout.loc[5, "intercept"] == 1
    assert df_heldout.loc[5, "fe_b_7"] == 1 / 4
    assert df_heldout.loc[5, "fe_b_x"] == 1 / 4
    assert df_heldout.loc[5, "fe_b_y"] == 1 / 4


def test_expanding_fixed_effects_basic():
    fixed_effects = {"c1": ["all"]}
    featurizer = Featurizer([], fixed_effects)
    df = pd.DataFrame({"c1": ["a", "b", "b", "c"], "c2": ["w", "x", "y", "z"], "c3": [2, 4, 1, 9]})
    expanded = featurizer._expand_fixed_effects(df)
    pd.testing.assert_frame_equal(
        expanded.sort_index(axis=1),
        pd.DataFrame(
            {
                "c2": ["w", "x", "y", "z"],
                "c3": [2, 4, 1, 9],
                "c1_a": [1, 0, 0, 0],
                "c1_b": [0, 1, 1, 0],
                "c1_c": [0, 0, 0, 1],
                "c1": ["a", "b", "b", "c"],
            }
        ).sort_index(axis=1),
    )

    df = pd.DataFrame({"c1": ["a", "b", "b", "c"], "c2": ["w", "x", "y", "z"], "c3": [2, 4, 1, 9]})
    expanded = featurizer._expand_fixed_effects(df)
    pd.testing.assert_frame_equal(
        expanded.sort_index(axis=1),
        pd.DataFrame(
            {
                "c2": ["w", "x", "y", "z"],
                "c3": [2, 4, 1, 9],
                "c1_a": [1, 0, 0, 0],
                "c1_b": [0, 1, 1, 0],
                "c1_c": [0, 0, 0, 1],
                "c1": ["a", "b", "b", "c"],
            }
        ).sort_index(axis=1),
    )

    fixed_effects = {"c1": ["all"], "c2": ["all"]}
    featurizer = Featurizer([], fixed_effects)
    expanded = featurizer._expand_fixed_effects(df)
    pd.testing.assert_frame_equal(
        expanded.sort_index(axis=1),
        pd.DataFrame(
            {
                "c3": [2, 4, 1, 9],
                "c1_a": [1, 0, 0, 0],
                "c1_b": [0, 1, 1, 0],
                "c1_c": [0, 0, 0, 1],
                "c2_w": [1, 0, 0, 0],
                "c2_x": [0, 1, 0, 0],
                "c2_y": [0, 0, 1, 0],
                "c2_z": [0, 0, 0, 1],
                "c1": ["a", "b", "b", "c"],
                "c2": ["w", "x", "y", "z"],
            }
        ).sort_index(axis=1),
    )


def test_expand_fixed_effects_selective():
    fixed_effects = {"c1": ["a", "b"]}
    featurizer = Featurizer([], fixed_effects)
    df = pd.DataFrame({"c1": ["a", "b", "b", "c"], "c2": ["w", "x", "y", "z"], "c3": [2, 4, 1, 9]})
    expanded = featurizer._expand_fixed_effects(df)
    pd.testing.assert_frame_equal(
        expanded.sort_index(axis=1),
        pd.DataFrame(
            {
                "c2": ["w", "x", "y", "z"],
                "c3": [2, 4, 1, 9],
                "c1_a": [1, 0, 0, 0],
                "c1_b": [0, 1, 1, 0],
                "c1_other": [0, 0, 0, 1],
                "c1": ["a", "b", "b", "c"],
            }
        ).sort_index(axis=1),
    )

    fixed_effects = {"c1": ["a"], "c2": ["w", "x"]}
    featurizer = Featurizer([], fixed_effects)
    expanded = featurizer._expand_fixed_effects(df)
    pd.testing.assert_frame_equal(
        expanded.sort_index(axis=1),
        pd.DataFrame(
            {
                "c1": ["a", "b", "b", "c"],
                "c2": ["w", "x", "y", "z"],
                "c3": [2, 4, 1, 9],
                "c1_a": [1, 0, 0, 0],
                "c1_other": [0, 1, 1, 1],
                "c2_other": [0, 0, 1, 1],
                "c2_w": [1, 0, 0, 0],
                "c2_x": [0, 1, 0, 0],
            }
        ).sort_index(axis=1),
    )

    fixed_effects = {"c1": ["all"], "c2": ["w", "x"]}
    featurizer = Featurizer([], fixed_effects)
    expanded = featurizer._expand_fixed_effects(df)

    pd.testing.assert_frame_equal(
        expanded.sort_index(axis=1),
        pd.DataFrame(
            {
                "c1": ["a", "b", "b", "c"],
                "c2": ["w", "x", "y", "z"],
                "c3": [2, 4, 1, 9],
                "c1_a": [1, 0, 0, 0],
                "c1_b": [0, 1, 1, 0],
                "c1_c": [0, 0, 0, 1],
                "c2_other": [0, 0, 1, 1],
                "c2_w": [1, 0, 0, 0],
                "c2_x": [0, 1, 0, 0],
            }
        ).sort_index(axis=1),
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

    (reporting_data, nonreporting_data, _) = combined_data_handler.get_units(99, 0.5, 1.5, [], [], False, False, 2, [])

    featurizer = Featurizer([], {"county_classification": "all"})

    n_train = reporting_data.shape[0]
    all_units = pd.concat([reporting_data, nonreporting_data], axis=0)

    x_all = featurizer.prepare_data(all_units, center_features=False, scale_features=False, add_intercept=True)

    reporting_data_features = featurizer.filter_to_active_features(x_all[:n_train])
    nonreporting_data_features = featurizer.generate_holdout_data(x_all[n_train:])

    assert combined_data_handler.data.shape == (133, 35)
    n_expected_columns = 6  # (6 - 1) fixed effects + 1 intercept
    assert reporting_data_features.shape == (133, n_expected_columns)
    assert nonreporting_data_features.shape == (0, n_expected_columns)

    assert "county_classification_nova" in reporting_data_features.columns
    assert "county_classification_nova" in nonreporting_data_features.columns

    assert "county_classification" in featurizer.fixed_effect_cols
    assert len(featurizer.expanded_fixed_effects) == 5  # 6 - 1
    assert len(featurizer.active_fixed_effects) == 5

    combined_data_handler = CombinedDataHandler(
        va_governor_county_data,
        current_data,
        estimands,
        "county",
        handle_unreporting="drop",
    )

    featurizer = Featurizer([], {"county_classification": ["all"], "county_fips": ["all"]})

    (reporting_data, nonreporting_data, _) = combined_data_handler.get_units(99, 0.5, 1.5, [], [], False, False, 2, [])

    n_train = reporting_data.shape[0]
    all_units = pd.concat([reporting_data, nonreporting_data], axis=0)

    x_all = featurizer.prepare_data(all_units, center_features=False, scale_features=False, add_intercept=True)

    reporting_data_features = featurizer.filter_to_active_features(x_all[:n_train])
    nonreporting_data_features = featurizer.generate_holdout_data(x_all[n_train:])

    assert combined_data_handler.data.shape == (133, 35)

    n_expected_columns = 138  # (6 - 1) + (133 - 1) fixed effects + 1 intercept
    assert reporting_data_features.shape == (133, n_expected_columns)
    assert nonreporting_data_features.shape == (0, n_expected_columns)

    assert "county_classification_nova" in reporting_data_features.columns
    assert "county_classification_nova" in nonreporting_data_features.columns

    assert "county_fips_51790" in reporting_data_features.columns
    assert "county_fips_51790" in nonreporting_data_features.columns

    assert "county_classification" in featurizer.fixed_effect_cols
    assert "county_fips" in featurizer.fixed_effect_cols
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

    (reporting_data, nonreporting_data, _) = combined_data_handler.get_units(99, 0.5, 1.5, [], [], False, False, 2, [])

    featurizer = Featurizer([], {"county_fips": ["all"]})
    n_train = reporting_data.shape[0]
    all_units = pd.concat([reporting_data, nonreporting_data], axis=0)

    x_all = featurizer.prepare_data(all_units, center_features=False, scale_features=False, add_intercept=True)

    reporting_data_features = featurizer.filter_to_active_features(x_all[:n_train])
    nonreporting_data_features = featurizer.generate_holdout_data(x_all[n_train:])

    assert combined_data_handler.data.shape == (133, 35)

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

    assert "county_fips" in featurizer.fixed_effect_cols
    assert len(featurizer.expanded_fixed_effects) == 133 - 1
    assert len(featurizer.active_fixed_effects) == n - 1

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

    (reporting_data, nonreporting_data, _) = combined_data_handler.get_units(99, 0.5, 1.5, [], [], False, False, 2, [])

    featurizer = Featurizer([], ["county_fips"])

    n_train = reporting_data.shape[0]
    n_test = nonreporting_data.shape[0]
    all_units = pd.concat([reporting_data, nonreporting_data], axis=0)

    x_all = featurizer.prepare_data(all_units, center_features=False, scale_features=False, add_intercept=True)

    reporting_data_features = featurizer.filter_to_active_features(x_all[:n_train])
    nonreporting_data_features = featurizer.generate_holdout_data(x_all[n_train:])

    assert combined_data_handler.data.shape == (2360, 35)

    n_expected_columns = 7  # when n = 100 we get to county 51013 (minus dropped fixed effect, plus intercept)
    assert reporting_data_features.shape == (n_train, n_expected_columns)  # use n_train since dropping columns
    assert nonreporting_data_features.shape == (n_test, n_expected_columns)

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

    assert "county_fips" in featurizer.fixed_effect_cols
    assert len(featurizer.expanded_fixed_effects) == 133 - 1


def test_separate_state_model():
    """
    This function tests to make sure that the featurizer returns the right columns
    """
    features = ["a", "b", "c"]
    fixed_effects = ["fe_a", "fe_b"]
    states_for_separate_model = ["CC"]

    featurizer = Featurizer(features, fixed_effects, states_for_separate_model)

    df = pd.DataFrame(
        {
            "postal_code": ["AA", "AA", "BB", "BB", "CC", "CC", "CC", "DD"],
            "a": [5, 3, 1, 5, 2, 2, 2, 2],
            "b": [2, 2, 2, 2, 3, 3, 3, 3],
            "c": [3, 3, 3, 3, 1, 2, 3, 4],
            "d": [1, 2, 3, 4, 5, 3, 1, 5],
            "fe_a": ["a", "a", "b", "c", "a", "a", "b", "d"],
            "fe_b": ["1", "x", "7", "y", "1", "z", "z", "w"],
            "reporting": [1, 1, 1, 1, 1, 0, 0, 0],
            "unit_category": ["expected"] * 8,
        }
    )

    df_new = featurizer.prepare_data(df, center_features=False, scale_features=False, add_intercept=True)
    assert df_new.loc[df.postal_code != "CC", "intercept"].all() == 1
    assert df_new.loc[df.postal_code == "CC", "intercept"].all() == 0

    assert df_new.loc[df.postal_code != "CC", "a"].all() > 0
    assert df_new.loc[df.postal_code == "CC", "a"].all() == 0
    assert df_new.loc[df.postal_code != "CC", "a_CC"].all() == 0
    assert df_new.loc[df.postal_code == "CC", "a_CC"].all() > 0

    assert df_new.loc[df.postal_code != "CC", "b"].all() > 0
    assert df_new.loc[df.postal_code == "CC", "b"].all() == 0
    assert df_new.loc[df.postal_code != "CC", "b_CC"].all() == 0
    assert df_new.loc[df.postal_code == "CC", "b_CC"].all() > 0

    assert df_new.loc[df.postal_code != "CC", "c"].all() > 0
    assert df_new.loc[df.postal_code == "CC", "c"].all() == 0
    assert df_new.loc[df.postal_code != "CC", "c_CC"].all() == 0
    assert df_new.loc[df.postal_code == "CC", "c_CC"].all() > 0

    # slightly more complicated, with two states
    states_for_separate_model = ["BB", "CC"]
    featurizer = Featurizer(features, fixed_effects, states_for_separate_model)
    df_new = featurizer.prepare_data(df, center_features=False, scale_features=False, add_intercept=True)

    assert df_new.loc[(df.postal_code != "CC") & (df.postal_code != "BB"), "intercept"].all() == 1
    assert df_new.loc[df.postal_code == "CC", "intercept"].all() == 0
    assert df_new.loc[df.postal_code == "BB", "intercept"].all() == 0

    assert df_new.loc[(df.postal_code != "CC") & (df.postal_code != "BB"), "a"].all() > 0
    assert df_new.loc[df.postal_code == "CC", "a"].all() == 0
    assert df_new.loc[df.postal_code == "BB", "a"].all() == 0
    assert df_new.loc[(df.postal_code != "CC") & (df.postal_code != "BB"), "a_CC"].all() == 0
    assert df_new.loc[(df.postal_code != "CC") & (df.postal_code != "BB"), "a_BB"].all() == 0
    assert df_new.loc[df.postal_code == "CC", "a_CC"].all() > 0
    assert df_new.loc[df.postal_code == "BB", "a_BB"].all() > 0
    assert df_new.loc[df.postal_code == "CC", "a_BB"].all() == 0
    assert df_new.loc[df.postal_code == "BB", "a_CC"].all() == 0

    # if postal code is in fixed effect, then don't add indivdual intercepts
    fixed_effects = ["fe_a", "fe_b", "postal_code"]
    featurizer = Featurizer(features, fixed_effects, states_for_separate_model)
    df_new = featurizer.prepare_data(df, center_features=False, scale_features=False, add_intercept=True)

    assert df_new.loc[(df.postal_code != "CC") & (df.postal_code != "BB"), "intercept"].all() == 1
    assert df_new.loc[df.postal_code == "CC", "intercept"].all() == 0
    assert df_new.loc[df.postal_code == "BB", "intercept"].all() == 0
