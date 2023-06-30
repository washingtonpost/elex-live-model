import numpy as np
import pandas as pd
import pytest

from elexmodel.distributions.GaussianModel import GaussianModel
from elexmodel.utils import math_utils

TOL = 1e-2
RELAX_TOL = 5e-2


def test_empty_gaussian_model_simple():
    """
    This is a basic test for whether we can return an empty Gaussian model when the
    conformalization set is of size zero.
    """
    gaussian_model = GaussianModel(model_settings={})

    df1 = pd.DataFrame({"c1": [], "c2": []})

    df2 = gaussian_model._empty_gaussian_model(df1, ["c1"])

    assert df2.empty
    assert df2.columns.tolist() == [
        "c1",
        "mu_lower_bound",
        "mu_upper_bound",
        "sigma_lower_bound",
        "sigma_upper_bound",
        "var_inflate",
    ]


def test_empty_gaussian_model(va_governor_precinct_data):
    """
    This is a more complex test for whether we can return an empty Gaussian model when the
    conformalization set is of size zero
    """
    gaussian_model = GaussianModel(model_settings={})

    df1 = va_governor_precinct_data.drop(va_governor_precinct_data.index)

    df2 = gaussian_model._empty_gaussian_model(df1, ["postal_code", "county_fips"])

    assert df2.empty
    assert df2.columns.tolist() == [
        "postal_code",
        "county_fips",
        "mu_lower_bound",
        "mu_upper_bound",
        "sigma_lower_bound",
        "sigma_upper_bound",
        "var_inflate",
    ]


def test_get_n_units_per_group_simple():
    """
    This is a basic test for whether we can return the correct number of elements per group.
    We should have a row for unique elements in df1 and df2, but only count the elements in df1 (conformalization set)
    """
    gaussian_model = GaussianModel(model_settings={})

    df1 = pd.DataFrame({"c1": ["a", "b", "b", "c"]})

    df2 = pd.DataFrame({"c1": ["a", "b", "d", "d"]})

    # if there is no group, then all elements should be in the same group.
    units_per_group = gaussian_model._get_n_units_per_group(df1, None, [])

    assert len(units_per_group) == 1
    assert units_per_group["n"] == 4  # four elements in df1

    # we now test this per group
    units_per_group = gaussian_model._get_n_units_per_group(df1, df2, ["c1"])

    assert units_per_group.iloc[0]["n"] == 1.0
    assert units_per_group.iloc[1]["n"] == 2.0
    assert units_per_group.iloc[2]["n"] == 0.0  # d is third since merginging df2 onto df1 and 0.0 because not in df1
    assert units_per_group.iloc[3]["n"] == 1.0


def test_get_n_units_per_group(va_governor_precinct_data):
    """
    A more complex test for whether we can return the correct number of elements per group
    """
    gaussian_model = GaussianModel(model_settings={})

    df1 = va_governor_precinct_data[:1000]
    df2 = va_governor_precinct_data[1000:]

    # all in the same group
    units_per_group = gaussian_model._get_n_units_per_group(df1, df2, [])

    assert len(units_per_group) == 1
    assert units_per_group["n"] == 1000  # there are 1000 elements in df1

    # group by postal code, still all in the same group
    units_per_group = gaussian_model._get_n_units_per_group(df1, df2, ["postal_code"])

    assert units_per_group.shape[0] == 1
    assert units_per_group.iloc[0]["n"] == 1000

    # group by fips
    units_per_group = gaussian_model._get_n_units_per_group(df1, df2, ["county_fips"])

    # there are as many elements as there are counties
    assert units_per_group.shape[0] == va_governor_precinct_data.county_fips.unique().shape[0]
    assert (
        units_per_group[units_per_group.county_fips == "51095"]["n"].iloc[0] == 2.0
    )  # 2 precincts from 51095 since that had index 1001
    assert units_per_group[units_per_group.county_fips == "51001"]["n"].iloc[0] == 16.0
    assert units_per_group[units_per_group.county_fips == "51107"]["n"].iloc[0] == 0.0


def test_fit():
    """
    We test the basic model fit function
    """
    random_number_generator = np.random.RandomState(42)

    # generate test data
    mean_lower = 5
    mean_upper = 7
    sd_lower = 2
    sd_upper = 0.5
    n = 100
    lower = random_number_generator.normal(loc=mean_lower, scale=sd_lower, size=n)
    upper = random_number_generator.normal(loc=mean_upper, scale=sd_upper, size=n)
    weights = random_number_generator.randint(low=1, high=100, size=n)
    alpha = 0.9
    estimand = "turnout"
    model_settings = {
        "election_id": "2017-11-07_VA_G",
        "office": "G",
        "geographic_unit_type": "county",
        "save_conformalization": False,
        "beta": 1,
        "winsorize": False,
    }

    gaussian_model = GaussianModel(model_settings)

    df = pd.DataFrame({f"last_election_results_{estimand}": weights, "lower_bounds": lower, "upper_bounds": upper})

    # all in the same group
    g = gaussian_model._fit(df, estimand, [], alpha)

    # assumes that weighted median and standard deviation bootstrap works
    # tests for that in test_utils
    assert math_utils.weighted_median(lower, weights / weights.sum()) == pytest.approx(g.mu_lower_bound[0], TOL)
    assert math_utils.boot_sigma(lower, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound[0], RELAX_TOL
    )
    assert math_utils.weighted_median(upper, weights / weights.sum()) == pytest.approx(g.mu_upper_bound[0], TOL)
    assert math_utils.boot_sigma(upper, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_upper_bound[0], RELAX_TOL
    )

    # generate test data for two different groups
    mean_a = 5
    mean_b = 7
    sd_a = 2
    sd_b = 0.5
    n = 100
    a = random_number_generator.normal(loc=mean_a, scale=sd_a, size=n)
    b = random_number_generator.normal(loc=mean_b, scale=sd_b, size=n)
    weights_a = random_number_generator.randint(low=1, high=100, size=n)
    weights_b = random_number_generator.randint(low=1, high=100, size=n)
    df_a = pd.DataFrame(
        {
            "postal_code": "VA",
            "geographic_unit_fips": 1,
            "lower_bounds": a,
            "upper_bounds": a,
            f"last_election_results_{estimand}": weights_a,
        }
    )
    df_a["group"] = "a"
    df_b = pd.DataFrame(
        {
            "postal_code": "VA",
            "geographic_unit_fips": 2,
            "lower_bounds": b,
            "upper_bounds": b,
            f"last_election_results_{estimand}": weights_b,
        }
    )
    df_b["group"] = "b"
    df = pd.concat([df_a, df_b])

    # fit model to multiple groups separately
    g = gaussian_model._fit(df, estimand, ["group"], alpha)

    assert math_utils.weighted_median(a, weights_a / weights_a.sum()) == pytest.approx(g.mu_lower_bound[0], TOL)
    assert math_utils.boot_sigma(a, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound[0], RELAX_TOL
    )
    assert math_utils.weighted_median(b, weights_b / weights_b.sum()) == pytest.approx(g.mu_lower_bound[1], TOL)
    assert math_utils.boot_sigma(b, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound[1], RELAX_TOL
    )


def test_fit_winsorized():
    """
    We test the basic model fit function
    """
    random_number_generator = np.random.RandomState(42)

    # generate test data
    mean_lower = 5
    mean_upper = 7
    sd_lower = 2
    sd_upper = 0.5
    n = 100
    lower = random_number_generator.normal(loc=mean_lower, scale=sd_lower, size=n)
    upper = random_number_generator.normal(loc=mean_upper, scale=sd_upper, size=n)
    weights = random_number_generator.randint(low=1, high=100, size=n)
    alpha = 0.9
    estimand = "turnout"
    model_settings = {
        "election_id": "2017-11-07_VA_G",
        "office": "G",
        "geographic_unit_type": "county",
        "save_conformalization": False,
        "beta": 1,
        "winsorize": True,
    }

    gaussian_model = GaussianModel(model_settings)

    df = pd.DataFrame({f"last_election_results_{estimand}": weights, "lower_bounds": lower, "upper_bounds": upper})

    # all in the same group
    g = gaussian_model._fit(df, estimand, [], alpha)

    # assumes that weighted median and standard deviation bootstrap works
    # tests for that in test_utils
    assert math_utils.weighted_median(lower, weights / weights.sum()) == pytest.approx(g.mu_lower_bound[0], TOL)
    assert math_utils.boot_sigma(lower, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound[0], RELAX_TOL
    )
    assert math_utils.weighted_median(upper, weights / weights.sum()) == pytest.approx(g.mu_upper_bound[0], TOL)
    assert math_utils.boot_sigma(upper, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_upper_bound[0], RELAX_TOL
    )

    # generate test data for two different groups
    mean_a = 5
    mean_b = 7
    sd_a = 2
    sd_b = 0.5
    n = 100
    a = random_number_generator.normal(loc=mean_a, scale=sd_a, size=n)
    b = random_number_generator.normal(loc=mean_b, scale=sd_b, size=n)
    weights_a = random_number_generator.randint(low=1, high=100, size=n)
    weights_b = random_number_generator.randint(low=1, high=100, size=n)
    df_a = pd.DataFrame(
        {
            "postal_code": "VA",
            "geographic_unit_fips": 1,
            "lower_bounds": a,
            "upper_bounds": a,
            f"last_election_results_{estimand}": weights_a,
        }
    )
    df_a["group"] = "a"
    df_b = pd.DataFrame(
        {
            "postal_code": "VA",
            "geographic_unit_fips": 2,
            "lower_bounds": b,
            "upper_bounds": b,
            f"last_election_results_{estimand}": weights_b,
        }
    )
    df_b["group"] = "b"
    df = pd.concat([df_a, df_b])

    # fit model to multiple groups separately
    g = gaussian_model._fit(df, estimand, ["group"], alpha)

    assert math_utils.weighted_median(a, weights_a / weights_a.sum()) == pytest.approx(g.mu_lower_bound[0], TOL)
    assert math_utils.boot_sigma(a, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound[0], RELAX_TOL
    )
    assert math_utils.weighted_median(b, weights_b / weights_b.sum()) == pytest.approx(g.mu_lower_bound[1], TOL)
    assert math_utils.boot_sigma(b, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound[1], RELAX_TOL
    )


def test_large_and_small_fit():
    """
    Here we test whether we can generate a large and a small model, when one group has too few elements.
    """
    random_number_generator = np.random.RandomState(42)

    estimand = "turnout"
    model_settings = {
        "election_id": "2017-11-07_VA_G",
        "office": "G",
        "geographic_unit_type": "county",
        "save_conformalization": False,
        "beta": 1,
        "winsorize": True,
    }

    gaussian_model = GaussianModel(model_settings)

    # create two very unequal sized groups
    mean_a = 2
    mean_b = 100
    sd_a = 0.5
    sd_b = 5
    n_a = 5  # can't fit a specific model to this group
    n_b = 100
    a = random_number_generator.normal(loc=mean_a, scale=sd_a, size=n_a)
    b = random_number_generator.normal(loc=mean_b, scale=sd_b, size=n_b)
    weights_a = random_number_generator.randint(low=1, high=100, size=n_a)
    weights_b = random_number_generator.randint(low=1, high=100, size=n_b)
    df_a = pd.DataFrame(
        {
            "postal_code": "VA",
            "geographic_unit_fips": 1,
            "lower_bounds": a,
            "upper_bounds": a,
            f"last_election_results_{estimand}": weights_a,
        }
    )
    df_a["group_2"] = "a"
    df_b = pd.DataFrame(
        {
            "postal_code": "VA",
            "geographic_unit_fips": 2,
            "lower_bounds": b,
            "upper_bounds": b,
            f"last_election_results_{estimand}": weights_b,
        }
    )
    df_b["group_2"] = "b"
    df = pd.concat([df_a, df_b])
    df["group_1"] = "general"
    general = df.lower_bounds.values
    general_weights = df[f"last_election_results_{estimand}"].values

    alpha = 0.9

    reporting = pd.DataFrame({"group_1": ["general", "general"], "group_2": ["a", "b"]})
    nonreporting = pd.DataFrame({"group_1": ["general", "general"], "group_2": ["a", "b"]})

    g = gaussian_model.fit(
        df,
        reporting,
        nonreporting,
        estimand,
        aggregate=["group_1", "group_2"],
        alpha=alpha,
        reweight=False,
    )

    assert math_utils.weighted_median(general, general_weights / general_weights.sum()) == pytest.approx(
        g.mu_lower_bound.values[0], TOL
    )
    assert math_utils.boot_sigma(general, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound.values[0], RELAX_TOL
    )
    assert math_utils.weighted_median(b, weights_b / weights_b.sum()) == pytest.approx(g.mu_lower_bound.values[1], TOL)
    assert math_utils.boot_sigma(b, conf=(3 + alpha) / 4, winsorize=model_settings["winsorize"]) == pytest.approx(
        g.sigma_lower_bound.values[1], RELAX_TOL
    )
    assert g.group_1.values[0] == "general"
    assert g.group_1.values[1] == "general"
    assert g.group_2.values[0] is np.nan
    assert g.group_2.values[1] == "b"
