import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

from elexmodel.client import HistoricalModelClient, ModelClient
from elexmodel.models import BaseElectionModel, BootstrapElectionModel, ConformalElectionModel

_TEST_FOLDER = os.path.dirname(__file__)
FIXTURE_DIR = os.path.join(_TEST_FOLDER, "fixtures")


@pytest.fixture(autouse=True, scope="session")
def setup_logging():
    LOG = logging.getLogger("elexmodel")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(handler)


@pytest.fixture(scope="session")
def get_fixture():
    def _get_fixture(filename, load=True, pandas=False):
        filepath = os.path.join(FIXTURE_DIR, filename)
        with open(filepath, encoding="utf-8") as fileobj:
            if load:
                return json.load(fileobj)
            if pandas:
                return pd.read_csv(
                    filepath,
                    dtype={
                        "geographic_unit_fips": str,
                        "geographic_unit_type": str,
                        "county_fips": str,
                        "district": str,
                    },
                )
            return fileobj

    return _get_fixture


@pytest.fixture(scope="session")
def model_client():
    return ModelClient()


@pytest.fixture(scope="session")
def historical_model_client():
    return HistoricalModelClient()


@pytest.fixture(scope="session")
def base_election_model():
    model_settings = {}
    return BaseElectionModel.BaseElectionModel(model_settings)


@pytest.fixture(scope="session")
def conformal_election_model():
    model_settings = {}
    return ConformalElectionModel.ConformalElectionModel(model_settings)


@pytest.fixture(scope="function")
def bootstrap_election_model():
    model_settings = {"features": ["baseline_normalized_margin"]}
    return BootstrapElectionModel.BootstrapElectionModel(model_settings)


@pytest.fixture(scope="session")
def rng():
    seed = 1941
    return np.random.default_rng(seed=seed)


@pytest.fixture(scope="function")
def va_config(get_fixture):
    path = os.path.join("config", "2017-11-07_VA_G.json")
    return get_fixture(path, load=True, pandas=False)


@pytest.fixture(scope="function")
def tx_primary_governor_config(get_fixture):
    path = os.path.join("config", "2018-03-06_TX_R.json")
    return get_fixture(path, load=True, pandas=False)


@pytest.fixture(scope="function")
def va_governor_precinct_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "G", "data_precinct.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="function")
def va_governor_county_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "G", "data_county.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="function")
def va_assembly_county_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "Y", "data_county-district.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="function")
def va_assembly_precinct_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "Y", "data_precinct-district.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="function")
def az_assembly_precinct_data(get_fixture):
    path = os.path.join("data", "2020-08-04_AZ_R", "S", "data_precinct.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="session")
def test_path():
    return _TEST_FOLDER


@pytest.fixture(scope="function")
def versioned_data_no_errors(get_fixture):
    path = os.path.join("data", "2024-11-05_USA_G", "S", "versioned_no_errors.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="function")
def versioned_data_non_monotone(get_fixture):
    path = os.path.join("data", "2024-11-05_USA_G", "S", "versioned_non_monotone.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="function")
def versioned_data_batch_margin(get_fixture):
    path = os.path.join("data", "2024-11-05_USA_G", "S", "versioned_batch_margin.csv")
    return get_fixture(path, load=False, pandas=True)
