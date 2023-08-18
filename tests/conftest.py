import json
import logging
import os
import sys

import pandas as pd
import pytest

from elexmodel.client import HistoricalModelClient, ModelClient

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
        fileobj = open(filepath, encoding="utf-8")
        if load:
            return json.load(fileobj)
        if pandas:
            return pd.read_csv(
                filepath,
                dtype={"geographic_unit_fips": str, "geographic_unit_type": str, "county_fips": str, "district": str},
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
def va_config(get_fixture):
    path = os.path.join("config", "2017-11-07_VA_G.json")
    return get_fixture(path, load=True, pandas=False)


@pytest.fixture(scope="session")
def tx_primary_governor_config(get_fixture):
    path = os.path.join("config", "2018-03-06_TX_R.json")
    return get_fixture(path, load=True, pandas=False)


@pytest.fixture(scope="session")
def va_governor_precinct_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "G", "data_precinct.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="session")
def va_governor_county_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "G", "data_county.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="session")
def va_assembly_county_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "Y", "data_county-district.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="session")
def va_assembly_precinct_data(get_fixture):
    path = os.path.join("data", "2017-11-07_VA_G", "Y", "data_precinct-district.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="session")
def az_assembly_precinct_data(get_fixture):
    path = os.path.join("data", "2020-08-04_AZ_R", "S", "data_precinct.csv")
    return get_fixture(path, load=False, pandas=True)


@pytest.fixture(scope="session")
def test_path():
    return _TEST_FOLDER
