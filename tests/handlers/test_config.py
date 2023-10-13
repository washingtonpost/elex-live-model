import os

from elexmodel.handlers.config import ConfigHandler


def test_init(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    assert election_id in config_handler.config.keys()


def test_get_office_subconfig(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    office_subconfig = config_handler._get_office_subconfig(office)
    assert len(office_subconfig) > 0


def test_get_offices(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    offices = config_handler.get_offices()
    assert ["Y", "G"] == offices


def test_get_baseline_pointer_general(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    baseline_pointer = config_handler.get_baseline_pointer(office)
    expected = {"dem": "dem", "gop": "gop", "turnout": "turnout"}
    assert expected == baseline_pointer


def test_get_baseline_pointer_primary(tx_primary_governor_config):
    election_id = "2018-03-06_TX_R"
    config_handler = ConfigHandler(election_id, config=tx_primary_governor_config)

    office = "G"
    baseline_pointer = config_handler.get_baseline_pointer(office)
    assert {
        "abbott_41404": "abbott_41404",
        "krueger_66077": "abbott_41404",
        "kilgore_57793": "abbott_41404",
        "turnout": "turnout",
    } == baseline_pointer


def test_get_estimand_baselines_general(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    estimands = ["turnout", "dem"]
    estimand_baselines = config_handler.get_estimand_baselines(office, estimands)
    assert estimand_baselines == {"turnout": "turnout", "dem": "dem"}


def test_get_estimand_baselines_primary(tx_primary_governor_config):
    election_id = "2018-03-06_TX_R"
    config_handler = ConfigHandler(election_id, config=tx_primary_governor_config)

    office = "G"
    estimands = ["abbott_41404", "turnout"]
    estimand_baselines = config_handler.get_estimand_baselines(office, estimands)
    assert estimand_baselines == {"abbott_41404": "abbott_41404", "turnout": "turnout"}


def test_get_estimands_general(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    estimands = config_handler.get_estimands(office)
    expected = ["dem", "gop", "turnout", "margin"]
    assert expected == estimands


def test_get_estimands_primary(tx_primary_governor_config):
    election_id = "2018-03-06_TX_R"
    config_handler = ConfigHandler(election_id, config=tx_primary_governor_config)

    office = "G"
    estimands = config_handler.get_estimands(office)
    assert ["abbott_41404", "krueger_66077", "kilgore_57793", "turnout"] == estimands


def test_get_states(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    states = config_handler.get_states(office)
    assert ["VA"] == states


def test_get_geographic_unit_types(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    states = config_handler.get_geographic_unit_types(office)

    assert ["precinct", "county"] == states

    office = "Y"
    states = config_handler.get_geographic_unit_types(office)
    assert ["precinct-district", "county-district"] == states


def test_get_features(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    features = config_handler.get_features(office)

    assert len(features) == 14
    assert features[0] == "age_le_30"
    assert features[-2] == "percent_bachelor_or_higher"
    assert features[-1] == "baseline_normalized_margin"


def test_get_aggregates(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    aggregates = config_handler.get_aggregates(office)

    assert ["postal_code", "county_classification", "county_fips", "unit"] == aggregates

    office = "Y"
    aggregates = config_handler.get_aggregates(office)
    assert ["postal_code", "county_classification", "county_fips", "district", "unit"] == aggregates


def test_get_fixed_effects(va_config):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)

    office = "G"
    fixed_effects = config_handler.get_fixed_effects(office)
    assert ["postal_code", "county_fips", "county_classification"] == fixed_effects

    office = "Y"
    fixed_effects = config_handler.get_fixed_effects(office)
    assert ["postal_code", "county_fips", "county_classification", "district"] == fixed_effects


def test_save(va_config, test_path):
    election_id = "2017-11-07_VA_G"
    config_handler = ConfigHandler(election_id, config=va_config)
    local_file_path = f"{test_path}/test_dir/config.json"
    if os.path.exists(local_file_path):
        os.remove(local_file_path)
    config_handler.local_file_path = local_file_path
    config_handler.save()

    assert os.path.exists(local_file_path)
    os.remove(local_file_path)
    os.rmdir(f"{test_path}/test_dir")
