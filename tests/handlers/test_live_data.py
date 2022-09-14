from elexmodel.handlers.data.LiveData import MockLiveDataHandler


def test_load(va_governor_county_data):
    data_handler = MockLiveDataHandler("2017-11-07_VA_G", "G", "county", ["turnout"], data=va_governor_county_data)
    data = data_handler.data

    assert data.shape == (133, 3)
    assert data.size == 399
    assert len(data.columns) == 3


def get_percent_fully_reported(va_governor_county_data):
    election = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    data_handler = MockLiveDataHandler(
        election, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    n = data_handler._frac_to_n(90, _round="up")
    assert n == 67

    n = data_handler._frac_to_n(90, _round="down")
    assert n == 66

    n = data_handler._frac_to_n(0, _round="down")
    assert n == 0

    n = data_handler._frac_to_n(100, _round="down")
    assert n == 133


def test_get_n_fully_reported(va_governor_county_data):
    election = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    data_handler = MockLiveDataHandler(
        election, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    n = 75
    current_reporting_data = data_handler.get_n_fully_reported(n)

    nrow_minus_n = data_handler.data.shape[0] - n
    assert current_reporting_data[current_reporting_data.percent_expected_vote == 100].shape[0] == n
    assert current_reporting_data[current_reporting_data.percent_expected_vote == 0].shape[0] == nrow_minus_n


def test_reporting_unexpected(va_governor_county_data):
    election = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]
    unexpected_units = 5

    data_handler = MockLiveDataHandler(
        election,
        office_id,
        geographic_unit_type,
        estimands,
        data=va_governor_county_data,
        unexpected_units=unexpected_units,
    )

    n = 75
    current_reporting_data = data_handler.get_n_fully_reported(n)

    nrow_minus_n = data_handler.data.shape[0] - n
    assert current_reporting_data[current_reporting_data.percent_expected_vote == 100].shape[0] == n
    # need to also include + unexpected_units here because they get taken out of the reporting total
    assert (
        current_reporting_data[current_reporting_data.percent_expected_vote == 0].shape[0]
        == nrow_minus_n + unexpected_units
    )
    # with county data, all fips should be the same length except for our unexpected0
    current_reporting_data["fips_length"] = current_reporting_data.geographic_unit_fips.map(len)
    assert current_reporting_data[current_reporting_data.fips_length > 5].shape[0] == unexpected_units


def test_sample_enforce(va_governor_county_data):
    election = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    data_handler = MockLiveDataHandler(
        election, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    data_handler.shuffle(seed=5, enforce=["51017", "51009"])
    assert data_handler.data.loc[0, "geographic_unit_fips"] == "51009"
    assert data_handler.data.loc[1, "geographic_unit_fips"] == "51017"


def test_sample_overweight(va_governor_county_data):
    election = "2017-11-07_VA_G"
    office_id = "G"
    geographic_unit_type = "county"
    estimands = ["turnout"]

    data_handler = MockLiveDataHandler(
        election, office_id, geographic_unit_type, estimands, data=va_governor_county_data
    )

    upweight = {
        "county_classification": {
            "central": 100,
            "hampton_roads": 1,
            "nova": 1,
            "shenandoah_valley": 1,
            "southside": 1,
            "southwest": 1,
        }
    }

    data_handler.shuffle(seed=7, upweight=upweight)
    assert (
        va_governor_county_data[
            va_governor_county_data.geographic_unit_fips == data_handler.data.iloc[0].geographic_unit_fips
        ].county_classification.iloc[0]
        == "central"
    )
    assert (
        va_governor_county_data[
            va_governor_county_data.geographic_unit_fips == data_handler.data.iloc[1].geographic_unit_fips
        ].county_classification.iloc[0]
        == "central"
    )
    assert (
        va_governor_county_data[
            va_governor_county_data.geographic_unit_fips == data_handler.data.iloc[2].geographic_unit_fips
        ].county_classification.iloc[0]
        == "central"
    )
    assert (
        va_governor_county_data[
            va_governor_county_data.geographic_unit_fips == data_handler.data.iloc[3].geographic_unit_fips
        ].county_classification.iloc[0]
        == "central"
    )
