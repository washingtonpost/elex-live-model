from elexmodel.handlers.data.VersionedData import VersionedDataHandler


def test_versioned_data_without_errors(versioned_data_no_errors):
    vdh = VersionedDataHandler("2024-11-05_USA_G", "S", "county", data=versioned_data_no_errors)
    assert len(vdh.data) == 100
    assert (vdh.data["nearest_observed_vote"] == 99.0).all()
    assert vdh.data["est_margin"].max().round(6) == 0.186405
    assert vdh.data["est_correction"].min() == -2.7755575615628914e-17
    assert (vdh.data["error_type"] == "none").all()


def test_versioned_data_with_non_monotone_ev(versioned_data_non_monotone):
    vdh = VersionedDataHandler("2024-11-05_USA_G", "S", "county", data=versioned_data_non_monotone)
    assert len(vdh.data) == 101
    assert len(vdh.data[vdh.data["est_margin"].isnull()]) == 101
    assert len(vdh.data[vdh.data["est_correction"].isnull()]) == 101
    assert (vdh.data["error_type"] == "non-monotone percent expected vote").all()


def test_versioned_data_with_batch_margin_error(versioned_data_batch_margin):
    vdh = VersionedDataHandler("2024-11-05_USA_G", "S", "county", data=versioned_data_batch_margin)
    assert len(vdh.data) == 101
    assert len(vdh.data[vdh.data["est_margin"].isnull()]) == 101
    assert len(vdh.data[vdh.data["est_correction"].isnull()]) == 101
    assert (vdh.data["error_type"] == "batch_margin").all()
