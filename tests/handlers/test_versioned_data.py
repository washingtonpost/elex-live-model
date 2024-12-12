from elexmodel.handlers.data.VersionedData import VersionedDataHandler


def test_versioned_data_without_errors(versioned_data_no_errors):
    vdh = VersionedDataHandler("2024-11-05_USA_G", "S", "county")
    results = vdh.compute_versioned_margin_estimate(data=versioned_data_no_errors)
    assert len(results) == 100
    assert (results["nearest_observed_vote"] == 99.0).all()
    assert results["est_margin"].max().round(6) == 0.186405
    assert results["est_correction"].min() == -2.7755575615628914e-17
    assert (results["error_type"] == "none").all()


def test_versioned_data_with_non_monotone_ev(versioned_data_non_monotone):
    vdh = VersionedDataHandler("2024-11-05_USA_G", "S", "county")
    results = vdh.compute_versioned_margin_estimate(data=versioned_data_non_monotone)
    assert len(results) == 101
    assert len(results[results["est_margin"].isnull()]) == 101
    assert len(results[results["est_correction"].isnull()]) == 101
    assert (results["error_type"] == "non-monotone percent expected vote").all()


def test_versioned_data_with_batch_margin_error(versioned_data_batch_margin):
    vdh = VersionedDataHandler("2024-11-05_USA_G", "S", "county")
    results = vdh.compute_versioned_margin_estimate(data=versioned_data_batch_margin)
    assert len(results) == 101
    assert len(results[results["est_margin"].isnull()]) == 101
    assert len(results[results["est_correction"].isnull()]) == 101
    assert (results["error_type"] == "batch_margin").all()
