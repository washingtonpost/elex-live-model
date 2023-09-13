import pytest

TOL = 1e-3


def test_compute_conf_frac(conformal_election_model):
    with pytest.raises(NotImplementedError):
        conformal_election_model._compute_conf_frac()


def test_get_conformalization_data_unit(conformal_election_model):
    with pytest.raises(NotImplementedError):
        conformal_election_model.get_all_conformalization_data_unit()


def test_get_all_conformalization_data_agg(conformal_election_model):
    with pytest.raises(NotImplementedError):
        conformal_election_model.get_all_conformalization_data_agg()
