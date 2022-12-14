from elexmodel.models import GaussianElectionModel


def test_instantiation():
    model_settings = {}
    model = GaussianElectionModel.GaussianElectionModel(model_settings=model_settings)

    assert model.beta == 1

    model_settings = {"beta": 1}
    model = GaussianElectionModel.GaussianElectionModel(model_settings=model_settings)

    assert model.beta == 1

    model_settings = {"beta": 3}
    model = GaussianElectionModel.GaussianElectionModel(model_settings=model_settings)

    assert model.beta == 3


def test_compute_conf_frac():
    model = GaussianElectionModel.GaussianElectionModel()
    conf_frac = model._compute_conf_frac()

    assert conf_frac == 0.7


def test_get_minimum_reporting_units():
    model = GaussianElectionModel.GaussianElectionModel()
    n_min = model.get_minimum_reporting_units(0.7)

    assert n_min == 7

    n_min = model.get_minimum_reporting_units(0.9)
    assert n_min == 7
