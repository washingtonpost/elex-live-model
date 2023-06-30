import math

import numpy as np
import pandas as pd
import pytest

from elexmodel.utils import math_utils


def test_var_inflate():
    x = np.asarray([1, 2, 3])
    y = math_utils.compute_inflate(x)
    assert y == pytest.approx(14 / 36)


def test_weighted_median():
    x = np.array([0, 10, 20, 25, 30, 30, 35, 50])
    w = np.array([0, 1, 0, 1, 2, 2, 2, 1])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 30

    w = np.ones_like(x)
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 27.5

    x = np.asarray([7, 1, 2, 4, 10])
    w = np.asarray([1, 1 / 3, 1 / 3, 1 / 3, 1])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 7

    w = np.ones_like(x)
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 4

    x = np.asarray([7, 1, 2, 4, 10, 15])
    w = np.asarray([1, 1 / 3, 1 / 3, 1 / 3, 1, 1])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 8.5

    x = np.asarray([1, 2, 4, 7, 10, 15])
    w = np.asarray([1 / 3, 1 / 3, 1 / 3, 1, 1, 1])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 8.5

    x = np.asarray([0, 10, 20, 30])
    w = np.asarray([30, 191, 9, 0])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 10

    x = np.asarray([1, 2, 3, 4, 5])
    w = np.asarray([10, 1, 1, 1, 9])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 2.5

    x = np.asarray([30, 40, 50, 60, 35])
    w = np.asarray([1, 3, 5, 4, 2])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 50

    x = np.asarray([2, 0.6, 1.3, 0.3, 0.3, 1.7, 0.7, 1.7, 0.4])
    w = np.asarray([2, 2, 0, 1, 2, 2, 1, 6, 0])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 1.7

    x = np.asarray([1, 3, 5, 7])
    w = np.asarray([1, 1, 1, 1])
    w = w / np.sum(w)
    median = math_utils.weighted_median(x, w)
    assert median == 4

    # unit unit is greater than 50% of the weights
    x = np.array([0, 10, 20])
    w = np.array([60, 10, 30])
    w = w / np.sum(w)
    assert math_utils.weighted_median(x, w) == 0

    # testing one unit only
    x = np.array([10])
    w = np.array([100])
    w = w / np.sum(w)
    assert math_utils.weighted_median(x, w) == 10

    # testing two units at exactly 50%
    x = np.array([10, 20])
    w = np.array([50, 50])
    w = w / np.sum(w)
    assert math_utils.weighted_median(x, w) == 15


def test_compute_mae():
    random_number_generator = np.random.RandomState(42)
    y_true = random_number_generator.exponential(size=100)
    y_pred = y_true + 180
    assert math_utils.compute_error(y_true, y_pred, type_="mae") == pytest.approx(180)


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_compute_mape():
    random_number_generator = np.random.RandomState(42)
    y_true = random_number_generator.exponential(size=100)
    y_pred = 1.8 * y_true
    assert math_utils.compute_error(y_true, y_pred, type_="mape") == pytest.approx(0.8)

    # if multiple true values are zero
    y_true = pd.Series(np.asarray([0, 1, 4, 0, 5, 3]))
    y_pred = pd.Series(np.asarray([10, 4, 8, 20, 5, 8]))
    mape = round((abs(1 - 4) / 1 + abs(4 - 8) / 4 + abs(5 - 5) / 5 + abs(3 - 8) / 3) / 4, 2)
    assert math_utils.compute_error(y_true, y_pred, type_="mape") == pytest.approx(mape)

    # if all true values are zero
    y_true = pd.Series(np.asarray([0, 0, 0, 0, 0, 0]))  # cast as series to generate same error exactly
    y_pred = pd.Series(np.asarray([10, 4, 8, 20, 5, 8]))
    assert math.isnan(math_utils.compute_error(y_true, y_pred, type_="mape"))


def test_compute_frac_within_pi():
    lower = np.asarray([0, 1, 4, 10, 5, 3])
    upper = np.asarray([10, 4, 8, 20, 5, 8])
    pred = np.asarray([5, 8, 5, 10, 5, 9])
    assert math_utils.compute_frac_within_pi(lower, upper, pred) == round(4 / 6, 2)


def test_compute_mean_pi_length():
    random_number_generator = np.random.RandomState(42)
    lower = random_number_generator.normal(loc=5, scale=1, size=100)
    length = random_number_generator.lognormal(mean=1, sigma=5, size=100)
    upper = lower + length
    assert math_utils.compute_mean_pi_length(lower, upper, 0) == np.mean(length).round(decimals=2)


def test_kfold_splits_empty():
    array = pd.DataFrame([])
    n_splits = 5

    with pytest.raises(ValueError):
        math_utils.get_kfold_splits(array, n_splits)


def test_kfold_splits_no_split():
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_splits = 0

    with pytest.raises(ValueError):
        math_utils.get_kfold_splits(array, n_splits)


def test_kfold_splits_one_split():
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_splits = 1

    with pytest.raises(ValueError):
        math_utils.get_kfold_splits(array, n_splits)


def test_kfold_splits_1D():
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_splits = 5

    splits = math_utils.get_kfold_splits(array, n_splits)

    assert len(splits) == 5
    assert len(splits[0][0]) == 9


def test_kfold_splits_five():
    array = [1, 2, 3, 4, 5]
    n_splits = 5

    splits = math_utils.get_kfold_splits(array, n_splits)

    assert len(splits) == 5
    assert len(splits[0][0]) == 5


def test_kfold_splits_3D():
    array = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    n_splits = 10

    splits = math_utils.get_kfold_splits(array, n_splits)

    assert len(splits) == 10
    assert len(splits[0][0]) == 4


def test_np_example():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    n_splits = 2

    splits = math_utils.get_kfold_splits(X, n_splits)

    assert len(splits) == 2
    assert len(splits[0]) == 2
    assert len(splits[0][0]) == 3


def test_df_example():
    X = pd.DataFrame([[1, 2], [3, 4], [1, 2], [3, 4]])
    n_splits = 2

    splits = math_utils.get_kfold_splits(X, n_splits)

    assert len(splits) == 2
    assert len(splits[0]) == 2
    assert len(splits[0][0]) == 3
