import logging
import math

import numpy as np
from scipy.stats import bootstrap
from scipy.stats.mstats import winsorize

LOG = logging.getLogger()


def compute_inflate(x):
    """
    Compute inflation factor. sum of squared divided by square of sum
    """
    return np.sum(np.power(x, 2)) / np.power(np.sum(x), 2)


def sample_std(x, axis):
    """
    Sample standard deviation
    """
    # ddof=1 to get unbiased sample estimate.
    return np.std(x, ddof=1, axis=-1)


def winsorize_std(x, axis):
    """
    Compute the winsorized standard deviation along the last axis. Limits
    are used to trim 5% of the extreme values on both ends of the data.
    """
    x_win = winsorize(x, limits=(0.05, 0.05), axis=-1).data
    return np.std(x_win, ddof=1, axis=-1)


def weighted_median(x, weights):
    """
    Compute weighted median. This function expectes weights to sum to 1.
    """
    # TODO: implement removing outliers

    # sort elements and weights by elements
    indices_sorted = np.argsort(x)
    x_sorted = x[indices_sorted]
    weights_sorted = weights[indices_sorted]

    # find index of largest x_i where weights are less than or equal 0.5
    weights_cumulative = np.cumsum(weights_sorted)

    # x-values are lined up in size order, but each is assigned a
    # weight based on unit population. The list is split in half according to
    # cumulative weights. But if the first element in the list is already over
    # 50% of total weight, there will be nothing in one side of the list. In
    # that case return the first element
    if weights_cumulative[0] > 0.5:
        LOG.warning("Warning: smallest x-value is greater than or equal to half the weight")
        return x_sorted[0]
    median_index = np.where(weights_cumulative <= 0.5)[0][-1]

    # if there is one element where weights are exactly 0.5, median is average
    # otherwise weighted median is the next largest element
    if weights_cumulative[median_index] == 0.5:
        lower = x_sorted[median_index]
        upper = x_sorted[median_index + 1]
        return (lower + upper) / 2
    return x_sorted[median_index + 1]


def robust_sample_std(x, axis):
    """
    Compute the robust sample standard deviation along the last axis by calling winsorize_std.
    """
    return winsorize_std(x, axis=-1)


def boot_sigma(data, conf, num_iterations=10000, winsorize=False):
    """
    Bootstrap standard deviation.
    """
    # we use upper bound of confidence interval for more robustness
    if winsorize:
        std_func = robust_sample_std
    else:
        std_func = sample_std

    return bootstrap(
        data.reshape(1, -1), std_func, confidence_level=conf, method="basic", n_resamples=num_iterations
    ).confidence_interval.high


def compute_error(true, pred, type_="mae"):
    """
    computes error. either mean absolute error or mean absolute percentage error
    """
    if type_ == "mae":
        return np.mean(np.abs(true - pred))
    if type_ == "mape":
        mask = true != 0
        mape = np.mean((np.abs((true - pred) / true))[mask])
        # if all true values are zero, then race was uncontested and mape doesn't make sense to compute
        if math.isnan(mape):
            return mape
        return mape


def compute_frac_within_pi(lower, upper, results):
    """
    computes coverage of prediction intervals.
    """
    return np.mean((upper >= results) & (lower <= results))


def compute_mean_pi_length(lower, upper, pred):
    """
    computes average relative length of prediction interval
    """
    return np.mean(np.abs(np.nan_to_num((upper - lower) / pred)))
