import logging
import math
import warnings
from abc import ABC
from collections import namedtuple

import cvxpy
import numpy as np
import pandas as pd
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

from elexmodel.handlers.data.Featurizer import Featurizer
from elexmodel.models import BaseElectionModel

warnings.filterwarnings("error", category=UserWarning, module="cvxpy")

PredictionIntervals = namedtuple("PredictionIntervals", ["lower", "upper", "conformalization"], defaults=(None,) * 3)

LOG = logging.getLogger(__name__)


class ConformalElectionModel(BaseElectionModel.BaseElectionModel, ABC):
    def __init__(self, model_settings: dict):
        super(ConformalElectionModel, self).__init__(model_settings)
        self.lambda_ = model_settings.get("lambda_", 0)

    @classmethod
    def _compute_conf_frac(cls) -> float:
        """
        Compute conformalization fraction for conformal models
        """
        raise NotImplementedError

    def fit_model(
        self,
        model: QuantileRegressionSolver,
        df_X: pd.DataFrame,
        df_y: pd.Series,
        tau: float,
        weights: pd.Series,
        normalize_weights: bool,
    ):
        """
        Fits the quantile regression for the model
        """
        X = df_X.values
        y = df_y.values
        weights = weights.values

        # normalizing weights speed up solution by a lot. However, if the relative size
        # of the smallest weight is too close to zero, it can lead to numerical instability
        # where the solver either throws a warning for inaccurate solution or breaks entirely
        # in that case, we catch the error and warning and re-run with normalize_weights false
        try:
            model.fit(
                X,
                y,
                taus=tau,
                weights=weights,
                lambda_=self.lambda_,
                fit_intercept=self.add_intercept,
            )
        except (UserWarning, cvxpy.error.SolverError):
            LOG.warning("Warning: solution was inaccurate or solver broke. Re-running with normalize_weights=False.")
            model.fit(X, y, tau_value=tau, weights=weights, lambda_=self.lambda_, normalize_weights=False)

    def get_unit_predictions(
        self, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, estimand: str, **kwargs
    ) -> pd.Series:
        """
        Produces unit level predictions. Fits quantile regression to reporting data, applies
        it to nonreporting data. The features are specified in model_settings.
        """
        self.n_train = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]
        all_units = pd.concat([reporting_units, nonreporting_units], axis=0)
        featurizer = Featurizer(self.features, self.fixed_effects)
        weights = reporting_units[f"last_election_results_{estimand}"]
        reporting_units_residuals = reporting_units[f"residuals_{estimand}"]

        x_all = featurizer.prepare_data(
            all_units, center_features=True, scale_features=False, add_intercept=self.add_intercept
        )
        reporting_units_features = featurizer.filter_to_active_features(x_all[: self.n_train])
        nonreporting_units_features = featurizer.generate_holdout_data(
            x_all[self.n_train : self.n_train + n_test]  # noqa: E203
        )

        qr = QuantileRegressionSolver()
        self.fit_model(qr, reporting_units_features, reporting_units_residuals, 0.5, weights, True)
        self.features_to_coefficients = dict(zip(featurizer.complete_features, qr.coefficients))

        preds = qr.predict(nonreporting_units_features.values).flatten()

        # multiply by total voters to get unnormalized residuals
        preds = preds * nonreporting_units[f"last_election_results_{estimand}"]

        # add in last election results to go from residual to number of votes in this election
        # max with results so that predictions are always at least ars large as actual number of votes
        preds = np.maximum(
            preds + nonreporting_units[f"last_election_results_{estimand}"], nonreporting_units[f"results_{estimand}"]
        )

        # round since we don't need the artificial precision
        return preds.round(decimals=0), None

    def get_unit_prediction_interval_bounds(
        self,
        reporting_units: pd.DataFrame,
        nonreporting_units: pd.DataFrame,
        conf_frac: float,
        alpha: float,
        estimand: str,
    ) -> PredictionIntervals:
        """
        Get unadjusted unit prediction intervals. Splits reporting data into training data and conformalization data,
        fits lower and upper quantile regression using training data and apply to both conformalization data
        and nonreporting data to get conformalization lower/upper bounds and nonreporting lower/upper bounds.
        Returns unadjusted bounds for nonreporting dat and conformalization data including bounds.
        """
        # split reporting data into training and conformalization data
        # seed is set during initialization, to make sure we always get the same
        # training/conformalization split for each alpha of one run
        reporting_units_shuffled = reporting_units.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        upper_bound = (1 + alpha) / 2
        lower_bound = (1 - alpha) / 2

        train_rows = math.floor(self.n_train * conf_frac)
        train_data = reporting_units_shuffled[:train_rows]

        # the fixed effects in train_data will be a subset of the fixed effect of reporting_units since all
        # units from one fixed effect category might be in the conformalization data. Note that we are

        # we need a new featurizer since otherwise we will continue to add intercepts to the features
        interval_featurizer = Featurizer(self.features, self.fixed_effects)
        # we need all units since we will apply the upper and lower models to the nonreporting_units also

        # so we need to make sure that they have the correct fixed effects
        all_units_shuffled = pd.concat([reporting_units_shuffled, nonreporting_units], axis=0)
        x_all = interval_featurizer.prepare_data(
            all_units_shuffled, center_features=True, scale_features=False, add_intercept=self.add_intercept
        )

        # x_all starts with the shuffled reporting units, so the first train_rows are the same as train_data
        train_data_features = interval_featurizer.filter_to_active_features(x_all[:train_rows])

        train_data_residuals = train_data[f"residuals_{estimand}"]
        train_data_weights = train_data[f"last_election_results_{estimand}"]

        # fit lower and upper model to training data. ECOS solver is better than SCS.
        lower_qr = QuantileRegressionSolver()
        self.fit_model(lower_qr, train_data_features, train_data_residuals, lower_bound, train_data_weights, True)

        upper_qr = QuantileRegressionSolver()
        self.fit_model(upper_qr, train_data_features, train_data_residuals, upper_bound, train_data_weights, True)

        # apply to conformalization data. Conformalization bounds will later tell us how much to adjust lower/upper
        # bounds for nonreporting data.
        conformalization_data = reporting_units_shuffled[train_rows:].reset_index(drop=True)

        # all_data starts with reporting_units_shuffled, so the rows between train_rows and n_train are the
        # conformalization set
        conformalization_data_features = interval_featurizer.generate_holdout_data(
            x_all[train_rows : self.n_train]  # noqa: E203
        )

        # we are interested in f(X) - r
        # since later conformity scores care about deviation of bounds from residuals
        conformalization_lower_bounds = (
            lower_qr.predict(conformalization_data_features.values).flatten()
            - conformalization_data[f"residuals_{estimand}"].values
        )
        conformalization_upper_bounds = (
            conformalization_data[f"residuals_{estimand}"].values
            - upper_qr.predict(conformalization_data_features.values).flatten()
        )

        # save conformalization bounds for later
        conformalization_data["upper_bounds"] = conformalization_upper_bounds
        conformalization_data["lower_bounds"] = conformalization_lower_bounds

        # apply lower/upper models to nonreporting data
        # since nonreporting_units is the second dataframe in a_all, all units after n_train are nonreporting
        # note: the features used may be different from the median predictions, but this guarantees that the features
        # are the same accross train_data, conformalization_data and nonreporting_units
        nonreporting_units_features = interval_featurizer.generate_holdout_data(x_all[self.n_train :])  # noqa: E203

        nonreporting_lower_bounds = lower_qr.predict(nonreporting_units_features.values).flatten()
        nonreporting_upper_bounds = upper_qr.predict(nonreporting_units_features.values).flatten()

        return PredictionIntervals(nonreporting_lower_bounds, nonreporting_upper_bounds, conformalization_data)

    @classmethod
    def get_all_conformalization_data_unit(cls):
        """
        Returns conformalization data at the unit level
        Conformalization data is adjustment from estimated % change from baseline
        """
        raise NotImplementedError

    @classmethod
    def get_all_conformalization_data_agg(cls):
        """
        Returns conformalization data at the aggregate level
        """
        raise NotImplementedError

    def get_national_summary_estimates(self, nat_sum_data_dict, called_states, base_to_add, alpha):
        raise NotImplementedError()
