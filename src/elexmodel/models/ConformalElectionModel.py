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
        self.qr = QuantileRegressionSolver(solver="ECOS")
        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.lambda_ = model_settings.get("lambda_", 0)

    @classmethod
    def _compute_conf_frac(self) -> float:
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
                tau_value=tau,
                weights=weights,
                lambda_=self.lambda_,
                fit_intercept=self.add_intercept,
                normalize_weights=normalize_weights,
            )
        except (UserWarning, cvxpy.error.SolverError):
            LOG.warning("Warning: solution was inaccurate or solver broke. Re-running with normalize_weights=False.")
            model.fit(X, y, tau_value=tau, weights=weights, lambda_=self.lambda_, normalize_weights=False)

    def get_unit_predictions(
        self, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, estimand: str
    ) -> pd.Series:
        """
        Produces unit level predictions. Fits quantile regression to reporting data, applies
        it to nonreporting data. The features are specified in model_settings.
        """
        # compute the means of both reporting_units and nonreporting_units for centering (part of featurizing)
        # we want them both, since together they are the subunit population
        self.featurizer.compute_means_for_centering(reporting_units, nonreporting_units)
        # reporting_units_features and nonreporting_units_features should have the same
        # features. Specifically also the same fixed effect columns.
        reporting_units_features = self.featurizer.featurize_fitting_data(
            reporting_units, add_intercept=self.add_intercept
        )
        nonreporting_units_features = self.featurizer.featurize_heldout_data(nonreporting_units)

        weights = reporting_units[f"last_election_results_{estimand}"]
        reporting_units_residuals = reporting_units[f"residuals_{estimand}"]

        self.fit_model(self.qr, reporting_units_features, reporting_units_residuals, 0.5, weights, True)
        self.features_to_coefficients = dict(zip(self.featurizer.complete_features, self.qr.coefficients))

        preds = self.qr.predict(nonreporting_units_features)

        # multiply by total voters to get unnormalized residuals
        preds = preds * nonreporting_units[f"last_election_results_{estimand}"]

        # add in last election results to go from residual to number of votes in this election
        # max with results so that predictions are always at least ars large as actual number of votes
        preds = np.maximum(
            preds + nonreporting_units[f"last_election_results_{estimand}"], nonreporting_units[f"results_{estimand}"]
        )

        # round since we don't need the artificial precision
        return preds.round(decimals=0)

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

        train_rows = math.floor(reporting_units.shape[0] * conf_frac)
        train_data = reporting_units_shuffled[:train_rows].reset_index(drop=True)

        # specifying self.features extracts the correct columns and makes sure they are in the correct
        # order. Necessary when fitting and predicting on the model.
        # the fixed effects in train_data will be a subset of the fixed effect of reporting_units since all
        # units from one fixed effect category might be in the conformalization data. Note that we are
        # overwritting featurizer.expanded_fixed_effects by doing this (which is what we want), since we
        # want the expanded_fixed_effects from train_data to be used by conformalization_data and nonreporting_data
        # in this function.
        train_data_features = self.featurizer.featurize_fitting_data(train_data, add_intercept=self.add_intercept)
        train_data_residuals = train_data[f"residuals_{estimand}"]
        train_data_weights = train_data[f"last_election_results_{estimand}"]

        # fit lower and upper model to training data. ECOS solver is better than SCS.
        lower_qr = QuantileRegressionSolver(solver="ECOS")
        self.fit_model(lower_qr, train_data_features, train_data_residuals, lower_bound, train_data_weights, True)

        upper_qr = QuantileRegressionSolver(solver="ECOS")
        self.fit_model(upper_qr, train_data_features, train_data_residuals, upper_bound, train_data_weights, True)

        # apply to conformalization data. Conformalization bounds will later tell us how much to adjust lower/upper
        # bounds for nonreporting data.
        conformalization_data = reporting_units_shuffled[train_rows:].reset_index(drop=True)
        # conformalization features will be the same as the features in train_data
        conformalization_data_features = self.featurizer.featurize_heldout_data(conformalization_data)

        # we are interested in f(X) - r
        # since later conformity scores care about deviation of bounds from residuals
        conformalization_lower_bounds = (
            lower_qr.predict(conformalization_data_features) - conformalization_data[f"residuals_{estimand}"].values
        )
        conformalization_upper_bounds = conformalization_data[f"residuals_{estimand}"].values - upper_qr.predict(
            conformalization_data_features
        )

        # save conformalization bounds for later
        conformalization_data["upper_bounds"] = conformalization_upper_bounds
        conformalization_data["lower_bounds"] = conformalization_lower_bounds

        # apply lower/upper models to nonreporting data
        # now the features of the nonreporting_units will be the same as the train_data features
        # they might differ slightly from the features used when fitting the median prediction
        nonreporting_units_features = self.featurizer.featurize_heldout_data(nonreporting_units)
        nonreporting_lower_bounds = lower_qr.predict(nonreporting_units_features)
        nonreporting_upper_bounds = upper_qr.predict(nonreporting_units_features)

        return PredictionIntervals(nonreporting_lower_bounds, nonreporting_upper_bounds, conformalization_data)

    @classmethod
    def get_all_conformalization_data_unit(self):
        """
        Returns conformalization data at the unit level
        Conformalization data is adjustment from estimated % change from baseline
        """
        raise NotImplementedError

    @classmethod
    def get_all_conformalization_data_agg(self):
        """
        Returns conformalization data at the aggregate level
        """
        raise NotImplementedError

    def get_coefficients(self):
        """
        Returns a dictionary of feature/fixed effect names to the quantile regression coefficients
        These coefficients are for the point prediciton only, not for the lower or upper intervals models.
        """
        return self.features_to_coefficients
