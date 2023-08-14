import logging
import math
import warnings
from collections import namedtuple

import cvxpy
import numpy as np
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

from elexmodel.handlers.data.Featurizer import Featurizer

warnings.filterwarnings("error", category=UserWarning, module="cvxpy")

PredictionIntervals = namedtuple("PredictionIntervals", ["lower", "upper", "conformalization"], defaults=(None,) * 3)

LOG = logging.getLogger(__name__)


class BaseElectionModel:
    def __init__(self, model_settings={}):
        self.qr = QuantileRegressionSolver(solver="ECOS")
        self.features = model_settings.get("features", [])
        self.fixed_effects = model_settings.get("fixed_effects", {})
        self.lambda_ = model_settings.get("lambda_", 0)
        self.features_to_coefficients = {}
        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.add_intercept = True
        self.seed = 4191  # set arbitrarily

    def fit_model(self, model, df_X, df_y, tau, weights, normalize_weights):
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

    def get_unit_predictions(self, reporting_units, nonreporting_units, estimand):
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

    def _get_reporting_aggregate_votes(self, reporting_units, unexpected_units, aggregate, estimand):
        """
        Aggregate reporting votes by aggregate (ie. postal_code, county_fips etc.). This function
        adds reporting data and reporting unexpected data by aggregate. Note that all unexpected units -
        whether or not they are fully reporting - are included in this function.
        """
        reporting_units_known_votes = reporting_units.groupby(aggregate).sum().reset_index(drop=False)

        # we cannot know the county classification of unexpected geographic units, so we can't add the votes back in
        if "county_classification" in aggregate:
            aggregate_votes = reporting_units_known_votes[aggregate + [f"results_{estimand}", "reporting"]]
        else:
            unexpected_units_known_votes = unexpected_units.groupby(aggregate).sum().reset_index(drop=False)

            # outer join to ensure that if entire districts of county classes are unexpectedly present, we
            # should still have them. Same reasoning to replace NA with zero
            # NA means there was no such geographic unit, so they don't capture any votes
            results_col = f"results_{estimand}"
            reporting_col = "reporting"
            aggregate_votes = (
                reporting_units_known_votes.merge(
                    unexpected_units_known_votes,
                    how="outer",
                    on=aggregate,
                    suffixes=("_expected", "_unexpected"),
                )
                .fillna(
                    {
                        f"results_{estimand}_expected": 0,
                        f"results_{estimand}_unexpected": 0,
                        "reporting_expected": 0,
                        "reporting_unexpected": 0,
                    }
                )
                .assign(
                    **{
                        results_col: lambda x: x[f"results_{estimand}_expected"] + x[f"results_{estimand}_unexpected"],
                        reporting_col: lambda x: x["reporting_expected"] + x["reporting_unexpected"],
                    },
                )[aggregate + [f"results_{estimand}", "reporting"]]
            )

        return aggregate_votes

    def _get_nonreporting_aggregate_votes(self, nonreporting_units, aggregate):
        """
        Aggregate nonreporting votes by aggregate (ie. postal_code, county_fips etc.). Note that all unexpected
        units - whether or not they are fully reporting - are handled in "_get_reporting_aggregate_votes" above
        """
        aggregate_nonreporting_units_known_votes = nonreporting_units.groupby(aggregate).sum().reset_index(drop=False)

        return aggregate_nonreporting_units_known_votes

    def get_aggregate_predictions(self, reporting_units, nonreporting_units, unexpected_units, aggregate, estimand):
        """
        Aggregate predictions and results by aggregate (ie. postal_code, county_fips etc.). Add results from reporting
        and reporting unexpected units and then sum in the predictions from nonreporting units.
        """
        # these are subunits that are already counted
        aggregate_votes = self._get_reporting_aggregate_votes(reporting_units, unexpected_units, aggregate, estimand)

        # these are subunits that are not already counted
        aggregate_preds = (
            nonreporting_units.groupby(aggregate)
            .sum()
            .reset_index(drop=False)
            .rename(
                columns={
                    f"pred_{estimand}": f"pred_only_{estimand}",
                    f"results_{estimand}": f"results_only_{estimand}",
                    "reporting": "reporting_only",
                }
            )
        )
        aggregate_data = (
            aggregate_votes.merge(aggregate_preds, how="outer", on=aggregate)
            .fillna(
                {
                    f"results_{estimand}": 0,
                    f"pred_only_{estimand}": 0,
                    f"results_only_{estimand}": 0,
                    "reporting": 0,
                    "reporting_only": 0,
                }
            )
            .assign(
                # don't need to sum results_only for predictions since those are superceded by pred_only
                # preds can't be smaller than results, since we maxed between predictions and results in unit function.
                **{
                    f"pred_{estimand}": lambda x: x[f"results_{estimand}"] + x[f"pred_only_{estimand}"],
                    f"results_{estimand}": lambda x: x[f"results_{estimand}"] + x[f"results_only_{estimand}"],
                    "reporting": lambda x: x["reporting"] + x["reporting_only"],
                },
            )
            .sort_values(aggregate)[aggregate + [f"pred_{estimand}", f"results_{estimand}", "reporting"]]
            .reset_index(drop=True)
        )

        return aggregate_data

    def get_unit_prediction_interval_bounds(self, reporting_units, nonreporting_units, conf_frac, alpha, estimand):
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

    def get_unit_prediction_intervals(self):
        pass

    def get_aggregate_prediction_intervals(self):
        pass

    def get_coefficients(self):
        """
        Returns a dictionary of feature/fixed effect names to the quantile regression coefficients
        These coefficients are for the point prediciton only, not for the lower or upper intervals models.
        """
        return self.features_to_coefficients
