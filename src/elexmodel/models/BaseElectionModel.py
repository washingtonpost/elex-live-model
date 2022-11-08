import logging
import math
import warnings
from collections import namedtuple

import cvxpy
import pandas as pd
import numpy as np
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

warnings.filterwarnings("error", category=UserWarning, module="cvxpy")

PredictionIntervals = namedtuple("PredictionIntervals", ["lower", "upper", "conformalization"], defaults=(None,) * 3)

LOG = logging.getLogger(__name__)


class BaseElectionModel(object):
    def __init__(self, model_settings={}):
        self.qr = QuantileRegressionSolver(solver="ECOS")
        self.features = (
            ["intercept"] + model_settings.get("features", []) + model_settings.get("expanded_fixed_effects", [])
        )
        self.seed = 4191  # set arbitrarily

    def fit_model(self, model, df_X, df_y, tau, weights, normalize_weights):
        """
        Fits the quantile regression for the model
        Removes two kinds of columns and then substitutes zeroes as those coefficients:
            Zero columns: these are caused by  fixed effects that don't appear in df_X.
                The dummy variables were created using all data (not just the reporting data).
                So the corresponding columns for fixed ffects that only appear in non-reporting units
                will be all zero in the reporting matrix.
            Duplicate columns: in case we accidentally specify a covariate twice. If a fixed effect
                column is all ones then it is a duplicate of the intercept column and so will be removed.
        """
        # identify columns to keep. we keep the non-zero ones
        columns_to_keep_duplicated = ~df_X.transpose().duplicated(keep="first")
        columns_to_keep_zero = df_X.sum(0) != 0
        columns_to_keep = columns_to_keep_duplicated & columns_to_keep_zero
        X = df_X.loc[:, columns_to_keep].values
        y = df_y.values
        weights = weights.values

        # normalizing weights speed up solution by a lot. However, if the relative size
        # of the smallest weight is too close to zero, it can lead to numerical instability
        # where the solver either throws a warning for inaccurate solution or breaks entirely
        # in that case, we catch the error and warning and re-run with normalize_weights false
        try:
            model.fit(X, y, tau_value=tau, weights=weights, normalize_weights=normalize_weights)
        except (UserWarning, cvxpy.error.SolverError):
            LOG.warning("Warning: solution was inaccurate or solver broke. Re-running with normalize_weights=False.")
            model.fit(X, y, tau_value=tau, weights=weights, normalize_weights=False)

        # generate new coefficient matrix with zeroes for all coefficents
        coefficients = np.zeros((df_X.shape[1],))
        # replace non-zero coefficients
        coefficients[columns_to_keep] = model.coefficients
        model.coefficients = coefficients

    def get_unit_predictions(self, reporting_units, nonreporting_units, estimand):
        """
        Produces unit level predictions. Fits quantile regression to reporting data, applies
        it to nonreporting data. The features are specified in model_settings.
        """
        reporting_units_features = reporting_units[self.features]
        nonreporting_units_features = nonreporting_units[self.features]

        weights = reporting_units[f"total_voters_{estimand}"]
        reporting_units_residuals = reporting_units[f"residuals_{estimand}"]

        self.fit_model(self.qr, reporting_units_features, reporting_units_residuals, 0.5, weights, True)

        preds = self.qr.predict(nonreporting_units_features)

        # multiply by total voters to get unnormalized residuals
        preds = preds * nonreporting_units[f"total_voters_{estimand}"]

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

    def compute_feature_importances(self, train_data, alpha, estimand, features):
        upper_bound = (1 + alpha) / 2
        lower_bound = (1 - alpha) / 2

        train_data_features = train_data[self.features]
        train_data_residuals = train_data[f"residuals_{estimand}"]
        train_data_weights = train_data[f"total_voters_{estimand}"]

        lower_qr = QuantileRegressionSolver(solver="ECOS")
        self.fit_model(lower_qr, train_data_features, train_data_residuals, lower_bound, train_data_weights, True)

        upper_qr = QuantileRegressionSolver(solver="ECOS")
        self.fit_model(upper_qr, train_data_features, train_data_residuals, upper_bound, train_data_weights, True)

        feature_importances = {}
        features = list(set(features) & set(train_data.columns))

        for feature in features:
            sorted_df = train_data.sort_values(feature)
            sorted_residuals = sorted_df[f"residuals_{estimand}"].values
            sorted_features = sorted_df[self.features]
            
            n = len(sorted_residuals)
            stat_lower = (lower_bound - (sorted_residuals - lower_qr.predict(sorted_features) < 0)).cumsum() / n
            stat_upper = (upper_bound - (sorted_residuals - upper_qr.predict(sorted_features) < 0)).cumsum() / n

            stat_lower = np.mean((stat_lower**2).values[:-1])
            stat_upper = np.mean((stat_upper**2).values[:-1])

            feature_importances[feature] = np.mean([stat_lower, stat_upper])
        
        return feature_importances

    def estimate_unit_nonreporting_coverage(self, reporting_units, nonreporting_units, alpha, estimand, features, K=2):
        """
        Estimate marginal coverage of prediction intervals by estimating covariate shift between
        reporting_units and nonreporting_units. 
        Returns estimate of marginal coverage on nonreporting_units + an 80% CI on that estimate.
        """
        prediction_intervals = self.get_unit_prediction_intervals(
            reporting_units, nonreporting_units, alpha, estimand
        )

        # cross-fit estimate of marginal coverage on nonreporting_units
        # not quite correct because the conformalization threshold is estimated
        # using all the data (but I suspect this is not a big deal)

        conformalization_folds = np.array_split(prediction_intervals.conformalization, K)
        nonreporting_folds = np.array_split(nonreporting_units.sample(frac=1, random_state=self.seed), K)

        estimates = []
        variances = []
        for i in range(K):
            training_observed = pd.concat(conformalization_folds[:i] + conformalization_folds[(i + 1):])
            training_unobserved = pd.concat(nonreporting_folds[:i] + nonreporting_folds[(i + 1):])


            est, var = self._estimate_miscoverage_one_step(training_observed, training_unobserved,
                conformalization_folds[i], nonreporting_folds[i], estimand, features
            )
            estimates.append(est * (conformalization_folds[i].shape[0] + nonreporting_folds[i].shape[0]))
            variances.append(var * (conformalization_folds[i].shape[0] + nonreporting_folds[i].shape[0]))
        
        n = prediction_intervals.conformalization.shape[0] + nonreporting_units.shape[0]
        coverage_est = 1 - np.sum(estimates) / n
        std_dev = np.sqrt(np.sum(variances) / n) / np.sqrt(n)
        return np.clip(coverage_est, 0, 1), std_dev
    
    def _estimate_miscoverage_one_step(
        self, 
        training_observed, 
        training_unobserved, 
        test_observed, 
        test_unobserved,
        estimand,
        features
    ):
        """
        Implements Algorithm 1 of Qiu et al. (2021) - aka estimator in PredSet-1Step
        """
        training_miscoverage = np.maximum(training_observed.lower_bounds_c, training_observed.upper_bounds_c) > 0
        test_miscoverage = np.maximum(test_observed.lower_bounds_c, test_observed.upper_bounds_c) > 0

        if np.sum(training_miscoverage) == 0:
            return 0, 0
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        features = list(set(features) & set(training_observed.columns))

        clf_1 = make_pipeline(StandardScaler(), SVC(gamma = 'auto', probability=True, random_state=0))
        clf_2 = make_pipeline(StandardScaler(), SVC(gamma = 'auto', probability=True, random_state=0))
        miscoverage_model = clf_1.fit(X = training_observed[features], y = training_miscoverage)
        propensity_model = clf_2.fit(X = pd.concat([training_observed[features], training_unobserved[features]]),
            y = np.concatenate([np.ones(training_observed.shape[0]), np.zeros(training_unobserved.shape[0])]))
        gamma = training_observed.shape[0] / (training_observed.shape[0] + training_unobserved.shape[0])

        # term 1
        miscoverage_pred_u = miscoverage_model.predict_proba(test_unobserved[features])[:,1]

        # term 2
        miscoverage_pred_o = miscoverage_model.predict_proba(test_observed[features])[:,1]
        propensity_pred_o = propensity_model.predict_proba(test_observed[features])[:,1]

        propensity_score = (gamma / (1 - gamma)) * ((1 - propensity_pred_o) / propensity_pred_o)
        derivative =  propensity_score * (test_miscoverage - miscoverage_pred_o)
        return np.mean(miscoverage_pred_u) + np.mean(derivative), np.mean(derivative**2)

    def get_unit_prediction_interval_bounds(self, reporting_units, nonreporting_units, conf_frac, alpha, estimand):
        """
        Get unadjusted unit prediction intervals. Splits reporting data into training data and conformalization data,
        fits lower and upper quantile regression using training data and apply to both conformalization data
        and nonreporting data to get conformalization lower/upper bounds and nonreporting lower/upper bounds.
        Returns unadjusted bounds for nonreporting dat and conformalization data including bounds.
        """
        # split reporting data into training and conformalization data
        # seed is set during initialization, to make sure we always get the same training/conformalization split for each alpha of one run
        reporting_units_shuffled = reporting_units.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        upper_bound = (1 + alpha) / 2
        lower_bound = (1 - alpha) / 2

        train_rows = math.floor(reporting_units.shape[0] * conf_frac)
        train_data = reporting_units_shuffled[:train_rows].reset_index(drop=True)

        train_data_features = train_data[self.features]
        train_data_residuals = train_data[f"residuals_{estimand}"]
        train_data_weights = train_data[f"total_voters_{estimand}"]

        # fit lower and upper model to training data. ECOS solver is better than SCS.
        lower_qr = QuantileRegressionSolver(solver="ECOS")
        self.fit_model(lower_qr, train_data_features, train_data_residuals, lower_bound, train_data_weights, True)

        upper_qr = QuantileRegressionSolver(solver="ECOS")
        self.fit_model(upper_qr, train_data_features, train_data_residuals, upper_bound, train_data_weights, True)

        # apply to conformalization data. Conformalization bounds will later tell us how much to adjust lower/upper
        # bounds for nonreporting data.
        conformalization_data = reporting_units_shuffled[train_rows:].reset_index(drop=True)
        conformalization_data_features = conformalization_data[self.features].values
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
        nonreporting_units_features = nonreporting_units[self.features].values
        nonreporting_lower_bounds = lower_qr.predict(nonreporting_units_features)
        nonreporting_upper_bounds = upper_qr.predict(nonreporting_units_features)

        return PredictionIntervals(nonreporting_lower_bounds, nonreporting_upper_bounds, conformalization_data)

    def get_unit_prediction_intervals(self):
        pass

    def get_aggregate_prediction_intervals(self):
        pass

    def get_coefficients(self):
        """
        Returns quantile regression coefficients for prediction model (not lower/upper models)
        """
        return self.qr.coefficients
