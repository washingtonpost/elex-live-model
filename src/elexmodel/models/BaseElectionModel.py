import logging
import math
import warnings
from collections import namedtuple
from itertools import combinations

import cvxpy
import numpy as np
import pandas as pd
import scipy.stats as st
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

from elexmodel.handlers.data.Featurizer import Featurizer

warnings.filterwarnings("error", category=UserWarning, module="cvxpy")

PredictionIntervals = namedtuple("PredictionIntervals", ["lower", "upper", "conformalization"], defaults=(None,) * 3)

LOG = logging.getLogger(__name__)

nat_sum_states_called = (
    pd.read_csv("data_for_agg_model/national_summary_states_called.csv").drop("state", axis=1).dropna()
)
nat_sum_cov_matrix_data = pd.read_csv("data_for_agg_model/covar_matrix_data.csv").dropna()


class BaseElectionModel(object):
    def __init__(self, model_settings={}):
        self.qr = QuantileRegressionSolver(solver="ECOS")
        self.features = model_settings.get("features", [])
        self.fixed_effects = model_settings.get("fixed_effects", [])
        self.featurizer = Featurizer(self.features, self.fixed_effects)
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
            model.fit(X, y, tau_value=tau, weights=weights, normalize_weights=normalize_weights)
        except (UserWarning, cvxpy.error.SolverError):
            LOG.warning("Warning: solution was inaccurate or solver broke. Re-running with normalize_weights=False.")
            model.fit(X, y, tau_value=tau, weights=weights, normalize_weights=False)

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
        reporting_units_features = self.featurizer.featurize_fitting_data(reporting_units)
        nonreporting_units_features = self.featurizer.featurize_heldout_data(nonreporting_units)

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

        # specifying self.features extracts the correct columns and makes sure they are in the correct
        # order. Necessary when fitting and predicting on the model.
        # the fixed effects in train_data will be a subset of the fixed effect of reporting_units since all
        # units from one fixed effect category might be in the conformalization data. Note that we are
        # overwritting featurizer.expanded_fixed_effects by doing this (which is what we want), since we
        # want the expanded_fixed_effects from train_data to be used by conformalization_data and nonreporting_data
        # in this function.
        train_data_features = self.featurizer.featurize_fitting_data(train_data)
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
        Returns quantile regression coefficients for prediction model (not lower/upper models)
        """
        return self.qr.coefficients

    def extend_str_with_list(self, string, list_to_add):
        return [f"{string}_{str(item)}" for item in list_to_add]

    def random_draws(self, mean_vec_dict, cov_matrix_dict, estimands, num_observations):
        draws_dict = {}
        for estimand in estimands:
            draws_dict[estimand] = pd.DataFrame(
                np.random.multivariate_normal(mean_vec_dict[estimand], cov_matrix_dict[estimand], size=num_observations)
            ).transpose()

        total_votes_matrix = sum(draws_dict.values())
        shares_dict = {estimand: draws_dict[estimand] / total_votes_matrix for estimand in estimands}
        return shares_dict

    def get_national_summary_count_trials(
        self,
        state_preds,
        estimands,
        #  primary_estimand,
        agg_model_states_not_used,
        trials,
        num_observations,
        nat_sum_data_dict,
        alpha=0.9,
    ):

        #  states_called = dict(zip(list(ecv_states_called["postal_code"]), list(ecv_states_called["called"])))
        # only make predictions for states that we want in the model
        # (i.e. those in preprocessed data)
        state_preds = state_preds[~state_preds["postal_code"].isin(agg_model_states_not_used)].reset_index()
        mean_vec_dict = {estimand: list(state_preds[f"pred_{estimand}"]) for estimand in estimands}

        cov_matrix_dict = self.construct_cov_matrix_dict(state_preds, estimands)
        wins_df = state_preds[["postal_code"]]

        nat_sum_vote_totals_by_trial = {estimand: [] for estimand in estimands}

        for k in range(trials):
            estimand_shares_dict = self.random_draws(mean_vec_dict, cov_matrix_dict, estimands, num_observations)
            for estimand in estimands:
                wins_df[["obs_" + str(n) for n in range(num_observations)]] = estimand_shares_dict[estimand]
                wins_df["vote"] = wins_df["postal_code"].map(nat_sum_data_dict)
                wins_votes_df = wins_df

                estimand_wins = (wins_votes_df[["obs_" + str(n) for n in range(num_observations)]] > 0.5).multiply(
                    wins_votes_df["vote"], axis="index"
                )

                estimand_trial_nat_sum_mean = np.mean(list(estimand_wins.sum(axis=0)))
                nat_sum_vote_totals_by_trial[estimand].append(estimand_trial_nat_sum_mean)

        return pd.DataFrame(data=nat_sum_vote_totals_by_trial)

    def get_national_summary_vote_estimates(
        self,
        state_preds,
        estimands,
        agg_model_states_not_used,
        num_observations,
        ci_method,
        alpha,
        nat_sum_data_dict,
        **kwargs,
    ):
        # options for method are percentile, normal_dist (which is just standard approach for large sample size > n = 30)
        # or t- distribution
        trials = kwargs.get("trials", 1000)
        #  primary_estimand = kwargs.get("primary_estimand", "dem")

        trials_df = self.get_national_summary_count_trials(
            state_preds,
            estimands,
            #  primary_estimand,
            agg_model_states_not_used,
            trials,
            num_observations,
            nat_sum_data_dict,
        )
        if ci_method == "normal_dist_mean":
            # in case of num_oberservations = 1, this gives PI around dem nat-sum estimate iteself
            # in case of num_observations > 1, this is CI of sample mean of dem nat-sum
            est_means = trials_df.mean().round(2)
            est_sem = trials_df.sem().round(2)
            est_CI = {
                estimand: st.norm.interval(confidence=alpha, loc=est_means[estimand], scale=est_sem[estimand])
                for estimand in estimands
            }

            est_mean_CI = {
                estimand: [est_means[estimand], round(est_CI[estimand][0], 2), round(est_CI[estimand][1], 2)]
                for estimand in estimands
            }

        if ci_method == "t_dist_mean":
            # in case of num_oberservations = 1, this gives PI around dem nat-sum estimate iteself
            # in case of num_observations > 1, this is CI of sample mean of dem nat-sum
            est_means = trials_df.mean().round(2)
            est_sem = trials_df.sem().round(2)
            est_CI = {
                estimand: st.t.interval(
                    alpha=alpha, df=len(trials_df) - 1, loc=est_means[estimand], scale=est_sem[estimand]
                )
                for estimand in estimands
            }

            est_mean_CI = {
                estimand: [est_means[estimand], round(est_CI[estimand][0], 2), round(est_CI[estimand][1], 2)]
                for estimand in estimands
            }

        elif ci_method == "percentile":
            # either percentile in group of national summary draws, or percentile in distribution of nat-sum mean
            est_means = trials_df.mean().round(2)
            ci_tail = (1 - alpha) / 2
            est_lb = trials_df.quantile(ci_tail, axis=0).round(2)
            est_ub = trials_df.quantile(1 - ci_tail, axis=0).round(2)

            est_mean_CI = {
                estimand: [est_means[estimand], round(est_lb[estimand], 2), round(est_ub[estimand], 2)]
                for estimand in estimands
            }

        return est_mean_CI

    def construct_cov_matrix_dict(self, state_preds, estimands, alpha=0.9):
        var_dict = {
            estimand: (
                (state_preds[f"upper_{alpha}_{estimand}"] - state_preds[f"lower_{alpha}_{estimand}"]) / (2 * 1.645)
            )
            ** 2
            for estimand in estimands
        }

        std_dev_dict = {k: [np.sqrt(x) for x in var_dict[k]] for k in var_dict.keys()}
        std_dev_matrix_dict = {k: np.diag(std_dev_dict[k]) for k in std_dev_dict.keys()}
        states_in_use = state_preds["postal_code"]

        # construct correlation matrix, which is then used in construction
        # of covariance matrix
        blue_states = [
            state for state in list(nat_sum_cov_matrix_data["blue_state"]) if state in states_in_use.to_list()
        ]
        red_states = [state for state in list(nat_sum_cov_matrix_data["red_state"]) if state in states_in_use.to_list()]
        swing_states = [
            state for state in list(nat_sum_cov_matrix_data["swing_state"]) if state in states_in_use.to_list()
        ]

        corr_matrix = np.zeros((len(states_in_use), len(states_in_use)))

        for pair in combinations(blue_states, 2):
            pair_indices = [
                states_in_use[states_in_use == pair[0]].index[0],
                states_in_use[states_in_use == pair[1]].index[0],
            ]
            corr_matrix[pair_indices] = 0.8
        for pair in combinations(red_states, 2):
            pair_indices = [
                states_in_use[states_in_use == pair[0]].index[0],
                states_in_use[states_in_use == pair[1]].index[0],
            ]
            corr_matrix[pair_indices] = 0.9
        for pair in combinations(swing_states, 2):
            pair_indices = [
                states_in_use[states_in_use == pair[0]].index[0],
                states_in_use[states_in_use == pair[1]].index[0],
            ]
            corr_matrix[pair_indices] = 0.7

        np.fill_diagonal(corr_matrix, 1)

        cov_matrix_dict = {
            estimand: std_dev_matrix_dict[estimand] @ corr_matrix @ std_dev_matrix_dict[estimand]
            for estimand in estimands
        }

        # cov_matrix = np.identity(n = len(states_in_use))
        cov_matrix_dict = {estimand: np.diag(var_dict[estimand]) for estimand in estimands}
        return cov_matrix_dict
