import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import linprog
from scipy.stats import norm

from elexmodel.handlers.data.Featurizer import Featurizer
from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals


class OLSRegression(object):
    def __init__(self):
        self.normal_eqs = None
        self.hat_matrix = None
        self.beta_hat = None

    def _compute_normal_equations(self, x, L):
        Q, R = np.linalg.qr(L @ x)
        return np.linalg.inv(R.T @ R) @ R.T @ Q.T

    def fit(self, x, y, weights=None, normal_eqs=None):
        if weights is None:
            weights = np.ones((y.shape[0],))
        L = np.diag(np.sqrt(weights.flatten()))
        if normal_eqs is not None:
            self.normal_eqs = normal_eqs
        else:
            self.normal_eqs = self._compute_normal_equations(x, L)
        self.hat_vals = np.diag(x @ self.normal_eqs)
        self.beta_hat = self.normal_eqs @ L @ y
        return self

    def predict(self, x):
        return x @ self.beta_hat

    def residuals(self, y, y_hat, loo=True, center=True):
        residuals = y - y_hat
        if loo:
            residuals /= (1 - self.hat_vals).reshape(-1, 1)
        if center:
            residuals -= np.mean(residuals, axis=0)
        return residuals


class QuantileRegression(object):
    def __init__(self):
        self.beta_hats = []

    def _fit(self, S, Phi, zeros, N, weights, tau):
        bounds = weights.reshape(-1, 1) * np.asarray([(tau - 1, tau)] * N)
        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, method="highs", options={"presolve": False})
        return -1 * res.eqlin.marginals

    def fit(self, x, y, taus=0.5, weights=None):
        if weights is None:
            weights = np.ones((y.shape[0],))

        S = y
        Phi = x
        zeros = np.zeros((Phi.shape[1],))
        N = y.shape[0]
        weights /= np.sum(weights)

        if isinstance(taus, float):
            taus = [taus]

        for tau in taus:
            self.beta_hats.append(self._fit(S, Phi, zeros, N, weights, tau))

        return self.beta_hats


class BootstrapElectionModel(BaseElectionModel):
    y_LB = -0.3
    y_UB = 0.3
    z_LB = -0.5
    z_UB = 0.5

    def __init__(self, model_settings={}):
        super().__init__(model_settings)
        self.seed = model_settings.get("seed", 0)
        self.B = model_settings.get("B", 1000)
        self.B = 40
        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.rng = np.random.default_rng(seed=self.seed)
        self.ran_bootstrap = False

    def get_minimum_reporting_units(self, alpha):
        return 10
        # return math.ceil(-1 * (alpha + 1) / (alpha - 1))

    def _estimate_conditional_distribution(self):
        pass

    # TODO:
    #  robust-ify OLS (e.g. regularization, quantile -> one bad AP county does not break everything)
    #  account for partial reporting in UB/LB
    #  implement aggregation
    def compute_bootstrap_errors(self, reporting_units, nonreporting_units, unexpected_units):
        n_train = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]

        self.featurizer.compute_means_for_centering(reporting_units, nonreporting_units)
        x_train = self.featurizer.featurize_fitting_data(
            reporting_units, add_intercept=self.add_intercept, center_features=False
        )
        y_train = reporting_units["normalized_margin"]
        z_train = reporting_units["turnout_factor"]
        df_train = pd.concat([x_train, y_train, z_train], axis=1)

        df_test = self.featurizer.featurize_heldout_data(nonreporting_units)
        weights_test = nonreporting_units["weights"].values.reshape(-1, 1)

        fixed_effect_string = "+".join(self.featurizer.expanded_fixed_effects)
        md_y = smf.mixedlm(
            f"normalized_margin ~ baseline_normalized_margin + {fixed_effect_string}",
            df_train,
            groups=reporting_units["postal_code"],
        )
        md_z = smf.mixedlm(
            f"turnout_factor ~ baseline_normalized_margin + {fixed_effect_string}",
            df_train,
            groups=reporting_units["postal_code"],
        )
        mdf_y = md_y.fit()
        mdf_z = md_z.fit()

        y_scale = mdf_y.scale
        z_scale = mdf_z.scale
        cov_mat_eps = [[mdf_y.cov_re.Group.item(), 0], [0, mdf_z.cov_re.Group.item()]]
        var_epsilon_y_dict = {key: value.Group.item() for (key, value) in mdf_y.random_effects_cov.items()}
        var_epsilon_z_dict = {key: value.Group.item() for (key, value) in mdf_z.random_effects_cov.items()}
        epsilon_y_hat_dict = {key: value.item() for (key, value) in mdf_y.random_effects.items()}
        epsilon_z_hat_dict = {key: value.item() for (key, value) in mdf_z.random_effects.items()}

        aggregate_indicator = pd.get_dummies(
            pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)["postal_code"]
        )
        aggregate_indicator_expected = aggregate_indicator.iloc[: (n_train + n_test)]
        aggregate_indicator_train = aggregate_indicator_expected.iloc[:n_train]
        aggregate_indicator_test = aggregate_indicator_expected.iloc[n_train:]

        epsilon_y_hat = aggregate_indicator_train.sum(axis=0)
        epsilon_z_hat = aggregate_indicator_train.sum(axis=0)
        epsilon_y_hat[epsilon_y_hat_dict.keys()] = list(epsilon_y_hat_dict.values())
        epsilon_z_hat[epsilon_z_hat_dict.keys()] = list(epsilon_z_hat_dict.values())
        epsilon_y_hat = epsilon_y_hat.values.reshape(-1, 1)
        epsilon_z_hat = epsilon_z_hat.values.reshape(-1, 1)
        epsilon_hat = np.concatenate([epsilon_y_hat, epsilon_z_hat], axis=1)

        var_epsilon_y = aggregate_indicator_train.sum(axis=0)
        var_epsilon_z = aggregate_indicator_train.sum(axis=0)
        var_epsilon_y[var_epsilon_y_dict.keys()] = list(var_epsilon_y_dict.values())
        var_epsilon_z[var_epsilon_z_dict.keys()] = list(var_epsilon_z_dict.values())
        var_epsilon_y = var_epsilon_y.values
        var_epsilon_z = var_epsilon_z.values

        epsilon_y_B = self.rng.multivariate_normal(mean=epsilon_hat[:, 0], cov=np.diag(var_epsilon_y), size=self.B).T
        epsilon_z_B = self.rng.multivariate_normal(mean=epsilon_hat[:, 1], cov=np.diag(var_epsilon_z), size=self.B).T

        y_train_pred = mdf_y.fittedvalues.values.reshape(-1, 1)
        z_train_pred = mdf_z.fittedvalues.values.reshape(-1, 1)

        delta_y_B = self.rng.normal(loc=0, scale=np.sqrt(y_scale), size=(n_train, self.B))
        delta_z_B = self.rng.normal(loc=0, scale=np.sqrt(z_scale), size=(n_train, self.B))

        y_train_B = y_train_pred + (aggregate_indicator_train.values @ epsilon_y_B) + delta_y_B
        z_train_B = z_train_pred + (aggregate_indicator_train.values @ epsilon_z_B) + delta_z_B

        df_train_B = df_train.copy()

        y_test_pred_B = np.zeros((n_test, self.B))
        z_test_pred_B = np.zeros((n_test, self.B))
        for b in range(self.B):
            df_train_B["normalized_margin"] = y_train_B[:, b]
            df_train_B["turnout_factor"] = z_train_B[:, b]
            md_y_b = smf.mixedlm(
                f"normalized_margin ~ baseline_normalized_margin + {fixed_effect_string}",
                df_train_B,
                groups=reporting_units["postal_code"],
            )
            md_z_b = smf.mixedlm(
                f"turnout_factor ~ baseline_normalized_margin + {fixed_effect_string}",
                df_train_B,
                groups=reporting_units["postal_code"],
            )
            mdf_y_b = md_y_b.fit()
            mdf_z_b = md_z_b.fit()

            y_test_pred_B[:, b] = (
                mdf_y_b.predict(df_test).values + (aggregate_indicator_test.values @ epsilon_y_hat).flatten()
            )
            z_test_pred_B[:, b] = (
                mdf_z_b.predict(df_test).values + (aggregate_indicator_test.values @ epsilon_y_hat).flatten()
            )

        yz_test_pred_B = y_test_pred_B * z_test_pred_B

        y_test_pred = mdf_y.predict(df_test).values.reshape(-1, 1) + (aggregate_indicator_test.values @ epsilon_y_hat)
        z_test_pred = mdf_z.predict(df_test).values.reshape(-1, 1) + (aggregate_indicator_test.values @ epsilon_z_hat)
        yz_test_pred = y_test_pred * z_test_pred

        test_delta_y = self.rng.normal(loc=0, scale=np.sqrt(y_scale), size=(n_test, self.B))
        test_delta_z = self.rng.normal(loc=0, scale=np.sqrt(z_scale), size=(n_test, self.B))

        # gives us indices of states for which we have no samples in the training set
        states_not_in_reporting_units = np.where(np.all(aggregate_indicator_train == 0, axis=0))[0]
        # gives us states for which there is at least one county not reporting
        states_in_nonreporting_units = np.where(np.any(aggregate_indicator_test > 0, axis=0))[0]
        states_that_need_random_effect = np.intersect1d(states_not_in_reporting_units, states_in_nonreporting_units)

        test_epsilon = self.rng.multivariate_normal(
            [0, 0], cov_mat_eps, size=(len(states_that_need_random_effect), self.B)
        )
        test_epsilon_y = test_epsilon[:, :, 0]
        test_epsilon_z = test_epsilon[:, :, 1]

        test_residuals_y = (
            test_delta_y + aggregate_indicator_test.values[:, states_that_need_random_effect] @ test_epsilon_y
        )
        test_residuals_z = (
            test_delta_z + aggregate_indicator_test.values[:, states_that_need_random_effect] @ test_epsilon_z
        )

        self.errors_B_1 = yz_test_pred_B * weights_test

        errors_B_2 = (y_test_pred + test_residuals_y).clip(min=-1, max=1)
        errors_B_2 *= (z_test_pred + test_residuals_z).clip(
            min=0.5, max=1.5
        )  # clipping: predicting turnout can't be less than 50% or more than 150% of previous election

        self.errors_B_2 = errors_B_2 * weights_test

        self.errors_B_3 = z_test_pred_B.clip(min=0.5, max=1.5) * weights_test  # denominator for percentage margin
        self.errors_B_4 = (z_test_pred + test_residuals_z).clip(
            min=0.5, max=1.5
        ) * weights_test  # clipping: predicting turnout can't be less than 50% or more than 150% of previous election

        self.weighted_yz_test_pred = yz_test_pred * weights_test
        self.weighted_z_test_pred = z_test_pred * weights_test

        self.ran_bootstrap = True

    def get_unit_predictions(self, reporting_units, nonreporting_units, estimand, **kwargs):
        if not self.ran_bootstrap:
            unexpected_units = kwargs["unexpected_units"]
            self.compute_bootstrap_errors(reporting_units, nonreporting_units, unexpected_units)
        return self.weighted_yz_test_pred

    def get_aggregate_predictions(self, reporting_units, nonreporting_units, unexpected_units, aggregate, estimand):
        n = reporting_units.shape[0]
        m = nonreporting_units.shape[0]

        aggregate_indicator = pd.get_dummies(
            pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)[aggregate]
        ).values
        aggregate_indicator_expected = aggregate_indicator[: (n + m)]

        aggregate_indicator_unexpected = aggregate_indicator[(n + m) :]
        turnout_unexpected = (unexpected_units["results_dem"] + unexpected_units["results_gop"]).values.reshape(-1, 1)

        aggregate_indicator_train = aggregate_indicator_expected[:n]
        aggregate_indicator_test = aggregate_indicator_expected[n:]
        weights_train = reporting_units["weights"].values.reshape(-1, 1)
        z_train = reporting_units["turnout_factor"].values.reshape(-1, 1)

        aggregate_z_train = aggregate_indicator_train.T @ (weights_train * z_train)

        aggregate_z_unexpected = aggregate_indicator_unexpected.T @ turnout_unexpected
        aggregate_z_total = (
            aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        )

        raw_margin_df = super().get_aggregate_predictions(
            reporting_units, nonreporting_units, unexpected_units, aggregate, estimand
        )
        raw_margin_df["pred_margin"] /= aggregate_z_total.flatten() + 1
        raw_margin_df["results_margin"] /= aggregate_z_total.flatten() + 1  # avoid NaN
        return raw_margin_df

    def get_unit_prediction_intervals(self, reporting_units, non_reporting_units, alpha, estimand):
        errors_B = self.errors_B_1 - self.errors_B_2

        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        interval_upper, interval_lower = (
            self.weighted_yz_test_pred - np.quantile(errors_B, q=[lower_q, upper_q], axis=-1).T
        ).T

        interval_upper = interval_upper.reshape(-1, 1)
        interval_lower = interval_lower.reshape(-1, 1)

        return PredictionIntervals(interval_lower.round(decimals=0), interval_upper.round(decimals=0))

    def get_aggregate_prediction_intervals(
        self, reporting_units, nonreporting_units, unexpected_units, aggregate, alpha, conformalization_data, estimand
    ):
        n = reporting_units.shape[0]
        m = nonreporting_units.shape[0]

        aggregate_indicator = pd.get_dummies(
            pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)[aggregate]
        ).values
        aggregate_indicator_expected = aggregate_indicator[: (n + m)]

        aggregate_indicator_unexpected = aggregate_indicator[(n + m) :]
        margin_unexpected = unexpected_units["results_margin"].values.reshape(-1, 1)
        turnout_unexpected = (unexpected_units["results_dem"] + unexpected_units["results_gop"]).values.reshape(-1, 1)
        aggregate_z_unexpected = aggregate_indicator_unexpected.T @ turnout_unexpected
        aggregate_yz_unexpected = aggregate_indicator_unexpected.T @ margin_unexpected

        aggregate_indicator_train = aggregate_indicator_expected[:n]
        aggregate_indicator_test = aggregate_indicator_expected[n:]
        weights_train = reporting_units["weights"].values.reshape(-1, 1)
        y_train = reporting_units["normalized_margin"].values.reshape(-1, 1)
        z_train = reporting_units["turnout_factor"].values.reshape(-1, 1)

        yz_train = y_train * z_train
        aggregate_z_train = aggregate_indicator_train.T @ (weights_train * z_train)
        aggregate_yz_train = aggregate_indicator_train.T @ (weights_train * yz_train)

        aggregate_yz_test_B = aggregate_yz_train + aggregate_indicator_test.T @ self.errors_B_1
        aggregate_yz_test_pred = aggregate_yz_train + aggregate_indicator_test.T @ self.errors_B_2
        aggregate_z_test_B = aggregate_z_train + aggregate_indicator_test.T @ self.errors_B_3
        aggregate_z_test_pred = aggregate_z_train + aggregate_indicator_test.T @ self.errors_B_4

        aggregate_error_B_1 = aggregate_yz_test_B
        aggregate_error_B_2 = aggregate_yz_test_pred
        aggregate_error_B_3 = aggregate_z_test_B
        aggregate_error_B_4 = aggregate_z_test_pred

        aggregate_error_B = (aggregate_error_B_1 / aggregate_error_B_3) - (aggregate_error_B_2 / aggregate_error_B_4)

        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        aggregate_z_total = (
            aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        )
        aggregate_yz_total = (
            aggregate_yz_unexpected + aggregate_yz_train + aggregate_indicator_test.T @ self.weighted_yz_test_pred
        )
        aggregate_perc_margin_total = aggregate_yz_total / aggregate_z_total

        interval_upper, interval_lower = (
            aggregate_perc_margin_total
            - np.quantile(aggregate_error_B, q=[lower_q, upper_q], axis=-1).T  # move into y space from residual space
        ).T
        interval_upper = interval_upper.reshape(-1, 1)
        interval_lower = interval_lower.reshape(-1, 1)

        return PredictionIntervals(interval_lower, interval_upper)  # removed round

    def get_all_conformalization_data_unit(self):
        return None, None

    def get_all_conformalization_data_agg(self):
        return None, None
