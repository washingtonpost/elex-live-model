import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import linprog
import math

from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals
from elexmodel.handlers.data.Featurizer import Featurizer

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
            weights = np.ones((y.shape[0], ))
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
        residuals = (y - y_hat).flatten()
        if loo:
            residuals /= (1 - self.hat_vals)
        if center:
            residuals -= np.mean(residuals)
        return residuals

class QuantileRegression(object):
    def __init__(self):
        self.beta_hats = []

    def _fit(self, S, Phi, zeros, N, weights, tau):
        bounds = weights.reshape(-1, 1) * np.asarray([(tau - 1, tau)] * N)
        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, 
            method='highs', options={'presolve': False})
        return -1 * res.eqlin.marginals
    
    def fit(self, x, y, taus=0.5, weights=None):
        if weights is None:
            weights = np.ones((y.shape[0], ))

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
        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.rng = np.random.default_rng(seed=self.seed)
        self.ran_bootstrap = False
    
    def get_minimum_reporting_units(self, alpha):
        return 10
        #return math.ceil(-1 * (alpha + 1) / (alpha - 1))

    def compute_bootstrap_errors(self, reporting_units, nonreporting_units):
        n = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]

        self.featurizer.compute_means_for_centering(reporting_units, nonreporting_units)
        x_train = self.featurizer.featurize_fitting_data(reporting_units, add_intercept=self.add_intercept).values
        y_train = reporting_units['normalized_margin'].values.reshape(-1, 1)
        z_train = reporting_units['turnout_factor'].values.reshape(-1, 1)
        weights_train = reporting_units['weights'].values.reshape(-1, 1)

        x_test = self.featurizer.featurize_heldout_data(nonreporting_units).values
        weights_test = nonreporting_units['weights'].values.reshape(-1, 1)

        x_all = np.concatenate([x_train, x_test], axis=0)
        ols_y = OLSRegression().fit(x_train, y_train, weights=weights_train)
        ols_z = OLSRegression().fit(x_train, z_train, weights=weights_train)

        y_train_pred = ols_y.predict(x_train)
        z_train_pred = ols_z.predict(x_train)

        residuals_y = ols_y.residuals(y_train, y_train_pred, loo=True, center=True)
        print(ols_y.beta_hat)
        residuals_z = ols_z.residuals(z_train, z_train_pred, loo=True, center=True)
        print(ols_z.beta_hat)

        taus_lower = np.arange(0.01, 0.5, 0.01)
        taus_upper = np.arange(0.50, 1, 0.01)
        taus = np.concatenate([taus_lower, taus_upper])

        # TODO: generalize this for the number of covariates
        x_strata_indices = [0] + self.featurizer.expanded_fixed_effects_cols
        x_strata = np.unique(x_all[:,x_strata_indices], axis=0).astype(int)
        x_train_strata = x_train[:,x_strata_indices]
        x_test_strata = x_test[:,x_strata_indices]

        stratum_ppfs_y = {}
        stratum_ppfs_z = {}

        stratum_cdfs_y = {}
        stratum_cdfs_z = {}

        def ppf_creator(betas, taus, lb, ub):
            return lambda p: np.interp(p, taus, betas, lb, ub)
        
        def cdf_creator(betas, taus):
            return lambda x: np.interp(x, betas, taus, right=1)
        
        for x_stratum in x_strata:
            x_train_aug = np.concatenate([x_train_strata, x_stratum.reshape(1, -1)], axis=0)
            y_aug_lb = np.concatenate([residuals_y, [self.y_LB]])
            y_aug_ub = np.concatenate([residuals_y, [self.y_UB]])
            z_aug_lb = np.concatenate([residuals_z, [self.z_LB]])
            z_aug_ub = np.concatenate([residuals_z, [self.z_UB]])
            betas_y_lower = QuantileRegression().fit(x_train_aug, y_aug_lb, taus_lower)
            betas_y_upper = QuantileRegression().fit(x_train_aug, y_aug_ub, taus_upper)
            betas_y = np.concatenate([betas_y_lower, betas_y_upper])
            betas_z_lower = QuantileRegression().fit(x_train_aug, z_aug_lb, taus_lower)
            betas_z_upper = QuantileRegression().fit(x_train_aug, z_aug_ub, taus_upper)
            betas_z = np.concatenate([betas_z_lower, betas_z_upper])

            betas_y_stratum = betas_y[:,np.where(x_stratum == 1)[0]].sum(axis=1)
            betas_z_stratum = betas_z[:,np.where(x_stratum == 1)[0]].sum(axis=1)

            stratum_ppfs_y[tuple(x_stratum)] = ppf_creator(betas_y_stratum, taus, self.y_LB, self.y_UB)
            stratum_ppfs_z[tuple(x_stratum)] = ppf_creator(betas_z_stratum, taus, self.z_LB, self.z_UB)

            stratum_cdfs_y[tuple(x_stratum)] = cdf_creator(betas_y_stratum, taus)
            stratum_cdfs_z[tuple(x_stratum)] = cdf_creator(betas_y_stratum, taus)

        unifs = []
        x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)
        for strata_dummies in x_train_strata_unique:
            residuals_y_strata = residuals_y[np.where((strata_dummies == x_train_strata).all(axis=1))[0]]
            residuals_z_strata = residuals_z[np.where((strata_dummies == x_train_strata).all(axis=1))[0]]

            unifs_y = stratum_cdfs_y[tuple(strata_dummies)](residuals_y_strata + 1e-6).reshape(-1, 1)
            unifs_z = stratum_cdfs_z[tuple(strata_dummies)](residuals_z_strata + 1e-6).reshape(-1, 1)

            unifs_y[np.isclose(unifs_y, 1)] = np.max(taus)
            unifs_y[np.isclose(unifs_y, 0)] = np.min(taus)

            unifs_z[np.isclose(unifs_z, 1)] = np.max(taus)
            unifs_z[np.isclose(unifs_z, 0)] = np.min(taus)            

            unifs_strata = np.concatenate([unifs_y, unifs_z], axis=1)
            # unifs_strata = unifs_y

            unifs.append(unifs_strata)
        unifs = np.concatenate(unifs, axis=0)

        unifs_B = self.rng.choice(unifs, (n, self.B), replace=True)
        # unifs_B = self.rng.choice(unifs.flatten(), (n, self.B), replace=True)

        residuals_y_B = np.zeros((n, self.B))
        residuals_z_B = np.zeros((n, self.B))
        
        for strata_dummies in x_train_strata_unique:
            strata_indices = np.where((strata_dummies == x_train_strata).all(axis=1))[0]
            unifs_strata = unifs_B[strata_indices]
            residuals_y_B[strata_indices] = stratum_ppfs_y[tuple(strata_dummies)](unifs_strata[:,:,0])
            residuals_z_B[strata_indices] = stratum_ppfs_z[tuple(strata_dummies)](unifs_strata[:,:,1])
            # residuals_y_B[strata_indices] = stratum_ppfs_y[tuple(strata_dummies)](unifs_strata)
        y_B = y_train_pred + residuals_y_B
        z_B = z_train_pred + residuals_z_B
        ols_y_B = OLSRegression().fit(x_train, y_B, weights_train, normal_eqs=ols_y.normal_eqs)
        ols_z_B = OLSRegression().fit(x_train, z_B, weights_train, normal_eqs=ols_z.normal_eqs)
        
        # y_test_pred_B = ols_y_B.predict(x_test)
        yz_test_pred_B = ols_y_B.predict(x_test) * ols_z_B.predict(x_test)
        
        y_test_pred = ols_y.predict(x_test)
        z_test_pred = ols_z.predict(x_test)
        yz_test_pred = y_test_pred * z_test_pred

        # sample uniforms for each outstanding state and outstanding stratum in state
        groups_test = nonreporting_units[['postal_code']].values.astype(str)
        unique_groups = np.unique(groups_test, axis=0)
        test_unifs_groups = self.rng.uniform(low=0, high=1, size=(len(unique_groups), self.B, 2))
        # test_unifs_groups = self.rng.uniform(low=0, high=1, size=(len(unique_groups), self.B))

        # for each row in groups_test, fetch the index of the matching row in unique_groups
        # matching_groups is the answer to ^this problem: matching_groups.shape = (groups_test.shape[0], ) 
        matching_groups = np.where((unique_groups == groups_test[:, None]).all(axis=-1))[1]

        # naive perfect correlation matching is below
        test_unifs = test_unifs_groups[matching_groups]  

        test_residuals_y = np.zeros((n_test, self.B))
        test_residuals_z = np.zeros((n_test, self.B))

        x_test_strata_unique = np.unique(x_test_strata, axis=0).astype(int)
        for strata_dummies in x_test_strata_unique:
            strata_indices = np.where((strata_dummies == x_test_strata).all(axis=1))[0]
            unifs_strata = test_unifs[strata_indices]
            test_residuals_y[strata_indices] = stratum_ppfs_y[tuple(strata_dummies)](unifs_strata[:,:,0])
            test_residuals_z[strata_indices] = stratum_ppfs_z[tuple(strata_dummies)](unifs_strata[:,:,1])
            # test_residuals_y[strata_indices] = stratum_ppfs_y[tuple(strata_dummies)](unifs_strata)

        self.errors_B_1 = yz_test_pred_B * weights_test
        # self.errors_B_1 = y_test_pred_B * weights_test

        errors_B_2 = test_residuals_z * y_test_pred
        errors_B_2 += test_residuals_y * z_test_pred
        errors_B_2 += test_residuals_y * test_residuals_z
        errors_B_2 += yz_test_pred
        # errors_B_2 = y_test_pred + test_residuals_y # did we previously forget this?

        self.errors_B_2 = errors_B_2 * weights_test

        self.weighted_yz_test_pred = yz_test_pred * weights_test
        # self.weighted_y_test_pred = y_test_pred * weights_test

        self.ran_bootstrap = True

    def get_unit_predictions(self,  reporting_units, nonreporting_units, estimand):
        if not self.ran_bootstrap:
            self.compute_bootstrap_errors(reporting_units, nonreporting_units)
        return self.weighted_yz_test_pred
        # return self.weighted_y_test_pred

    def get_unit_prediction_intervals(self, reporting_units, non_reporting_units, alpha, estimand):
        errors_B = self.errors_B_1 - self.errors_B_2

        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        interval_upper, interval_lower = (self.weighted_yz_test_pred - np.quantile(errors_B, q=[lower_q, upper_q], axis=-1).T).T
        # interval_upper, interval_lower = (self.weighted_y_test_pred - np.quantile(errors_B, q=[lower_q, upper_q], axis=-1).T).T

        interval_upper = interval_upper.reshape(-1,1)
        interval_lower = interval_lower.reshape(-1,1)

        return PredictionIntervals(interval_lower.round(decimals=0), interval_upper.round(decimals=0))
    
    def get_aggregate_prediction_intervals(
        self,
        reporting_units,
        nonreporting_units,
        unexpected_units,
        aggregate,
        alpha,
        conformalization_data,
        estimand
    ):
        n = reporting_units.shape[0]
        m = nonreporting_units.shape[0]

        aggregate_indicator = pd.get_dummies(pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)[aggregate]).values
        aggregate_indicator_expected = aggregate_indicator[:(n + m)]

        aggregate_indicator_unexpected = aggregate_indicator[(n + m):]
        margin_unexpected = unexpected_units['results_margin'].values.reshape(-1,1)
        aggregate_yz_unexpected = aggregate_indicator_unexpected.T @ margin_unexpected
        # aggregate_y_unexpected = aggregate_indicator_unexpected.T @ margin_unexpected

        aggregate_indicator_train = aggregate_indicator_expected[:n]
        aggregate_indicator_test = aggregate_indicator_expected[n:]
        weights_train = reporting_units['weights'].values.reshape(-1, 1)
        y_train = reporting_units['normalized_margin'].values.reshape(-1, 1)
        z_train = reporting_units['turnout_factor'].values.reshape(-1, 1)

        yz_train = y_train * z_train
        aggregate_yz_train = aggregate_indicator_train.T @ (weights_train * yz_train)
        # aggregate_y_train = aggregate_indicator_train.T @ (weights_train * y_train)

        aggregate_yz_test_B = aggregate_yz_train + aggregate_indicator_test.T @ self.errors_B_1
        aggregate_yz_test_pred = aggregate_yz_train + aggregate_indicator_test.T @ self.errors_B_2
        # aggregate_y_test_B = aggregate_y_train + aggregate_indicator_test.T @ self.errors_B_1
        # aggregate_y_test_pred = aggregate_y_train + aggregate_indicator_test.T @ self.errors_B_2

        aggregate_error_B_1 = aggregate_yz_test_B
        aggregate_error_B_2 = aggregate_yz_test_pred
        # aggregate_error_B_1 = aggregate_y_test_B
        # aggregate_error_B_2 = aggregate_y_test_pred

        aggregate_error_B = aggregate_error_B_1 - aggregate_error_B_2

        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        aggregate_yz_total = aggregate_yz_unexpected + aggregate_yz_train + aggregate_indicator_test.T @ self.weighted_yz_test_pred
        # aggregate_y_total = aggregate_y_unexpected + aggregate_y_train + aggregate_indicator_test.T @ self.weighted_y_test_pred
        
        interval_upper, interval_lower = (
            aggregate_yz_total - # move into y space from residual space
            # aggregate_y_total -
            np.quantile(
                aggregate_error_B, 
                q=[lower_q, upper_q],
                axis=-1
            ).T
        ).T
        interval_upper = interval_upper.reshape(-1,1)
        interval_lower = interval_lower.reshape(-1,1)

        return PredictionIntervals(interval_lower.round(decimals=0), interval_upper.round(decimals=0))

    def get_all_conformalization_data_unit(self):
        return None, None
    
    def get_all_conformalization_data_agg(self):
        return None, None
