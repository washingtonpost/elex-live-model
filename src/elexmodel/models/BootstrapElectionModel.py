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

    def _compute_normal_equations(self, x, L, lambda_):
        Q, R = np.linalg.qr(L @ x)
        lambda_I = lambda_ * np.eye(R.shape[1])
        lambda_I[0, 0] = 0
        lambda_I[1, 1] = 0
        return np.linalg.inv(R.T @ R + lambda_I) @ R.T @ Q.T

    def fit(self, x, y, weights=None, lambda_=0, normal_eqs=None):
        if weights is None:
            weights = np.ones((y.shape[0], ))
        # normalize + sqrt
        L = np.diag(np.sqrt(weights.flatten() / weights.sum()))
        if normal_eqs is not None:
            self.normal_eqs = normal_eqs
        else:
            self.normal_eqs = self._compute_normal_equations(x, L, lambda_)
        self.hat_vals = np.diag(x @ self.normal_eqs @ L)
        self.beta_hat = self.normal_eqs @ L @ y
        print(self.beta_hat)
        return self

    def predict(self, x):
        return x @ self.beta_hat
    
    def residuals(self, y, y_hat, loo=True, center=True):
        residuals = (y - y_hat)
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
        self.B = 40
        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.rng = np.random.default_rng(seed=self.seed)
        self.ran_bootstrap = False
    
    def cv_lambda(self, x, y, lambdas_, weights=None, k=5):
        if weights is None:
            weights = np.ones((y.shape[0], 1))
        x_y_w = np.concatenate([x, y, weights], axis=1)
        self.rng.shuffle(x_y_w)
        chunks = np.array_split(x_y_w, k, axis=0)
        ols = OLSRegression()
        errors = np.zeros((len(lambdas_), ))
        for i, lambda_ in enumerate(lambdas_):
            for test_chunk in range(k):
                x_y_w_test = chunks[test_chunk]
                x_y_w_train = np.concatenate(chunks[:test_chunk] + chunks[test_chunk + 1:], axis=0)
                x_test = x_y_w_test[:,:-2]
                y_test = x_y_w_test[:,-2]
                w_test = x_y_w_test[:,-1]
                x_train = x_y_w_train[:,:-2]
                y_train = x_y_w_train[:,-2]
                w_train = x_y_w_train[:,-1]
                ols_lambda = ols.fit(x_train, y_train, weights=w_train, lambda_=lambda_)
                y_hat_lambda = ols_lambda.predict(x_test)
                errors[i] += np.sum(w_test * ols_lambda.residuals(y_test, y_hat_lambda, loo=False, center=False) ** 2) / np.sum(w_test)
        return lambdas_[np.argmin(errors)]

    def get_minimum_reporting_units(self, alpha):
        return 10
        #return math.ceil(-1 * (alpha + 1) / (alpha - 1))

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
        self.featurizer.compute_stds_for_scaling(reporting_units, nonreporting_units)
        
        x_train = self.featurizer.featurize_fitting_data(reporting_units, add_intercept=self.add_intercept, center_features=False, scale_features=False).values
        y_train = reporting_units['normalized_margin'].values.reshape(-1, 1)
        z_train = reporting_units['turnout_factor'].values.reshape(-1, 1)
        weights_train = reporting_units['weights'].values.reshape(-1, 1)
       # weights_train = weights_train / np.sum(weights_train)

        x_test = self.featurizer.featurize_heldout_data(nonreporting_units).values
        weights_test = nonreporting_units['weights'].values.reshape(-1, 1)

        x_all = np.concatenate([x_train, x_test], axis=0)

        optimal_lambda_y = 0 # self.cv_lambda(x_train, y_train, np.logspace(-3, 2, 20), weights=weights_train)
        optimal_lambda_z = 0 # self.cv_lambda(x_train, z_train, np.logspace(-3, 2, 20), weights=weights_train)
        print(optimal_lambda_y)
        ols_y = OLSRegression().fit(x_train, y_train, weights=weights_train, lambda_=optimal_lambda_y)
        ols_z = OLSRegression().fit(x_train, z_train, weights=weights_train, lambda_=optimal_lambda_z)

        y_train_pred = ols_y.predict(x_train)
        z_train_pred = ols_z.predict(x_train)

        residuals_y = ols_y.residuals(y_train, y_train_pred, loo=True, center=True)
        residuals_z = ols_z.residuals(z_train, z_train_pred, loo=True, center=True)

        aggregate_indicator = pd.get_dummies(pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)['postal_code']).values
        aggregate_indicator_expected = aggregate_indicator[:(n_train + n_test)]
        aggregate_indicator_unexpected = aggregate_indicator[(n_train + n_test):]
        
        aggregate_indicator_train = aggregate_indicator_expected[:n_train]
        aggregate_indicator_test = aggregate_indicator_expected[n_train:]

        epsilon_y_hat = (aggregate_indicator_train.T @ residuals_y) / aggregate_indicator_train.sum(axis=0).reshape(-1, 1)
        epsilon_y_hat[aggregate_indicator_train.sum(axis=0) < 2] = 0 # can't estimate state random effecct if we only have 1 unit
        epsilon_z_hat = (aggregate_indicator_train.T @ residuals_z) / aggregate_indicator_train.sum(axis=0).reshape(-1, 1)
        epsilon_z_hat[aggregate_indicator_train.sum(axis=0) < 2] = 0
        epsilon_hat = np.concatenate([epsilon_y_hat, epsilon_z_hat], axis=1)

        delta_y_hat = (residuals_y - (aggregate_indicator_train @ epsilon_y_hat)).flatten()
        delta_z_hat = (residuals_z - (aggregate_indicator_train @ epsilon_z_hat)).flatten()

        taus_lower = np.arange(0.01, 0.5, 0.01)
        taus_upper = np.arange(0.50, 1, 0.01)
        taus = np.concatenate([taus_lower, taus_upper])

        x_strata_indices = [0] + self.featurizer.expanded_fixed_effects_cols
        x_strata = np.unique(x_all[:,x_strata_indices], axis=0).astype(int)
        x_train_strata = x_train[:,x_strata_indices]
        x_test_strata = x_test[:,x_strata_indices]

        stratum_ppfs_delta_y = {}
        stratum_ppfs_delta_z = {}

        stratum_cdfs_delta_y = {}
        stratum_cdfs_delta_z = {}

        def ppf_creator(betas, taus, lb, ub):
            return lambda p: np.interp(p, taus, betas, lb, ub)
        
        def cdf_creator(betas, taus):
            return lambda x: np.interp(x, betas, taus, right=1)
        
        for x_stratum in x_strata:
            x_train_aug = np.concatenate([x_train_strata, x_stratum.reshape(1, -1)], axis=0)
            delta_y_aug_lb = np.concatenate([delta_y_hat, [self.y_LB]])
            delta_y_aug_ub = np.concatenate([delta_y_hat, [self.y_UB]])
            delta_z_aug_lb = np.concatenate([delta_z_hat, [self.z_LB]])
            delta_z_aug_ub = np.concatenate([delta_z_hat, [self.z_UB]])
            betas_y_lower = QuantileRegression().fit(x_train_aug, delta_y_aug_lb, taus_lower)
            betas_y_upper = QuantileRegression().fit(x_train_aug, delta_y_aug_ub, taus_upper)
            betas_y = np.concatenate([betas_y_lower, betas_y_upper])
            betas_z_lower = QuantileRegression().fit(x_train_aug, delta_z_aug_lb, taus_lower)
            betas_z_upper = QuantileRegression().fit(x_train_aug, delta_z_aug_ub, taus_upper)
            betas_z = np.concatenate([betas_z_lower, betas_z_upper])

            betas_y_stratum = betas_y[:,np.where(x_stratum == 1)[0]].sum(axis=1)
            betas_z_stratum = betas_z[:,np.where(x_stratum == 1)[0]].sum(axis=1)

            stratum_ppfs_delta_y[tuple(x_stratum)] = ppf_creator(betas_y_stratum, taus, self.y_LB, self.y_UB)
            stratum_ppfs_delta_z[tuple(x_stratum)] = ppf_creator(betas_z_stratum, taus, self.z_LB, self.z_UB)

            stratum_cdfs_delta_y[tuple(x_stratum)] = cdf_creator(betas_y_stratum, taus)
            stratum_cdfs_delta_z[tuple(x_stratum)] = cdf_creator(betas_y_stratum, taus)

        unifs = []
        x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)
        for strata_dummies in x_train_strata_unique:
            delta_y_strata = delta_y_hat[np.where((strata_dummies == x_train_strata).all(axis=1))[0]]
            delta_z_strata = delta_z_hat[np.where((strata_dummies == x_train_strata).all(axis=1))[0]]

            unifs_y = stratum_cdfs_delta_y[tuple(strata_dummies)](delta_y_strata + 1e-6).reshape(-1, 1)
            unifs_z = stratum_cdfs_delta_z[tuple(strata_dummies)](delta_z_strata + 1e-6).reshape(-1, 1)

            unifs_y[np.isclose(unifs_y, 1)] = np.max(taus)
            unifs_y[np.isclose(unifs_y, 0)] = np.min(taus)

            unifs_z[np.isclose(unifs_z, 1)] = np.max(taus)
            unifs_z[np.isclose(unifs_z, 0)] = np.min(taus)            

            unifs_strata = np.concatenate([unifs_y, unifs_z], axis=1)

            unifs.append(unifs_strata)
        unifs = np.concatenate(unifs, axis=0)

        iqr_y_strata = {}
        iqr_z_strata = {}
        for x_stratum in x_strata:
            x_stratum_delta_y_ppf = stratum_ppfs_delta_y[tuple(x_stratum)]
            iqr_y = x_stratum_delta_y_ppf(.75) - x_stratum_delta_y_ppf(.25)
            iqr_y_strata[tuple(x_stratum)] = iqr_y

            x_stratum_delta_z_ppf = stratum_ppfs_delta_z[tuple(x_stratum)]
            iqr_z = x_stratum_delta_z_ppf(.75) - x_stratum_delta_z_ppf(.25)
            iqr_z_strata[tuple(x_stratum)] = iqr_z

        var_epsilon_y = np.zeros((aggregate_indicator_train.shape[1], ))
        var_epsilon_z = np.zeros((aggregate_indicator_train.shape[1], ))

        for strata_dummies in x_train_strata_unique:
            strata_indices = np.where((strata_dummies == x_train_strata).all(axis=1))[0]
            var_epsilon_y += (aggregate_indicator_train[strata_indices] * (iqr_y_strata[tuple(strata_dummies)] ** 2)).sum(axis=0)
            var_epsilon_z += (aggregate_indicator_train[strata_indices] * (iqr_z_strata[tuple(strata_dummies)] ** 2)).sum(axis=0)
        var_epsilon_y /= (1.349 ** 2)
        var_epsilon_z /= (1.349 ** 2)
        var_epsilon_y /= aggregate_indicator_train.sum(axis=0)
        var_epsilon_z /= aggregate_indicator_train.sum(axis=0)
        var_epsilon_y[aggregate_indicator_train.sum(axis=0) < 2] = 0
        var_epsilon_z[aggregate_indicator_train.sum(axis=0) < 2] = 0

        epsilon_y_B = self.rng.multivariate_normal(mean=epsilon_hat[:,0], cov=np.diag(var_epsilon_y), size=self.B).T
        epsilon_z_B = self.rng.multivariate_normal(mean=epsilon_hat[:,1], cov=np.diag(var_epsilon_z), size=self.B).T

        unifs_B = self.rng.choice(unifs, (n_train, self.B), replace=True)

        delta_y_B = np.zeros((n_train, self.B))
        delta_z_B = np.zeros((n_train, self.B))

        for strata_dummies in x_train_strata_unique:
            strata_indices = np.where((strata_dummies == x_train_strata).all(axis=1))[0]
            unifs_strata = unifs_B[strata_indices]
            delta_y_B[strata_indices] = stratum_ppfs_delta_y[tuple(strata_dummies)](unifs_strata[:,:,0])
            delta_z_B[strata_indices] = stratum_ppfs_delta_z[tuple(strata_dummies)](unifs_strata[:,:,1])

        y_train_B = y_train_pred + (aggregate_indicator_train @ epsilon_y_B) + delta_y_B
        z_train_B = z_train_pred + (aggregate_indicator_train @ epsilon_z_B) + delta_z_B
        
        ols_y_B = OLSRegression().fit(x_train, y_train_B, weights_train, normal_eqs=ols_y.normal_eqs)
        ols_z_B = OLSRegression().fit(x_train, z_train_B, weights_train, normal_eqs=ols_z.normal_eqs)

        y_train_pred_B = ols_y_B.predict(x_train)
        z_train_pred_B = ols_z_B.predict(x_train)

        residuals_y_B = ols_y_B.residuals(y_train_B, y_train_pred_B, loo=True, center=True)
        residuals_z_B = ols_z_B.residuals(z_train_B, z_train_pred_B, loo=True, center=True)

        epsilon_y_hat_B = (aggregate_indicator_train.T @ residuals_y_B) / aggregate_indicator_train.sum(axis=0).reshape(-1, 1)
        epsilon_y_hat_B[aggregate_indicator_train.sum(axis=0) < 2] = 0 
        epsilon_z_hat_B = (aggregate_indicator_train.T @ residuals_z_B) / aggregate_indicator_train.sum(axis=0).reshape(-1, 1)
        epsilon_z_hat_B[aggregate_indicator_train.sum(axis=0) < 2] = 0 

        y_test_pred_B = ols_y_B.predict(x_test) + (aggregate_indicator_test @ epsilon_y_hat_B)
        z_test_pred_B = ols_z_B.predict(x_test) + (aggregate_indicator_test @ epsilon_y_hat_B)
        
        yz_test_pred_B = y_test_pred_B * z_test_pred_B
        
        y_test_pred = ols_y.predict(x_test) + (aggregate_indicator_test @ epsilon_y_hat)
        z_test_pred = ols_z.predict(x_test) + (aggregate_indicator_test @ epsilon_z_hat)
        yz_test_pred = y_test_pred * z_test_pred

        test_unifs = self.rng.uniform(low=0, high=1, size=(n_test, self.B, 2))

        # # sample uniforms for each outstanding state and outstanding stratum in state
        # groups_test = nonreporting_units[['postal_code']].values.astype(str)
        # unique_groups = np.unique(groups_test, axis=0)
        # test_unifs_groups = self.rng.uniform(low=0, high=1, size=(len(unique_groups), self.B, 2))
        # # test_unifs_groups = self.rng.uniform(low=0, high=1, size=(len(unique_groups), self.B))

        # # for each row in groups_test, fetch the index of the matching row in unique_groups
        # # matching_groups is the answer to ^this problem: matching_groups.shape = (groups_test.shape[0], ) 
        # matching_groups = np.where((unique_groups == groups_test[:, None]).all(axis=-1))[1]

        # # naive perfect correlation matching is below
        # test_unifs = test_unifs_groups[matching_groups]  

        test_residuals_y = np.zeros((n_test, self.B))
        test_residuals_z = np.zeros((n_test, self.B))

        x_test_strata_unique = np.unique(x_test_strata, axis=0).astype(int)
        for strata_dummies in x_test_strata_unique:
            strata_indices = np.where((strata_dummies == x_test_strata).all(axis=1))[0]
            unifs_strata = test_unifs[strata_indices]
            test_residuals_y[strata_indices] = stratum_ppfs_delta_y[tuple(strata_dummies)](unifs_strata[:,:,0])
            test_residuals_z[strata_indices] = stratum_ppfs_delta_z[tuple(strata_dummies)](unifs_strata[:,:,1])
        
        # gives us indices of states for which we have no samples in the training set
        states_not_in_reporting_units = np.where(np.all(aggregate_indicator_train == 0, axis=0))[0]
        # gives us states for which there is at least one county not reporting
        states_in_nonreporting_units = np.where(np.any(aggregate_indicator_test > 0, axis=0))[0]
        states_that_need_random_effect = np.intersect1d(states_not_in_reporting_units, states_in_nonreporting_units)
        
        # TODO: replace gaussian model with QR (??)
        sigma_hat = np.zeros((2, 2))
        sigma_hat_denominator = 0
        sample_var_epsilon_y = np.var(aggregate_indicator_train * residuals_y, axis=0)
        sample_var_epsilon_z = np.var(aggregate_indicator_train * residuals_z, axis=0)

        for (epsilon_i, sample_var_epsilon_y_i, sample_var_epsilon_z_i) in zip(epsilon_hat, sample_var_epsilon_y, sample_var_epsilon_z):
            if np.isclose(epsilon_i.sum(), 0): continue
            sigma_hat += np.outer(epsilon_i, epsilon_i) - np.diag([sample_var_epsilon_y_i, sample_var_epsilon_z_i])
            sigma_hat_denominator += 1
        sigma_hat /= sigma_hat_denominator
        sigma_hat[0, 1] = 0 # setting covariances to zero for now
        sigma_hat[1, 0] = 0

        test_epsilon = self.rng.multivariate_normal([0, 0], sigma_hat, size=(len(states_that_need_random_effect), self.B))
        test_epsilon_y = test_epsilon[:,:,0]
        test_epsilon_z = test_epsilon[:,:,1]
 
        test_residuals_y += aggregate_indicator_test[:,states_that_need_random_effect] @ test_epsilon_y
        test_residuals_z += aggregate_indicator_test[:,states_that_need_random_effect] @ test_epsilon_z
        
        self.errors_B_1 = yz_test_pred_B * weights_test

        errors_B_2 = (y_test_pred + test_residuals_y).clip(min=-1, max=1)
        errors_B_2 *= (z_test_pred + test_residuals_z).clip(min=0.5, max=1.5) # clipping: predicting turnout can't be less than 50% or more than 150% of previous election

        self.errors_B_2 = errors_B_2 * weights_test 

        self.errors_B_3 = z_test_pred_B.clip(min=0.5, max=1.5) * weights_test # denominator for percentage margin
        self.errors_B_4 = (z_test_pred + test_residuals_z).clip(min=0.5, max=1.5) * weights_test # clipping: predicting turnout can't be less than 50% or more than 150% of previous election

        self.weighted_yz_test_pred = yz_test_pred * weights_test
        self.weighted_z_test_pred = z_test_pred * weights_test

        self.ran_bootstrap = True

        # if n_train > 150:
            # import pdb; pdb.set_trace()

    def get_unit_predictions(self,  reporting_units, nonreporting_units, estimand, **kwargs):
        if not self.ran_bootstrap:
            unexpected_units = kwargs['unexpected_units']
            self.compute_bootstrap_errors(reporting_units, nonreporting_units, unexpected_units)
        return self.weighted_yz_test_pred


    def get_aggregate_predictions(self, reporting_units, nonreporting_units, unexpected_units, aggregate, estimand):
        n = reporting_units.shape[0]
        m = nonreporting_units.shape[0]

        aggregate_indicator = pd.get_dummies(pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)[aggregate]).values
        aggregate_indicator_expected = aggregate_indicator[:(n + m)]

        aggregate_indicator_unexpected = aggregate_indicator[(n + m):]
        turnout_unexpected = (unexpected_units['results_dem'] + unexpected_units['results_gop']).values.reshape(-1, 1)
        
        aggregate_indicator_train = aggregate_indicator_expected[:n]
        aggregate_indicator_test = aggregate_indicator_expected[n:]
        weights_train = reporting_units['weights'].values.reshape(-1, 1)
        z_train = reporting_units['turnout_factor'].values.reshape(-1, 1)

        aggregate_z_train = aggregate_indicator_train.T @ (weights_train * z_train)
        
        aggregate_z_unexpected = aggregate_indicator_unexpected.T @ turnout_unexpected
        aggregate_z_total = aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred

        raw_margin_df = super().get_aggregate_predictions(reporting_units, nonreporting_units, unexpected_units, aggregate, estimand)
        raw_margin_df['pred_margin'] /= (aggregate_z_total.flatten() + 1)
        raw_margin_df['results_margin'] /= (aggregate_z_total.flatten() + 1) # avoid NaN
        return raw_margin_df

    def get_unit_prediction_intervals(self, reporting_units, non_reporting_units, alpha, estimand):
        errors_B = self.errors_B_1 - self.errors_B_2

        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        interval_upper, interval_lower = (self.weighted_yz_test_pred - np.quantile(errors_B, q=[lower_q, upper_q], axis=-1).T).T

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
        turnout_unexpected = (unexpected_units['results_dem'] + unexpected_units['results_gop']).values.reshape(-1, 1)
        aggregate_z_unexpected = aggregate_indicator_unexpected.T @ turnout_unexpected
        aggregate_yz_unexpected = aggregate_indicator_unexpected.T @ margin_unexpected

        aggregate_indicator_train = aggregate_indicator_expected[:n]
        aggregate_indicator_test = aggregate_indicator_expected[n:]
        weights_train = reporting_units['weights'].values.reshape(-1, 1)
        y_train = reporting_units['normalized_margin'].values.reshape(-1, 1)
        z_train = reporting_units['turnout_factor'].values.reshape(-1, 1)

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

        aggregate_z_total = aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        aggregate_yz_total = aggregate_yz_unexpected + aggregate_yz_train + aggregate_indicator_test.T @ self.weighted_yz_test_pred
        aggregate_perc_margin_total = aggregate_yz_total / aggregate_z_total
        
        interval_upper, interval_lower = (
            aggregate_perc_margin_total - # move into y space from residual space
            np.quantile(
                aggregate_error_B, 
                q=[lower_q, upper_q],
                axis=-1
            ).T
        ).T
        interval_upper = interval_upper.reshape(-1,1)
        interval_lower = interval_lower.reshape(-1,1)

        return PredictionIntervals(interval_lower, interval_upper) # removed round

    def get_all_conformalization_data_unit(self):
        return None, None
    
    def get_all_conformalization_data_agg(self):
        return None, None
