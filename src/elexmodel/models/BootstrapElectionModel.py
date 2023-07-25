import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import linprog
from scipy.special import expit
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

    def _fit(self, S, Phi, zeros, N, weights, tau, constraints):
        if constraints is None:
            bounds = weights.reshape(-1, 1) * np.asarray([(tau - 1, tau)] * N)
            res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, 
                method='highs', options={'presolve': False})
        else:
            S_aug = np.concatenate([S, [constraints[0] - S[-1]], [S[-1] - constraints[1]]])
            Phi_aug = np.concatenate([Phi, np.zeros((2, Phi.shape[1]))], axis=0)
            zeros_aug = np.zeros((Phi_aug.shape[1], ))
            weights_aug = np.concatenate([weights, [1], [1]])
            bounds_aug = weights_aug.reshape(-1, 1) * np.asarray([(tau - 1, tau)] * (N + 2))
            bounds_aug[-3] = np.asarray([None, None])
            bounds_aug[-2] = np.asarray([0, None])
            bounds_aug[-1] = np.asarray([0, None])
            b_ub = np.asarray([tau, 1 - tau])
            A_ub = np.zeros((2, weights_aug.shape[0]))
            A_ub[0,-2] = weights_aug[-3]
            A_ub[0,-1] = -1 * weights_aug[-3]
            A_ub[1,-2] = -1 * weights_aug[-3]
            A_ub[1,-1] = -1 * weights_aug[-3]
            res = linprog(-1 * S_aug, A_eq=Phi_aug.T, b_eq=zeros_aug, bounds=bounds_aug, A_ub=A_ub, b_ub=b_ub,
                method='highs', options={'presolve': False})
        return -1 * res.eqlin.marginals
            
    def fit(self, x, y, taus=0.5, weights=None, constraints=None):
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
            self.beta_hats.append(self._fit(S, Phi, zeros, N, weights, tau, constraints))

        return self.beta_hats
    
class BootstrapElectionModel(BaseElectionModel):
    y_LB = -0.3
    y_UB = 0.3
    z_LB = -0.5
    z_UB = 0.5

    def __init__(self, model_settings={}):
        super().__init__(model_settings)
        self.seed = model_settings.get("seed", 0)
        self.B = model_settings.get("B", 2000)
        self.strata = model_settings.get("strata", ['county_classification']) # change this
        self.T = model_settings.get("T", 5000)
        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.rng = np.random.default_rng(seed=self.seed)
        self.ran_bootstrap = False
        self.taus_lower = np.arange(0.01, 0.5, 0.01)
        self.taus_upper = np.arange(0.50, 1, 0.01)
        self.taus = np.concatenate([self.taus_lower, self.taus_upper])
    
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

    def _estimate_epsilon(self, residuals, aggregate_indicator, shrinkage=False):
        epsilon_hat = (aggregate_indicator.T @ residuals) / aggregate_indicator.sum(axis=0).reshape(-1, 1)
        epsilon_hat[aggregate_indicator.sum(axis=0) < 2] = 0 # can't estimate state random effect if we only have 1 unit

        if shrinkage:
            # shrinkage code
            epsilon_hat_centered = (aggregate_indicator * (aggregate_indicator * residuals - epsilon_hat.T))**2
            epsilon_hat_var = epsilon_hat_centered.sum(axis=0) / aggregate_indicator.sum(axis=0)
            epsilon_hat_var = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, epsilon_hat_centered)
            epsilon_hat_var = np.nan_to_num(epsilon_hat_var) # get rid of divide by zero nan
            epsilon_hat_var = epsilon_hat_var.reshape(-1,1)
            shrinkage_factor = (1 - (epsilon_hat.shape[0] - 2) * epsilon_hat_var / np.linalg.norm(epsilon_hat)**2).clip(min=0)
            return shrinkage_factor * epsilon_hat
        return epsilon_hat

    def _estimate_delta(self, residuals, epsilon_hat, aggregate_indicator):
        return (residuals - (aggregate_indicator @ epsilon_hat)).flatten()

    # estimates epsilon and delta
    def _estimate_model_errors(self, model, x, y, aggregate_indicator):
        y_pred = model.predict(x)
        residuals_y = model.residuals(y, y_pred, loo=True, center=True)
        epsilon_y_hat = self._estimate_epsilon(residuals_y, aggregate_indicator, shrinkage=True)
        delta_y_hat = self._estimate_delta(residuals_y, epsilon_y_hat, aggregate_indicator)
        return residuals_y, epsilon_y_hat, delta_y_hat       

    def _estimate_strata_dist(self, x_train, x_train_strata, x_test, x_test_strata, delta_hat, lb, ub):
        stratum_ppfs_delta = {}
        stratum_cdfs_delta = {}

        def ppf_creator(betas, taus, lb, ub):
            return lambda p: np.interp(p, taus, betas, lb, ub)
        
        def cdf_creator(betas, taus):
            return lambda x: np.interp(x, betas, taus, right=1)
        
        x_strata = np.unique(np.concatenate([x_train_strata, x_test_strata], axis=0), axis=0).astype(int)
        for x_stratum in x_strata:
            x_train_aug = np.concatenate([x_train_strata, x_stratum.reshape(1, -1)], axis=0)
            delta_aug_lb = np.concatenate([delta_hat, [lb]])
            delta_aug_ub = np.concatenate([delta_hat, [ub]])
            betas_lower = QuantileRegression().fit(x_train_aug, delta_aug_lb, self.taus_lower)
            betas_upper = QuantileRegression().fit(x_train_aug, delta_aug_ub, self.taus_upper)

            betas = np.concatenate([betas_lower, betas_upper])

            betas_stratum = betas[:,np.where(x_stratum == 1)[0]].sum(axis=1)

            stratum_ppfs_delta[tuple(x_stratum)] = ppf_creator(betas_stratum, self.taus, lb, ub)

            stratum_cdfs_delta[tuple(x_stratum)] = cdf_creator(betas_stratum, self.taus)

        return stratum_ppfs_delta, stratum_cdfs_delta

    def _generate_nonreporting_bounds(self, nonreporting_units, bootstrap_estimand, n_bins=10):
        # TODO: figure out how to better estimate margin_upper/lower_bound
        # TODO: pass in the magic numbers
        nonreporting_expected_vote_frac = nonreporting_units.percent_expected_vote.values.clip(max=100) / 100
        if bootstrap_estimand == 'normalized_margin':
            unobserved_upper_bound = 1
            unobserved_lower_bound = -1
            upper_bound = nonreporting_expected_vote_frac * nonreporting_units[bootstrap_estimand] + (1 - nonreporting_expected_vote_frac) * unobserved_upper_bound
            lower_bound = nonreporting_expected_vote_frac * nonreporting_units[bootstrap_estimand] + (1 - nonreporting_expected_vote_frac) * unobserved_lower_bound
        elif bootstrap_estimand == 'turnout_factor':
            percent_expected_vote_error_bound = 0.25
            unobserved_upper_bound = 1.5
            unobserved_lower_bound = 0.5
            lower_bound = nonreporting_units[bootstrap_estimand] / (nonreporting_expected_vote_frac + percent_expected_vote_error_bound)
            upper_bound = nonreporting_units[bootstrap_estimand] / (nonreporting_expected_vote_frac - percent_expected_vote_error_bound).clip(min=0.01)
            upper_bound[np.isclose(upper_bound, 0)] = unobserved_upper_bound # turnout is at m

        # if percent reporting is 0 or 1, don't try to compute anything and revert to naive bounds
        lower_bound[np.isclose(nonreporting_expected_vote_frac, 0) | np.isclose(nonreporting_expected_vote_frac, 1)] = unobserved_lower_bound
        upper_bound[np.isclose(nonreporting_expected_vote_frac, 0) | np.isclose(nonreporting_expected_vote_frac, 1)] = unobserved_upper_bound

        return lower_bound.values.reshape(-1, 1), upper_bound.values.reshape(-1, 1)
        # quantiles = np.linspace(0, 1, num=n_bins+1) # linspace returns the start, but we want to upper bound
        
        # upper_quantiles = np.quantile(upper_bound, q=quantiles[1:])
        # upper_quantiles[-1] += 1e-6 # lol wtf
        # upper_bins = upper_quantiles[np.digitize(upper_bound, bins=upper_quantiles)]
        # lower_quantiles = np.quantile(lower_bound, q=quantiles)
        # lower_bins = lower_quantiles[np.digitize(lower_bound, bins=lower_quantiles[1:])]

        # nonreporting_units[f'{bootstrap_estimand}_upper'] = upper_bins
        # nonreporting_units[f'{bootstrap_estimand}_lower'] = lower_bins
        # return np.unique(nonreporting_units[[f'{bootstrap_estimand}_lower', f'{bootstrap_estimand}_upper']].values, axis=0)

    # probability integral transform for each stratum (lol)
    def _strata_pit(self, x_train_strata, x_train_strata_unique, delta_hat, stratum_cdfs_delta):
        unifs = []
        for strata_dummies in x_train_strata_unique:
            delta_strata = delta_hat[np.where((strata_dummies == x_train_strata).all(axis=1))[0]]

            unifs_strata = stratum_cdfs_delta[tuple(strata_dummies)](delta_strata + 1e-6).reshape(-1, 1)

            unifs_strata[np.isclose(unifs_strata, 1)] = np.max(self.taus)
            unifs_strata[np.isclose(unifs_strata, 0)] = np.min(self.taus)      

            unifs.append(unifs_strata)
        return np.concatenate(unifs).reshape(-1, 1)

    def _bootstrap_deltas(self, unifs, x_train_strata, x_train_strata_unique, stratum_ppfs_delta_y, stratum_ppfs_delta_z):
        n_train = unifs.shape[0]

        unifs_B = self.rng.choice(unifs, (n_train, self.B), replace=True)

        delta_y_B = np.zeros((n_train, self.B))
        delta_z_B = np.zeros((n_train, self.B))

        for strata_dummies in x_train_strata_unique:
            strata_indices = np.where((strata_dummies == x_train_strata).all(axis=1))[0]
            unifs_strata = unifs_B[strata_indices]
            delta_y_B[strata_indices] = stratum_ppfs_delta_y[tuple(strata_dummies)](unifs_strata[:,:,0])
            delta_z_B[strata_indices] = stratum_ppfs_delta_z[tuple(strata_dummies)](unifs_strata[:,:,1])
        return delta_y_B, delta_z_B
    
    def _bootstrap_epsilons(self, epsilon_y_hat, epsilon_z_hat, x_train_strata, x_train_strata_unique, stratum_ppfs_delta_y, stratum_ppfs_delta_z, aggregate_indicator_train):
        iqr_y_strata = {}
        iqr_z_strata = {}
        for x_stratum in x_train_strata_unique:
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
        iqr_scale = 1.349
        var_epsilon_y /= (iqr_scale ** 2)
        var_epsilon_z /= (iqr_scale ** 2)
        var_epsilon_y /= aggregate_indicator_train.sum(axis=0)
        var_epsilon_z /= aggregate_indicator_train.sum(axis=0)
        var_epsilon_y[aggregate_indicator_train.sum(axis=0) < 2] = 0
        var_epsilon_z[aggregate_indicator_train.sum(axis=0) < 2] = 0

        epsilon_y_B = self.rng.multivariate_normal(mean=epsilon_y_hat.flatten(), cov=np.diag(var_epsilon_y), size=self.B).T
        epsilon_z_B = self.rng.multivariate_normal(mean=epsilon_z_hat.flatten(), cov=np.diag(var_epsilon_z), size=self.B).T
        return epsilon_y_B, epsilon_z_B
    
    def _bootstrap_errors(self, epsilon_y_hat, epsilon_z_hat, delta_y_hat, delta_z_hat, x_train_strata, stratum_cdfs_y, stratum_cdfs_z, stratum_ppfs_delta_y, stratum_ppfs_delta_z, aggregate_indicator_train):
        x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)
        
        epsilon_y_B, epsilon_z_B = self._bootstrap_epsilons(epsilon_y_hat, epsilon_z_hat, x_train_strata, x_train_strata_unique, stratum_ppfs_delta_y, stratum_ppfs_delta_z, aggregate_indicator_train)

        unifs_y = self._strata_pit(x_train_strata, x_train_strata_unique, delta_y_hat, stratum_cdfs_y)
        unifs_z = self._strata_pit(x_train_strata, x_train_strata_unique, delta_z_hat, stratum_cdfs_z)
        unifs = np.concatenate([unifs_y, unifs_z], axis=1)

        delta_y_B, delta_z_B = self._bootstrap_deltas(unifs, x_train_strata, x_train_strata_unique, stratum_ppfs_delta_y, stratum_ppfs_delta_z)

        return (epsilon_y_B, epsilon_z_B), (delta_y_B, delta_z_B)

    def _sample_test_delta(self, x_test_strata, stratum_ppfs_delta_y, stratum_ppfs_delta_z):
        n_test = x_test_strata.shape[0]
        test_unifs = self.rng.uniform(low=0, high=1, size=(n_test, self.B, 2))

        test_delta_y = np.zeros((n_test, self.B))
        test_delta_z = np.zeros((n_test, self.B))

        x_test_strata_unique = np.unique(x_test_strata, axis=0).astype(int)
        for strata_dummies in x_test_strata_unique:
            strata_indices = np.where((strata_dummies == x_test_strata).all(axis=1))[0]
            unifs_strata = test_unifs[strata_indices]
            test_delta_y[strata_indices] = stratum_ppfs_delta_y[tuple(strata_dummies)](unifs_strata[:,:,0])
            test_delta_z[strata_indices] = stratum_ppfs_delta_z[tuple(strata_dummies)](unifs_strata[:,:,1])
        
        return test_delta_y, test_delta_z
    
    def _sample_test_epsilon(self, residuals_y, residuals_z, epsilon_y_hat, epsilon_z_hat, aggregate_indicator_train, aggregate_indicator_test):
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

        for (epsilon_y_i, epsilon_z_i, sample_var_epsilon_y_i, sample_var_epsilon_z_i) in zip(epsilon_y_hat, epsilon_z_hat, sample_var_epsilon_y, sample_var_epsilon_z):
            epsilon_i = np.asarray([epsilon_y_i, epsilon_z_i])
            if np.isclose(epsilon_i.sum(), 0): continue
            sigma_hat += np.outer(epsilon_i, epsilon_i) - np.diag([sample_var_epsilon_y_i, sample_var_epsilon_z_i])
            sigma_hat_denominator += 1
        sigma_hat /= sigma_hat_denominator
        sigma_hat[0, 1] = 0 # setting covariances to zero for now
        sigma_hat[1, 0] = 0

        sigma_hat = 1e-5 * np.eye(2)

        test_epsilon = self.rng.multivariate_normal([0, 0], sigma_hat, size=(len(states_that_need_random_effect), self.B))
        
        test_epsilon_y = aggregate_indicator_test[:,states_that_need_random_effect] @ test_epsilon[:,:,0]
        test_epsilon_z = aggregate_indicator_test[:,states_that_need_random_effect] @ test_epsilon[:,:,1]

        return test_epsilon_y, test_epsilon_z

    def _sample_test_errors(self, residuals_y, residuals_z, epsilon_y_hat, epsilon_z_hat, x_test_strata, stratum_ppfs_delta_y, stratum_ppfs_delta_z, aggregate_indicator_train, aggregate_indicator_test):
        test_epsilon_y, test_epsilon_z = self._sample_test_epsilon(residuals_y, residuals_z, epsilon_y_hat, epsilon_z_hat, aggregate_indicator_train, aggregate_indicator_test)
        test_delta_y, test_delta_z = self._sample_test_delta(x_test_strata, stratum_ppfs_delta_y, stratum_ppfs_delta_z)
        test_error_y = test_epsilon_y + test_delta_y
        test_error_z = test_epsilon_z + test_delta_z
        return test_error_y, test_error_z

    def _get_strata(self, reporting_units, nonreporting_units):
        # TODO: potentially generalize binning features for strata
        n_train = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]
        strata_featurizer = Featurizer([], self.strata)
        all_units = pd.concat([reporting_units, nonreporting_units], axis=0)
        strata_all = strata_featurizer.prepare_data(all_units, center_features=False, scale_features=False, add_intercept=self.add_intercept)
        x_train_strata = strata_all[:n_train]
        x_test_strata = strata_all[n_train:]
        return x_train_strata, x_test_strata


    # TODO: 
    # post-Sally meeting:
        #  more robust sampling scheme for test epsilons
        #  less conservative accounting of partial reporting (unit level predictions should adapt and not just snap to lower/upper bounds)
    def compute_bootstrap_errors(self, reporting_units, nonreporting_units, unexpected_units):
        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)
        x_all = self.featurizer.prepare_data(all_units, center_features=False, scale_features=False, add_intercept=self.add_intercept)
        n_train = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]
        
        x_train_df = self.featurizer.filter_to_active_features(x_all[:n_train])
        x_train = x_train_df.values
        y_train = reporting_units['normalized_margin'].values.reshape(-1, 1)
        z_train = reporting_units['turnout_factor'].values.reshape(-1, 1)
        weights_train = reporting_units['weights'].values.reshape(-1, 1)

        x_test_df = self.featurizer.generate_holdout_data(x_all[n_train:(n_train + n_test)])
        x_test = x_test_df.values
        # y_test = nonreporting_units['normalized_margin'].values.reshape(-1, 1)
        # z_test = nonreporting_units['turnout_factor'].values.reshape(-1, 1)
        weights_test = nonreporting_units['weights'].values.reshape(-1, 1)

        aggregate_indicator = pd.get_dummies(pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)['postal_code']).values
        aggregate_indicator_expected = aggregate_indicator[:(n_train + n_test)]
        aggregate_indicator_unexpected = aggregate_indicator[(n_train + n_test):]
        aggregate_indicator_train = aggregate_indicator_expected[:n_train]
        aggregate_indicator_test = aggregate_indicator_expected[n_train:]
        
        y_partial_reporting_lower, y_partial_reporting_upper = self._generate_nonreporting_bounds(nonreporting_units, 'normalized_margin')
        z_partial_reporting_lower, z_partial_reporting_upper = self._generate_nonreporting_bounds(nonreporting_units, 'turnout_factor')
        optimal_lambda_y = 0.1 # self.cv_lambda(x_train, y_train, np.logspace(-3, 2, 20), weights=weights_train)
        optimal_lambda_z = 0.1 # self.cv_lambda(x_train, z_train, np.logspace(-3, 2, 20), weights=weights_train)
        
        # nonreporting_units_to_keep = ((y_partial_reporting_upper - y_partial_reporting_lower) < 0.1).flatten()
        # x_all = np.concatenate([x_train, x_test[nonreporting_units_to_keep]], axis=0)
        # y_all = np.concatenate([y_train, y_test[nonreporting_units_to_keep]])
        # z_all = np.concatenate([z_train, z_test[nonreporting_units_to_keep]])
        # weights_all_y = np.concatenate([weights_train, weights_test[nonreporting_units_to_keep]])
        # weights_all_z = np.concatenate([weights_train, weights_test[nonreporting_units_to_keep]])
        # weights_bound_y = (1 / ((y_partial_reporting_upper - y_partial_reporting_lower)**2 + 1))[nonreporting_units_to_keep]
        # weights_bound_z = (1 / ((z_partial_reporting_upper - z_partial_reporting_lower)**2 + 1))[nonreporting_units_to_keep]
        # weights_all_y[n_train:] *= weights_bound_y
        # weights_all_z[n_train:] *= weights_bound_z

        # ols_y_all = OLSRegression().fit(x_all, y_all, weights=weights_all_y, lambda_=optimal_lambda_y)
        # ols_z_all = OLSRegression().fit(x_all, z_all, weights=weights_all_z, lambda_=optimal_lambda_z)

        ols_y = OLSRegression().fit(x_train, y_train, weights=weights_train, lambda_=optimal_lambda_y)
        ols_z = OLSRegression().fit(x_train, z_train, weights=weights_train, lambda_=optimal_lambda_z)

        y_train_pred = ols_y.predict(x_train)
        z_train_pred = ols_z.predict(x_train)

        residuals_y, epsilon_y_hat, delta_y_hat = self._estimate_model_errors(ols_y, x_train, y_train, aggregate_indicator_train)
        residuals_z, epsilon_z_hat, delta_z_hat = self._estimate_model_errors(ols_z, x_train, z_train, aggregate_indicator_train)
        x_train_strata, x_test_strata = self._get_strata(reporting_units, nonreporting_units)

        stratum_ppfs_delta_y, stratum_cdfs_delta_y = self._estimate_strata_dist(x_train, x_train_strata, x_test, x_test_strata, delta_y_hat, self.y_LB, self.y_UB)
        stratum_ppfs_delta_z, stratum_cdfs_delta_z = self._estimate_strata_dist(x_train, x_train_strata, x_test, x_test_strata, delta_z_hat, self.z_LB, self.z_UB)

        epsilon_B, delta_B = self._bootstrap_errors(epsilon_y_hat, epsilon_z_hat, delta_y_hat, delta_z_hat, x_train_strata, stratum_cdfs_delta_y, stratum_cdfs_delta_z, stratum_ppfs_delta_y, stratum_ppfs_delta_z, aggregate_indicator_train)
        epsilon_y_B, epsilon_z_B = epsilon_B
        delta_y_B, delta_z_B = delta_B
 
        y_train_B = y_train_pred + (aggregate_indicator_train @ epsilon_y_B) + delta_y_B
        z_train_B = z_train_pred + (aggregate_indicator_train @ epsilon_z_B) + delta_z_B
        
        ols_y_B = OLSRegression().fit(x_train, y_train_B, weights_train, normal_eqs=ols_y.normal_eqs)
        ols_z_B = OLSRegression().fit(x_train, z_train_B, weights_train, normal_eqs=ols_z.normal_eqs)

        y_train_pred_B = ols_y_B.predict(x_train)
        z_train_pred_B = ols_z_B.predict(x_train)

        residuals_y_B = ols_y_B.residuals(y_train_B, y_train_pred_B, loo=True, center=True)
        residuals_z_B = ols_z_B.residuals(z_train_B, z_train_pred_B, loo=True, center=True)

        epsilon_y_hat_B = self._estimate_epsilon(residuals_y_B, aggregate_indicator_train)
        epsilon_z_hat_B = self._estimate_epsilon(residuals_z_B, aggregate_indicator_train)

        y_test_pred_B = (ols_y_B.predict(x_test) + (aggregate_indicator_test @ epsilon_y_hat_B)).clip(min=y_partial_reporting_lower, max=y_partial_reporting_upper)
        z_test_pred_B = (ols_z_B.predict(x_test) + (aggregate_indicator_test @ epsilon_z_hat_B)).clip(min=z_partial_reporting_lower, max=z_partial_reporting_upper)
        
        yz_test_pred_B = y_test_pred_B * z_test_pred_B
        
        y_test_pred = (ols_y.predict(x_test) + (aggregate_indicator_test @ epsilon_y_hat)).clip(min=y_partial_reporting_lower, max=y_partial_reporting_upper)
        z_test_pred = (ols_z.predict(x_test) + (aggregate_indicator_test @ epsilon_z_hat)).clip(min=z_partial_reporting_lower, max=z_partial_reporting_upper)
        yz_test_pred = y_test_pred * z_test_pred

        test_residuals_y, test_residuals_z = self._sample_test_errors(
            residuals_y, 
            residuals_z, 
            epsilon_y_hat, 
            epsilon_z_hat, 
            x_test_strata, 
            stratum_ppfs_delta_y, 
            stratum_ppfs_delta_z,
            aggregate_indicator_train, 
            aggregate_indicator_test
        )
        
        self.errors_B_1 = yz_test_pred_B * weights_test

        errors_B_2 = (y_test_pred + test_residuals_y).clip(min=y_partial_reporting_lower, max=y_partial_reporting_upper)
        errors_B_2 *= (z_test_pred + test_residuals_z).clip(min=z_partial_reporting_lower, max=z_partial_reporting_upper)

        self.errors_B_2 = errors_B_2 * weights_test 

        self.errors_B_3 = z_test_pred_B * weights_test # has already been clipped above
        self.errors_B_4 = (z_test_pred + test_residuals_z).clip(min=z_partial_reporting_lower, max=z_partial_reporting_upper) * weights_test 

        # y_test_pred_all = (ols_y_all.predict(x_test) + (aggregate_indicator_test @ epsilon_y_hat)).clip(min=y_partial_reporting_lower, max=y_partial_reporting_upper)
        # z_test_pred_all = (ols_z_all.predict(x_test) + (aggregate_indicator_test @ epsilon_z_hat)).clip(min=z_partial_reporting_lower, max=z_partial_reporting_upper)
        # import IPython; IPython.embed()

        self.weighted_yz_test_pred = yz_test_pred * weights_test
        self.weighted_z_test_pred = z_test_pred * weights_test
        self.ran_bootstrap = True

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

        aggregate_yz_test_B =  aggregate_indicator_test.T @ self.errors_B_1
        aggregate_yz_test_pred =  aggregate_indicator_test.T @ self.errors_B_2
        aggregate_z_test_B = aggregate_indicator_test.T @ self.errors_B_3
        aggregate_z_test_pred = aggregate_indicator_test.T @ self.errors_B_4

        aggregate_yz_total_B = aggregate_yz_train + aggregate_yz_test_B  + aggregate_yz_unexpected
        aggregate_yz_total_pred = aggregate_yz_train + aggregate_yz_test_pred + aggregate_yz_unexpected
        aggregate_z_total_B = aggregate_z_train + aggregate_z_test_B + aggregate_z_unexpected
        aggregate_z_total_pred = aggregate_z_train + aggregate_z_test_pred + aggregate_z_unexpected

        aggregate_error_B_1 = aggregate_yz_total_B
        aggregate_error_B_2 = aggregate_yz_total_pred
        aggregate_error_B_3 = aggregate_z_total_B
        aggregate_error_B_4 = aggregate_z_total_pred



        aggregate_error_B = (aggregate_error_B_1 / aggregate_error_B_3) - (aggregate_error_B_2 / aggregate_error_B_4)

        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        aggregate_z_total = aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        aggregate_yz_total = aggregate_yz_unexpected + aggregate_yz_train + aggregate_indicator_test.T @ self.weighted_yz_test_pred
        aggregate_perc_margin_total = np.nan_to_num(aggregate_yz_total / aggregate_z_total)
        
        if 'postal_code' in aggregate:
            self.aggregate_error_B_1 = aggregate_error_B_1
            self.aggregate_error_B_2 = aggregate_error_B_2
            self.aggregate_error_B_3 = aggregate_error_B_3
            self.aggregate_error_B_4 = aggregate_error_B_4
            self.aggregate_perc_margin_total = aggregate_perc_margin_total

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

    def get_national_summary_estimates(self, nat_sum_data_dict, called_states):
        nat_sum_data_dict_sorted = sorted(nat_sum_data_dict.items())
        nat_sum_data_dict_sorted_vals = np.asarray([x[1] for x in nat_sum_data_dict_sorted]).reshape(-1, 1)
        # TODO: divide by states previous election raw margin instead of aggregate error B3/B4
        # TODO: implement called_states
        # TODO: we could get more conservative uncertainty from the hard threshold
        aggregate_dem_prob_B_1 = expit(self.T * np.nan_to_num(self.aggregate_error_B_1 / self.aggregate_error_B_3))
        aggregate_dem_prob_B_2 = expit(self.T * np.nan_to_num(self.aggregate_error_B_2 / self.aggregate_error_B_4))
        
        aggregate_dem_vals_B_1 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_1
        aggregate_dem_vals_B_2 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_2
        aggregate_dem_vals_B = np.sum(aggregate_dem_vals_B_1, axis=0) - np.sum(aggregate_dem_vals_B_2, axis=0)

        aggregate_dem_vals_pred = np.sum(nat_sum_data_dict_sorted_vals * expit(self.T * self.aggregate_perc_margin_total))
        alpha = 0.9
        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        interval_upper, interval_lower = (
            aggregate_dem_vals_pred - 
            np.quantile(
                aggregate_dem_vals_B, 
                q=[lower_q, upper_q],
                axis=-1
            ).T
        ).T

        national_summary_estimates = {'margin': [aggregate_dem_vals_pred, interval_lower, interval_upper]}

        return national_summary_estimates
