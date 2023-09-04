from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import linprog
from scipy.special import expit
import math

import logging
from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals
from elexmodel.handlers.data.Featurizer import Featurizer

LOG = logging.getLogger(__name__)


class OLSRegression(object):
    """
    A class for Ordinary Least Squares Regression.
    We have our own implementation because this allows us to save the normal equations for re-use, which 
    saves times during the bootstrap
    """

    # OLS setup: 
    #       X \beta = y
    # since X might not be square, we multiply the above equation on both sides by X^T to generate X^T X, which is guaranteed
    # to be square
    #       X^T X \beta = X^T y
    # Since X^T X is square we can invert it
    #       \beta = (X^T X)^{-1} X^T y
    # Since our version of the model bootstraps y, but keeps X constant we can 
    # pre-compute (X^T X)^{-1} X^T and then re-use it to compute \beta_b for every bootstrap sample
    
    def __init__(self):
        self.normal_eqs = None
        self.hat_matrix = None
        self.beta_hat = None

    def _compute_normal_equations(self, x: np.ndarray, L: np.ndarray, lambda_: float, n_feat_ignore_reg: int) -> np.ndarray:
        """
        Computes the Normal Equations for OLS: (X^T X)^{-1} X^T
        """
        # Inverting X^T X directly is computationally expensive and mathematically unstable, so we use QR factorization
        # which factors x into the sum of an orthogonal matrix Q and a upper tringular matrix R
        # L is a diagonal matrix of weights
        Q, R = np.linalg.qr(L @ x)
        # lambda_I is the matrix for regularization, which need to be the same shape as R and
        # have the regularization constant lambda_ along the diagonal
        lambda_I = lambda_ * np.eye(R.shape[1])
        # we don't want to regularize the coefficient for intercept and some features
        # so set regularization constant to zero for those features
        # this assumes that these are the first features
        for i in range(n_feat_ignore_reg):
            lambda_I[i, i] = 0
        # substitute X = QR into the normal equations to get
        #       R^T Q^T Q R \beta = R^T Q^T y
        #       R^T R \beta = R^T Q^T y
        #       \beta = (R^T R)^{-1} R^T Q^T y
        # since R is upper triangular it is eqsier to invert
        # lambda_I is the regularization matrix
        return np.linalg.inv(R.T @ R + lambda_I) @ R.T @ Q.T

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None =None, lambda_: float=0.0, normal_eqs: np.ndarray | None = None, n_feat_ignore_reg: int=2) -> OLSRegression:
        """
        Fits the OLS model. 
        Computes weights, computes normal equations and then computes 
        """
        # if weights is none, assume that that all weights should be 1
        if weights is None:
            weights = np.ones((y.shape[0], ))
        # normalize weights and turn into diagional matrix
        # square root because will be squared when R^T R happens later
        L = np.diag(np.sqrt(weights.flatten() / weights.sum()))
        # if normal equations are provided then use those, otherwise compute them
        if normal_eqs is not None:
            self.normal_eqs = normal_eqs
        else:
            self.normal_eqs = self._compute_normal_equations(x, L, lambda_, n_feat_ignore_reg)
        # compute hat matrix: X (X^T X)^{-1} X^T
        self.hat_vals = np.diag(x @ self.normal_eqs @ L)
        # compute beta_hat: (X^T X)^{-1} X^T y
        self.beta_hat = self.normal_eqs @ L @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Uses beta_hat to predict on new x matrix
        """
        return x @ self.beta_hat
    
    def residuals(self, y: np.ndarray, y_hat: np.ndarray, loo: bool = True, center: bool = True) -> np.ndarray:
        """
        Computes residuals for the model
        """
        # compute standard residuals
        residuals = (y - y_hat)
        # if leave one out is True, inflate by (1 - P)
        # in OLS setting inflating by (1 - P) is the same as computing the leave one out residuals
        # the un-inflated training residuals are too small, since training covariates were observed during fitting
        if loo:
            residuals /= (1 - self.hat_vals).reshape(-1, 1)
        # centering removes the column mean
        if center:
            residuals -= np.mean(residuals, axis=0)
        return residuals
    
class QuantileRegression(object):
    """
    A new version of quantile regression that uses the dual to solve faster 
    """
    def __init__(self):
        self.beta_hats = []

    def _fit(self, S: np.ndarray, Phi: np.ndarray, zeros: np.ndarray, N: int, weights: np.ndarray, tau: float) -> np.ndarray:
        """
        Fits the dual problem of a quantile regression, for more information see appendix 6 here: https://arxiv.org/pdf/2305.12616.pdf
        """
        bounds = weights.reshape(-1, 1) * np.asarray([(tau - 1, tau)] * N)
        # A_eq are the equality constraint matrix
        # b_eq is the equality constraint vector (ie. A_eq @ x = b_eq)
        # bounds are the (min, max) possible values of every element of x
        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, 
            method='highs', options={'presolve': False})
        # marginal are the dual values, since we are solving the dual this is equivalent to the primal
        return -1 * res.eqlin.marginals
            
    def fit(self, x: np.ndarray, y: np.ndarray, taus: list | float =0.5, weights: np.ndarray | None = None) -> np.ndarray:
        """
        Fits the quantile regression
        """
        # if weights is none, assume that that all weights should be 1
        if weights is None: 
            weights = np.ones((y.shape[0], ))

        S = y
        Phi = x
        zeros = np.zeros((Phi.shape[1],))
        N = y.shape[0]
        # normalize weights
        weights /= np.sum(weights)

        # _fit assumes that taus is list, so if we want to do one value of tau then turn into a list
        if isinstance(taus, float):
            taus = [taus]
        
        for tau in taus:
            self.beta_hats.append(self._fit(S, Phi, zeros, N, weights, tau))

        return self.beta_hats
    
class BootstrapElectionModelException(Exception):
    pass

class BootstrapElectionModel(BaseElectionModel):
    """
    The bootstrap election model. 
    This approach to live election modeling uses ordinary least squares regression for point predictions and the bootstrap to generate prediction intervals.
    Also, in this model we are modeling the two party margin as estimand only.

    We have found that decomposing the normalized margin into two quantities that we model separetely works better:
        y = (D^{Y} - R^{Y}) / (D^{Y} + D^{Y})
        z = (D^{Y} + R^{Y}) / (D^{Y'} + R^{Y'})
    where Y is the year we are modeling and Y' is the previous election year.
    if we define weights to be the to total two party turnout in a previous election: w = (D^{Y'} + R^{Y'})
    then we get that: wyz = (D^{Y} - R^{Y}) 
    We cannot model y directly since we also need a model for turnout in order to sum our unit level predictions to aggregate predictions
    y is the normalized two party margin and z is the turnout factor between this election and a previous election. We have found that
    modeling those two quantities is relatively easy given the county level information that we have.
    
    We define our model as:
        y_i = f_y(x) + \epsilon_y(x)
        z_i = f_z(x) + \epsilon_z(x)
    where f_y(x) and f_z(x) are ordinary least squares regressions and the epsilons are contest (state/district) level random effects.
    
    Our model also assumes that we have a baseline normalized margin (D^{Y'} - R^{Y'}) / (D^{Y'} + R^{Y'}) is one of the covariates
    since we have found that predicts y and z very well.
    """

    def __init__(self, model_settings={}):
        super().__init__(model_settings)
        self.seed = model_settings.get("seed", 0)
        self.B = model_settings.get("B", 2000)
        self.strata = model_settings.get("strata", ['county_classification']) # change this
        self.T = model_settings.get("T", 5000)
        self.hard_threshold = model_settings.get("agg_model_hard_threshold", False)
        self.district_election = model_settings.get("district_election", False)
        self.y_LB = model_settings.get("y_LB", -0.3)
        self.y_UB = model_settings.get("y_UB", 0.3)
        self.z_LB = model_settings.get("z_LB", -0.5)
        self.z_UB = model_settings.get("z_UB", 0.5)
        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.rng = np.random.default_rng(seed=self.seed)
        self.ran_bootstrap = False
        self.taus_lower = np.arange(0.01, 0.5, 0.01)
        self.taus_upper = np.arange(0.50, 1, 0.01)
        self.taus = np.concatenate([self.taus_lower, self.taus_upper])
        if 'baseline_normalized_margin' not in self.features:
            raise BootstrapElectionModelException("baseline_normalized_margin not included as feature. This is necessary for the model to work.")

    
    def cv_lambda(self, x: np.ndarray, y: np.ndarray, lambdas_: np.ndarray, weights: np.ndarray | None = None, k: int=5) -> float:
        """
        This function does k-fold cross validation for a OLS regression model given x, y and a set of lambdas to try out
        This function returns the lambda that minimizes the k-fold cross validation loss
        """
        # if weights are none assume that all samples have equal weights
        if weights is None:
            weights = np.ones((y.shape[0], 1))
        # concatenate since we need to keep x, y, weight samples together
        x_y_w = np.concatenate([x, y, weights], axis=1)
        self.rng.shuffle(x_y_w)
        # generate k chunks
        chunks = np.array_split(x_y_w, k, axis=0)
        ols = OLSRegression()
        errors = np.zeros((len(lambdas_), ))
        # for each possible lambda value perform k-fold cross validation
        # ie. train model on k-1 chunks and evaluate on one chunk (for all possible k combinations of heldout chunk)
        for i, lambda_ in enumerate(lambdas_):
            for test_chunk in range(k):
                x_y_w_test = chunks[test_chunk]
                # extract all chunks except for the current test chunk
                x_y_w_train = np.concatenate(chunks[:test_chunk] + chunks[test_chunk + 1:], axis=0)
                # undo the concatenation above
                x_test = x_y_w_test[:,:-2]
                y_test = x_y_w_test[:,-2]
                w_test = x_y_w_test[:,-1]
                x_train = x_y_w_train[:,:-2]
                y_train = x_y_w_train[:,-2]
                w_train = x_y_w_train[:,-1]
                ols_lambda = ols.fit(x_train, y_train, weights=w_train, lambda_=lambda_, n_feat_ignore_reg=2)
                y_hat_lambda = ols_lambda.predict(x_test)
                # error is the weighted sum of squares of the residual between the actual heldout y and the predicted y on the heldout set
                errors[i] += np.sum(w_test * ols_lambda.residuals(y_test, y_hat_lambda, loo=False, center=False) ** 2) / np.sum(w_test)
        # return lambda that minimizes the k-fold error
        # np.argmin returns the first occurence if multiple minimum values
        return lambdas_[np.argmin(errors)]

    def get_minimum_reporting_units(self, alpha: float) -> int:
        return 10
        #return math.ceil(-1 * (alpha + 1) / (alpha - 1))

    def _estimate_epsilon(self, residuals: np.ndarray, aggregate_indicator: np.ndarray, shrinkage: bool =False) -> np.ndarray:
        """
        This function estimates the epsilon (contest level random effects)
        """
        # the estimate for epsilon is the average of the residuals in each contest
        epsilon_hat = (aggregate_indicator.T @ residuals) / aggregate_indicator.sum(axis=0).reshape(-1, 1)
        # we can't estimate a contest level effect if only have 1 unit in that contest (since our residual can just)
        # be made equal to zero by setting the random effect to that value
        epsilon_hat[aggregate_indicator.sum(axis=0) < 2] = 0 

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

    def _estimate_delta(self, residuals: np.ndarray, epsilon_hat: np.ndarray, aggregate_indicator: np.ndarray) -> np.ndarray:
        """
        This function estimates delta (the model residuals)
        """
        # our estimate for delta is the difference between the residual and 
        # what can be explained by the contest level random effect
        return (residuals - (aggregate_indicator @ epsilon_hat)).flatten()

    def _estimate_model_errors(self, model: OLSRegression, x: np.ndarray, y: np.ndarray, aggregate_indicator: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function estimates all components of the error in our bootstrap model
        residual: the centered leave one out residual, ie the difference between our prediction and our actual training values
        epsilon_hat: our estimate for the contest level effect in our model (ie. how much did each state contribute)
        deta_hat: the difference between the residual and what our random effect can explain
        """
        # get unit level predictions for our model
        y_pred = model.predict(x)
        # compute residuals
        residuals_y = model.residuals(y, y_pred, loo=True, center=True)
        # estimate epsilon
        epsilon_y_hat = self._estimate_epsilon(residuals_y, aggregate_indicator, shrinkage=False)
        # compute delta
        delta_y_hat = self._estimate_delta(residuals_y, epsilon_y_hat, aggregate_indicator)
        return residuals_y, epsilon_y_hat, delta_y_hat       

    def _estimate_strata_dist(self, x_train: np.ndarray, x_train_strata: np.ndarray, x_test: np.ndarray, x_test_strata: np.ndarray, delta_hat: np.ndarray, lb: float, ub: float) -> Tuple[dict, dict]:
        """
        This function generates the distribution (ie. ppf/cdf) for the strata in which we want to exchange
        bootstrap errors in this model.
        """
        stratum_ppfs_delta = {}
        stratum_cdfs_delta = {}

        def ppf_creator(betas: np.ndarray, taus: np.ndarray, lb: float, ub: float) -> float:
            """
            Creates a probability point function (inverse of a cumulative distribution function -- CDF)
            Provides the value of a given percentile of the data
            """
            # because we want a smooth ppf, we want to interpolate
            return lambda p: np.interp(p, taus, betas, lb, ub)
        
        def cdf_creator(betas: np.ndarray, taus: np.ndarray) -> float:
            """
            Creates a cumulative distribution function
            Provides the probability that a value is at most x
            """
            return lambda x: np.interp(x, betas, taus, right=1)
        
        # we need the unique strata that exist in both the training and in the holdout data
        x_strata = np.unique(np.concatenate([x_train_strata, x_test_strata], axis=0), axis=0).astype(int)
        # for each stratum we want to add a worst case lower bound (for the taus between 0-0.49)
        # and upper bound (for the taus between 0.5-1) in case we see a value that is smaller/larger
        # than anything we have observed in the training data. This lower/upper bound is set 
        # manually, note that if we observe a value that is more extreme than the lower/upper bound
        # we are fine, since the quantile regression will use that instead
        for x_stratum in x_strata:
            x_train_aug = np.concatenate([x_train_strata, x_stratum.reshape(1, -1)], axis=0)
            delta_aug_lb = np.concatenate([delta_hat, [lb]])
            delta_aug_ub = np.concatenate([delta_hat, [ub]])
            betas_lower = QuantileRegression().fit(x_train_aug, delta_aug_lb, self.taus_lower)
            betas_upper = QuantileRegression().fit(x_train_aug, delta_aug_ub, self.taus_upper)

            betas = np.concatenate([betas_lower, betas_upper])

            betas_stratum = betas[:,np.where(x_stratum == 1)[0]].sum(axis=1)

            # for this stratum value create ppf
            stratum_ppfs_delta[tuple(x_stratum)] = ppf_creator(betas_stratum, self.taus, lb, ub)

            # for this stratum value create cdf
            stratum_cdfs_delta[tuple(x_stratum)] = cdf_creator(betas_stratum, self.taus)

        return stratum_ppfs_delta, stratum_cdfs_delta

    def _generate_nonreporting_bounds(self, nonreporting_units: pd.DataFrame, bootstrap_estimand: str, n_bins: int=10) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates upper and lower bounds for y and z based on the expected vote
        that we have for each unit. This is used to clip our predictions
        """
        # TODO: figure out how to better estimate margin_upper/lower_bound
        # TODO: pass in the magic numbers
        
        # turn expected for nonreporting units into decimal (also clip at 100)
        nonreporting_expected_vote_frac = nonreporting_units.percent_expected_vote.values.clip(max=100) / 100
        if bootstrap_estimand == 'normalized_margin':
            unobserved_upper_bound = 1
            unobserved_lower_bound = -1
            # the upper bound for LHS party is if all the outstanding vote go in their favour
            upper_bound = nonreporting_expected_vote_frac * nonreporting_units[bootstrap_estimand] + (1 - nonreporting_expected_vote_frac) * unobserved_upper_bound
            # the lower bound for the LHS party is if all the outstanding vote go against them
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

    def _strata_pit(self, x_train_strata, x_train_strata_unique, delta_hat, stratum_cdfs_delta):
        """
        Apply the probability integral transform for each strata
        """
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

        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)
        if self.district_election:
            all_units['postal_code-district'] = all_units[['postal_code', 'district']].agg('_'.join, axis=1)
            aggregate_indicator = pd.get_dummies(all_units['postal_code-district']).values
        else:
            aggregate_indicator = pd.get_dummies(all_units['postal_code']).values

        aggregate_indicator_expected = aggregate_indicator[:(n_train + n_test)]
        aggregate_indicator_unexpected = aggregate_indicator[(n_train + n_test):]
        aggregate_indicator_train = aggregate_indicator_expected[:n_train]
        aggregate_indicator_test = aggregate_indicator_expected[n_train:]
        
        y_partial_reporting_lower, y_partial_reporting_upper = self._generate_nonreporting_bounds(nonreporting_units, 'normalized_margin')
        z_partial_reporting_lower, z_partial_reporting_upper = self._generate_nonreporting_bounds(nonreporting_units, 'turnout_factor')
        optimal_lambda_y = self.cv_lambda(x_train, y_train, np.logspace(-3, 2, 20), weights=weights_train)
        optimal_lambda_z = self.cv_lambda(x_train, z_train, np.logspace(-3, 2, 20), weights=weights_train)
        
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
        
        # we don't regularize intercept and baseline_margin
        ols_y = OLSRegression().fit(x_train, y_train, weights=weights_train, lambda_=optimal_lambda_y, n_feat_ignore_reg=2)
        ols_z = OLSRegression().fit(x_train, z_train, weights=weights_train, lambda_=optimal_lambda_z, n_feat_ignore_reg=2)

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
        
        ols_y_B = OLSRegression().fit(x_train, y_train_B, weights_train, normal_eqs=ols_y.normal_eqs, n_feat_ignore_reg=2)
        ols_z_B = OLSRegression().fit(x_train, z_train_B, weights_train, normal_eqs=ols_z.normal_eqs, n_feat_ignore_reg=2)

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

    def _is_top_level_aggregate(self, aggregate):
        # case 1:
        # top level aggregate is postal code (ie. we are generating up to a state level -> ECV or Senate). We know this is the case
        # because aggregate length is just 1 and postal code is the only aggregate
        # case 2:
        # top level aggregate is postal code and district (ie. we are generating up to a district level -> House or State Senate). 
        # We know this is the case because aggregate length is 2 and postal code and district are the two aggregates.
        return (len(aggregate) == 1 and 'postal_code' in aggregate) or (len(aggregate) == 2 and 'postal_code' in aggregate and 'district' in aggregate)


    def get_aggregate_predictions(self, reporting_units, nonreporting_units, unexpected_units, aggregate, estimand):
        n = reporting_units.shape[0]
        m = nonreporting_units.shape[0]

        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)

        if len(aggregate) > 1:
            aggregate_temp_column_name = '-'.join(aggregate)
            all_units[aggregate_temp_column_name] = all_units[aggregate].agg('_'.join, axis=1)
            aggregate_indicator = pd.get_dummies(all_units[aggregate_temp_column_name]).values
        else:
            aggregate_indicator = pd.get_dummies(all_units[aggregate]).values
            aggregate_temp_column_name = aggregate

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
        if self._is_top_level_aggregate(aggregate):
            aggregate_sum = all_units.groupby(aggregate_temp_column_name).sum()
            self.aggregate_baseline_margin = ((aggregate_sum.baseline_dem - aggregate_sum.baseline_gop) / (aggregate_sum.baseline_turnout + 1)).values

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

        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)

        if len(aggregate) > 1:
            aggregate_temp_column_name = '-'.join(aggregate)
            all_units[aggregate_temp_column_name] = all_units[aggregate].agg('_'.join, axis=1)
            aggregate_indicator = pd.get_dummies(all_units[aggregate_temp_column_name]).values
        else:
            aggregate_indicator = pd.get_dummies(all_units[aggregate]).values
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

        divided_error_B_1 = np.nan_to_num(aggregate_error_B_1 / aggregate_error_B_3)
        divided_error_B_2 = np.nan_to_num(aggregate_error_B_2 / aggregate_error_B_4)

        aggregate_error_B = divided_error_B_1 - divided_error_B_2

        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        aggregate_z_total = aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        aggregate_yz_total = aggregate_yz_unexpected + aggregate_yz_train + aggregate_indicator_test.T @ self.weighted_yz_test_pred
        aggregate_perc_margin_total = np.nan_to_num(aggregate_yz_total / aggregate_z_total)
        
        # saves the aggregate errors in case we want to generate somem form of super-aggregate statistic (like ECV or House seats)
        if self._is_top_level_aggregate(aggregate):
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

    def get_national_summary_estimates(self, nat_sum_data_dict, called_states, base_to_add, alpha):
        # if nat_sum_data_dict is None then we assign 1 for every contest (ie. Senate or House)
        if nat_sum_data_dict is None:
            # the order does not matter since all contests have the same weight, so we can use anything as the key when sorting
            nat_sum_data_dict = {i: 1 for i in range(self.aggregate_error_B_1.shape[0])}
        if called_states is None:
            called_states = {i: -1 for i in range(self.aggregate_error_B_1.shape[0])}
        # sort in order to get in the same order as the contests, which have been sorted when getting dummies for aggregate indicators
        # in get_aggregate_prediction_intervals
        nat_sum_data_dict_sorted = sorted(nat_sum_data_dict.items())
        nat_sum_data_dict_sorted_vals = np.asarray([x[1] for x in nat_sum_data_dict_sorted]).reshape(-1, 1)

        called_states_sorted = sorted(called_states.items())
        called_states_sorted_vals = np.asarray([x[1] for x in called_states_sorted]).reshape(-1, 1) * 1.0 # multiplying by 1.0 to turn into floats
        # since we max/min the state called values with contest win probabilities, we don't want the uncalled states to have a number to max/min 
        # in order for those states to keep their original computed win probability
        called_states_sorted_vals[called_states_sorted_vals == -1] = np.nan 

        # divided_error_B_1 = np.nan_to_num(self.aggregate_error_B_1 / self.aggregate_baseline_margin.reshape(-1, 1))
        divided_error_B_1 = np.nan_to_num(self.aggregate_error_B_1 / self.aggregate_error_B_3)
        # divided_error_B_2 = np.nan_to_num(self.aggregate_error_B_2 / self.aggregate_baseline_margin.reshape(-1, 1))
        divided_error_B_2 = np.nan_to_num(self.aggregate_error_B_2 / self.aggregate_error_B_4)

        if self.hard_threshold:
            aggregate_dem_prob_B_1 = divided_error_B_1 > 0.5
            aggregate_dem_prob_B_1 = divided_error_B_2 > 0.5
        else:
            aggregate_dem_prob_B_1 = expit(self.T * divided_error_B_1)
            aggregate_dem_prob_B_2 = expit(self.T * divided_error_B_2)
        
        # since called_states_sorted_vals has value 1 if the state is called for the LHS party, maxing the probabilities
        # gives a probability of 1 for the LHS party
        # and called_states_sorted_vals has value 0 if the state is called for the RHS party, so mining with probabilities
        # gives a probability of 0 for the LHS party
        # and called_states_sorted_vals has value np.nan if the state is uncalled, since we use fmax/fmin the actual number
        # and not nan gets propagated, so we maintain the probability
        aggregate_dem_prob_B_1_called = np.fmin(np.fmax(aggregate_dem_prob_B_1, called_states_sorted_vals), called_states_sorted_vals)
        aggregate_dem_prob_B_2_called = np.fmin(np.fmax(aggregate_dem_prob_B_2, called_states_sorted_vals), called_states_sorted_vals)

        aggregate_dem_vals_B_1 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_1_called
        aggregate_dem_vals_B_2 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_2_called
        aggregate_dem_vals_B = np.sum(aggregate_dem_vals_B_1, axis=0) - np.sum(aggregate_dem_vals_B_2, axis=0)

        if self.hard_threshold:
            aggregate_dem_probs_total = self.aggregate_perc_margin_total > 0.5
        else:
            aggregate_dem_probs_total = expit(self.T * self.aggregate_perc_margin_total)

        # same as for the intervals
        aggregate_dem_probs_total_called = np.fmin(np.fmax(aggregate_dem_probs_total, called_states_sorted_vals), called_states_sorted_vals)
        aggregate_dem_vals_pred = np.sum(nat_sum_data_dict_sorted_vals * aggregate_dem_probs_total_called)
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
        national_summary_estimates = {'margin': [aggregate_dem_vals_pred + base_to_add, interval_lower + base_to_add, interval_upper + base_to_add]}

        print(national_summary_estimates)

        return national_summary_estimates
