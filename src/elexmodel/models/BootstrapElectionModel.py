from __future__ import annotations  # pylint: disable=too-many-lines

import logging

import numpy as np
import pandas as pd
from elexsolver.OLSRegressionSolver import OLSRegressionSolver
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver
from scipy.special import expit

from elexmodel.handlers.data.Featurizer import Featurizer
from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals

LOG = logging.getLogger(__name__)


class BootstrapElectionModelException(Exception):
    pass


class BootstrapElectionModel(BaseElectionModel):
    """
    The bootstrap election model.

    This model uses ordinary least squares regression for point predictions and the bootstrap to generate
    prediction intervals.

    In this setup, we are modeling normalized two party margin.
    But because we need to sum our estimand from the unit level to the aggregate level, we need
    to be able to convert normalized margin to unormalized margin (since normalized margins don't sum).
    This means on the unit level we will be modeling unnormalized margin and on the aggregate level normalized margin.

    We have found that instead of modeling the unit margin directly,
    decomposing the estimand into two quantites works better because the linear
    model assumption is more plausible for each individually.
    The two quantities are normalized margin (y) and a turnout factor (z)
            y = (D^{Y} - R^{Y}) / (D^{Y} + D^{Y})
            z = (D^{Y} + R^{Y}) / (D^{Y'} + R^{Y'})
    where Y is the year we are modeling and Y' is the previous election year.

    If we define weights to be the total two party turnout in a previous election:
        w = (D^{Y'} + R^{Y'}) -> w * y * z = (D^{Y} - R^{Y})
    so (w * y * z) is the unnormalized margin.

    We define our model as:
        y_i = f_y(x) + epsilon_y(x)
        z_i = f_z(x) + epsilon_z(x)
    where f_y(x) and f_z(x) are ordinary least squares regressions
    and the epsilons are contest (state/district) level random effects.
    """

    def __init__(self, model_settings={}):
        super().__init__(model_settings)
        self.B = model_settings.get("B", 2000)  # number of bootstrap samples
        self.strata = model_settings.get("strata", ["county_classification"])  # columns to stratify the data by
        self.T = model_settings.get("T", 5000)  # temperature for aggregate model
        self.hard_threshold = model_settings.get(
            "agg_model_hard_threshold", False
        )  # use sigmoid or hard thresold when calculating agg model
        self.district_election = model_settings.get("district_election", False)

        # upper and lower bounds for the quantile regression which define the strata distributions
        # these make sure that we can control the worst cases for the distributions in case we
        # haven't seen enough data ayet
        self.y_LB = model_settings.get("y_LB", -0.3)  # normalied margin lower bound
        self.y_UB = model_settings.get("y_UB", 0.3)  # normalized margin upper bound
        self.z_LB = model_settings.get("z_LB", -0.5)  # turnout factor lower bound
        self.z_UB = model_settings.get("z_UB", 0.5)  # turnout factor upper bound

        # percentiles to compute the strata distributions for
        self.taus_lower = np.arange(0.01, 0.5, 0.01)
        self.taus_upper = np.arange(0.50, 1, 0.01)
        self.taus = np.concatenate([self.taus_lower, self.taus_upper])

        # upper and lower bounds for normalized margin and turnout factor as to how the "outstanding vote" in
        # non-reporting units can go. Used to clip our predictions
        self.y_unobserved_lower_bound = model_settings.get("y_unobserved_lower_bound", -1.0)
        self.y_unobserved_upper_bound = model_settings.get("y_unobserved_upper_bound", 1.0)
        self.percent_expected_vote_error_bound = model_settings.get("percent_expected_vote_error_bound", 0.5)
        self.z_unobserved_upper_bound = model_settings.get("z_unobserved_upper_bound", 1.5)
        self.z_unobserved_lower_bound = model_settings.get("z_unobserved_lower_bound", 0.5)

        self.featurizer = Featurizer(self.features, self.fixed_effects)
        self.seed = model_settings.get("seed", 0)
        self.rng = np.random.default_rng(seed=self.seed)  # used for sampling
        self.ran_bootstrap = False

        # these are the max/min values for called races. Ie. if a contest is called for LHS party then the prediction/intervals should be at least lhs_called_threshold
        # if a contest is called for RHS party then the prediction/interval should be at most rhs_called_threshold (at most because the values are negative)
        self.lhs_called_threshold = 0.005
        self.rhs_called_threshold = -0.005

        # Assume that we have a baseline normalized margin
        # (D^{Y'} - R^{Y'}) / (D^{Y'} + R^{Y'}) is one of the covariates
        if "baseline_normalized_margin" not in self.features:
            raise BootstrapElectionModelException(
                "baseline_normalized_margin not included as feature. This is necessary for the model to work."
            )

    def cv_lambda(
        self, x: np.ndarray, y: np.ndarray, lambdas_: np.ndarray, weights: np.ndarray | None = None, k: int = 5
    ) -> float:
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
        errors = np.zeros((len(lambdas_),))
        # for each possible lambda value perform k-fold cross validation
        # ie. train model on k-1 chunks and evaluate on one chunk (for all possible k combinations of heldout chunk)
        for i, lambda_ in enumerate(lambdas_):
            for test_chunk in range(k):
                x_y_w_test = chunks[test_chunk]
                # extract all chunks except for the current test chunk
                x_y_w_train = np.concatenate(chunks[:test_chunk] + chunks[(test_chunk + 1) :], axis=0)  # noqa: 203
                # undo the concatenation above
                x_test = x_y_w_test[:, :-2]
                y_test = x_y_w_test[:, -2]
                w_test = x_y_w_test[:, -1]
                x_train = x_y_w_train[:, :-2]
                y_train = x_y_w_train[:, -2]
                w_train = x_y_w_train[:, -1]
                ols_lambda = OLSRegressionSolver()
                ols_lambda.fit(
                    x_train,
                    y_train,
                    weights=w_train,
                    lambda_=lambda_,
                    fit_intercept=True,
                    regularize_intercept=False,
                    n_feat_ignore_reg=1,
                )
                y_hat_lambda = ols_lambda.predict(x_test)
                # error is the weighted sum of squares of the residual between
                # the actual heldout y and the predicted y on the heldout set
                errors[i] += np.sum(
                    w_test * ols_lambda.residuals(y_test, y_hat_lambda, loo=False, center=False) ** 2
                ) / np.sum(w_test)
        # return lambda that minimizes the k-fold error
        # np.argmin returns the first occurence if multiple minimum values
        return lambdas_[np.argmin(errors)]

    def get_minimum_reporting_units(self, alpha: float) -> int:
        # arbitrary, just enough to fit coefficients
        return 10

    def _estimate_epsilon(self, residuals: np.ndarray, aggregate_indicator: np.ndarray) -> np.ndarray:
        """
        This function estimates the epsilon (contest level random effects)
        """
        # the estimate for epsilon is the average of the residuals in each contest
        epsilon_hat = (aggregate_indicator.T @ residuals) / aggregate_indicator.sum(axis=0).reshape(-1, 1)
        # we can't estimate a contest level effect if only have 1 unit in that contest (since our residual can just)
        # be made equal to zero by setting the random effect to that value
        epsilon_hat[aggregate_indicator.sum(axis=0) < 2] = 0

        return epsilon_hat

    def _estimate_delta(
        self, residuals: np.ndarray, epsilon_hat: np.ndarray, aggregate_indicator: np.ndarray
    ) -> np.ndarray:
        """
        This function estimates delta (the final unit level residual of the model)
        """
        # our estimate for delta is the difference between the residual and
        # what can be explained by the contest level random effect
        return (residuals - (aggregate_indicator @ epsilon_hat)).flatten()

    def _estimate_model_errors(
        self, model: OLSRegressionSolver, x: np.ndarray, y: np.ndarray, aggregate_indicator: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function estimates all components of the error in our bootstrap model:

        residual: the centered leave one out residual, ie the difference between
        our OLS prediction and our actual training values.
        this includes the component of the error that we want to explain using a contest level random effect

        epsilon_hat: our estimate for the contest level effect in our model
        (e.g. how much did each state contribute)

        deta_hat: the unit level error
        (ie. the difference between the OLS residual and the contest level state effect)
        """
        # get unit level predictions from OLS
        y_pred = model.predict(x)
        # compute residuals
        residuals_y = model.residuals(y, y_pred, loo=True, center=True)
        # estimate contest level effect (the average residual in units of a contest)
        epsilon_y_hat = self._estimate_epsilon(residuals_y, aggregate_indicator)
        # compute delta, which is the left over residual after removing the contest level effect
        delta_y_hat = self._estimate_delta(residuals_y, epsilon_y_hat, aggregate_indicator)
        return residuals_y, epsilon_y_hat, delta_y_hat

    def _estimate_strata_dist(
        self,
        x_train: np.ndarray,
        x_train_strata: np.ndarray,
        x_test: np.ndarray,
        x_test_strata: np.ndarray,
        delta_hat: np.ndarray,
        lb: float,
        ub: float,
    ) -> tuple[dict, dict]:
        """
        This function generates the distribution (ie. CDF/PPF) for the strata in which we want to exchange
        bootstrap errors in this model.
        """
        stratum_ppfs_delta = {}
        stratum_cdfs_delta = {}

        def ppf_creator(betas: np.ndarray, taus: np.ndarray, lb: float, ub: float) -> float:
            """
            Creates a probability point function (inverse of a cumulative distribution function -- CDF)
            Given a percentile, provides the value of the CDF at that point
            """
            # we interpolate, because we want to return smooth betas
            return lambda p: np.interp(p, taus, betas, lb, ub)

        def cdf_creator(betas: np.ndarray, taus: np.ndarray) -> float:
            """
            Creates a cumulative distribution function (CDF)
            Provides the probability that a value is at most x
            """
            # interpolates because we want to provide smooth probabilites
            return lambda x: np.interp(x, betas, taus, right=1)

        # we need the unique strata that exist in both the training and in the holdout data
        # since we want a distribution for all strata
        x_strata = np.unique(np.concatenate([x_train_strata, x_test_strata], axis=0), axis=0).astype(int)

        # We compute the probability distribution for each stratum by fitting quantile regressions
        # this works because the regression covariates are only dummy variables that define
        # the strata. Therefore the i-th coefficient for a quantile regression at level tau is the
        # tau-th delta for where dummy == 1
        # ie. if tau is 0.5 and there are two unique dummies (x = [[0, 1], [1, 0], ...]), then
        # the first coefficient is the median of all deltas where x = [1, 0] and the second coefficient
        # is the median of all deltas where x = [0, 1]
        # since the covariates define the strata, we get the tau-th (e.g. median or 30th percentile)
        # delta per strata as the beta for that regression, which defines a probability distribution.

        # for each stratum we want to add a worst case lower bound (for the taus between 0-0.49)
        # and upper bound (for the taus between 0.5-1) so that if we sample a uniform from a different
        # stratum that is larger/smaller than any one we have seen in this strata, we have a worst case
        # value that is larger/smaller than what we saw in that strata. We do this by simply adding the
        # lower/upper bound to the regression, one pair for each stratum.
        # This lower/upper bound is set manually, note that if we observe a value that is more extreme than
        # the lower/upper bound we are fine, since the quantile regression will use that instead
        for x_stratum in x_strata:
            x_train_aug = np.concatenate([x_train_strata, x_stratum.reshape(1, -1)], axis=0)

            delta_aug_lb = np.concatenate([delta_hat, [lb]])
            delta_aug_ub = np.concatenate([delta_hat, [ub]])

            # fit the regressions to create the probability distributions
            # for a single regression beta[i] is the tau-th (e.g. median or 30th percentile)
            # for where dummy variable position i is equal to 1
            # since we are fitting many quantile regressions at the same time, our beta is
            # beta[tau, i] where tau stretches from 0.01 to 0.99
            qr_lower = QuantileRegressionSolver()
            qr_lower.fit(x_train_aug, delta_aug_lb, self.taus_lower, fit_intercept=False)
            betas_lower = qr_lower.coefficients

            qr_upper = QuantileRegressionSolver()
            qr_upper.fit(x_train_aug, delta_aug_ub, self.taus_upper, fit_intercept=False)
            betas_upper = qr_upper.coefficients

            betas = np.concatenate([betas_lower, betas_upper])

            # for each strata, we take the betas that belong to that stratum
            # ie. for stratum [0, 1, 0] we take the betas (there is one for each tau between 0.01, 0.99])
            # at position 1 (0-indexed)

            # get all the betas for where x_stratum has a 1 (ie [1, 0, 0] position 0, [0, 1, 0] position 1 etc.)
            betas_stratum = betas[:, np.where(x_stratum == 1)[0]].sum(axis=1)

            # for this stratum value create ppf
            # we want the lower bounds and upper bounds to be the actual lower and upper values taken from beta
            stratum_ppfs_delta[tuple(x_stratum)] = ppf_creator(
                betas_stratum, self.taus, np.min(betas_stratum), np.max(betas_stratum)
            )

            # for this stratum value create cdf
            stratum_cdfs_delta[tuple(x_stratum)] = cdf_creator(betas_stratum, self.taus)

        return stratum_ppfs_delta, stratum_cdfs_delta

    # TODO: figure out how to better estimate margin_upper/lower_bound
    def _generate_nonreporting_bounds(
        self, nonreporting_units: pd.DataFrame, bootstrap_estimand: str, n_bins: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function creates upper and lower bounds for y and z based on the expected vote
        that we have for each unit. This is used to clip our predictions
        """

        # turn expected for nonreporting units into decimal (also clip at 100)
        nonreporting_expected_vote_frac = nonreporting_units.percent_expected_vote.values.clip(max=100) / 100
        if bootstrap_estimand == "results_normalized_margin":
            unobserved_upper_bound = self.y_unobserved_upper_bound
            unobserved_lower_bound = self.y_unobserved_lower_bound
            # the upper bound for LHS party if all the outstanding vote go in their favour
            upper_bound = (
                nonreporting_expected_vote_frac * nonreporting_units[bootstrap_estimand]
                + (1 - nonreporting_expected_vote_frac) * unobserved_upper_bound
            )
            # the lower bound for the LHS party if all the outstanding vote go against them
            lower_bound = (
                nonreporting_expected_vote_frac * nonreporting_units[bootstrap_estimand]
                + (1 - nonreporting_expected_vote_frac) * unobserved_lower_bound
            )
        elif bootstrap_estimand == "turnout_factor":
            # our error bound for how much error we think our results provider has with expected vote
            # e.g. 0.7 of the vote is in for a county, if percent_expected_vote_error_bound is 0.1
            #   then we are saying that we believe between 0.6 and 0.8 of the vote is in for that county
            percent_expected_vote_error_bound = self.percent_expected_vote_error_bound
            unobserved_upper_bound = self.z_unobserved_upper_bound
            unobserved_lower_bound = self.z_unobserved_lower_bound
            # inflate or deflate turnout factor appropriately
            lower_bound = nonreporting_units[bootstrap_estimand] / (
                nonreporting_expected_vote_frac + percent_expected_vote_error_bound
            )
            upper_bound = nonreporting_units[bootstrap_estimand] / (
                nonreporting_expected_vote_frac - percent_expected_vote_error_bound
            ).clip(min=0.01)
            # if 0 percent of the vote is in, the upper bound would be zero if we used the above
            # code. So instead we set it to the naive bound
            upper_bound[np.isclose(upper_bound, 0)] = unobserved_upper_bound

        # if percent reporting is 0 or 1, don't try to compute anything and revert to naive bounds
        lower_bound[
            np.isclose(nonreporting_expected_vote_frac, 0) | np.isclose(nonreporting_expected_vote_frac, 1)
        ] = unobserved_lower_bound
        upper_bound[
            np.isclose(nonreporting_expected_vote_frac, 0) | np.isclose(nonreporting_expected_vote_frac, 1)
        ] = unobserved_upper_bound

        return lower_bound.values.reshape(-1, 1), upper_bound.values.reshape(-1, 1)

    def _strata_pit(
        self,
        x_train_strata: pd.DataFrame,
        x_train_strata_unique: np.ndarray,
        delta_hat: np.ndarray,
        stratum_cdfs_delta: dict,
    ) -> np.ndarray:
        """
        Apply the probability integral transform for each strata
        """
        # We convert the deltas in the training data to their percentiles in uniform space
        unifs = []
        # for each strata, apply the CDF to the residual to turn the residual into perecntil
        for strata_dummies in x_train_strata_unique:
            # grab all deltas that belong to strata defined by strata_dummies
            delta_strata = delta_hat[np.where((strata_dummies == x_train_strata).all(axis=1))[0]]

            # plug the deltas into their CDF to get uniform random variables (probability integral transform)
            # 1e-6 is added to solve with numerical issues
            unifs_strata = stratum_cdfs_delta[tuple(strata_dummies)](delta_strata + 1e-6).reshape(-1, 1)

            # if the uniform is close to 1/0 set percentile to 0.01/0.99 also for numerical issues
            unifs_strata[np.isclose(unifs_strata, 1)] = np.max(self.taus)
            unifs_strata[np.isclose(unifs_strata, 0)] = np.min(self.taus)

            unifs.append(unifs_strata)
        return np.concatenate(unifs).reshape(-1, 1)

    def _bootstrap_deltas(
        self,
        unifs: np.ndarray,
        x_train_strata: pd.DataFrame,
        x_train_strata_unique: np.ndarray,
        stratum_ppfs_delta_y: dict,
        stratum_ppfs_delta_z: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap deltas (unit level errors of our entire model)
        this is done by re-sampling from uniform random variables
        """
        n_train = unifs.shape[0]

        # re-sample uniform random variables
        # we are re-sampling over all uniforms (not just per strata), which gives us more randomness
        # but we can guarantee that the deltas per strata will be correct because when we convert
        # the uniforms back, we use the distribution of the stratum that the training point came from.
        # also note unifs is of size: (n_train, self.B, 2) so we are re-sampling the deltas
        # for y and z jointly
        unifs_B = self.rng.choice(unifs, (n_train, self.B), replace=True)

        delta_y_B = np.zeros((n_train, self.B))
        delta_z_B = np.zeros((n_train, self.B))

        # convert percentile (uniforms) back into residuals
        # we need to do this separately for each strata because there is a different
        # inverse function per stratum
        for strata_dummies in x_train_strata_unique:
            # grab all training indices where units belong to strata strata_dummies
            strata_indices = np.where((strata_dummies == x_train_strata).all(axis=1))[0]
            # get their corresponding uniform random variables
            unifs_strata = unifs_B[strata_indices]
            # convert back to deltas. unifs_stratas last dimension defines either y or z
            delta_y_B[strata_indices] = stratum_ppfs_delta_y[tuple(strata_dummies)](unifs_strata[:, :, 0])
            delta_z_B[strata_indices] = stratum_ppfs_delta_z[tuple(strata_dummies)](unifs_strata[:, :, 1])
        return delta_y_B, delta_z_B

    # TODO: what if hat_epsilon_y, hat_epsilon_z are correlated?
    def _bootstrap_epsilons(
        self,
        epsilon_y_hat: np.ndarray,
        epsilon_z_hat: np.ndarray,
        x_train_strata: pd.DataFrame,
        x_train_strata_unique: np.ndarray,
        stratum_ppfs_delta_y: dict,
        stratum_ppfs_delta_z: dict,
        aggregate_indicator_train: np.ndarray,
    ) -> tuple[np.ndarray.np.ndarray]:
        """
        Bootstrap epsilons (contest level random effects) using the parametric bootstrap

        In a random effects model we assume that the contest level random effects are drawn
        from a normal distribution with mean zero and variance Sigma
        epsilon_y, epsilon_z ~ N(0, Sigma)
        and for now we assume Sigma is diagional
        (e.g. contest level random effects between normalized margin and turnout are uncorrelated)
        """
        # We first fit the parameters of the multivariate normal distribution that we are assuming epsilon comes
        # from and then we sample from it B times to generate bootstrapped contest level random effects.

        # The mean of our normal distribution is epsilon_hat (our best guess for contest level effects)
        # we do this to maintain the sign of the contest level effect

        # We now need to estimate Sigma, the covariance matrix of our contest level effects.
        # We assume that the contest level random effects between contests is uncorrelated so the diagonals
        # of \Sigma are zero.
        # To estimate the variance (diagonals) we use the sum of the variances of the units within a contest.
        # This is because residuals = epsilons + delta, if we have observed units in a contest then
        #   the true epsilon is no longer a random quantity and no residual_{i, j} ~ N(epsilon_{i}, \sigma_{j})
        #   (for i in contest and j in strata)
        #   which means that var(residuals) = var(delta). So that epsilon_hat = mean(residuals)
        #   so that var(epsilon_hat) = var(mean(residuals)) = mean(var(residuals)) = mean(var(delta))

        #   The variance of deltas is defined by their probability distribution (it's the same per stratum
        #   since we assume that deltas are iid within a stratum). So we use the distribution to compute
        #   the variance of the delta. We use interquartile-range (IQR) since this is more robust than
        #   sample standard deviation

        # we first compute the unit variance estimate for each stratum (since that defines the variance per unit)
        iqr_y_strata = {}
        iqr_z_strata = {}
        for x_stratum in x_train_strata_unique:
            x_stratum_delta_y_ppf = stratum_ppfs_delta_y[tuple(x_stratum)]
            iqr_y = x_stratum_delta_y_ppf(0.75) - x_stratum_delta_y_ppf(0.25)
            iqr_y_strata[tuple(x_stratum)] = iqr_y

            x_stratum_delta_z_ppf = stratum_ppfs_delta_z[tuple(x_stratum)]
            iqr_z = x_stratum_delta_z_ppf(0.75) - x_stratum_delta_z_ppf(0.25)
            iqr_z_strata[tuple(x_stratum)] = iqr_z

        # set variance per contest to be zero
        var_epsilon_y = np.zeros((aggregate_indicator_train.shape[1],))
        var_epsilon_z = np.zeros((aggregate_indicator_train.shape[1],))

        # we are now adding the unit variances to create the contest level variances
        for strata_dummies in x_train_strata_unique:
            # grab the indices of the units per state that within strata defined by strata_dummies
            strata_indices = np.where((strata_dummies == x_train_strata).all(axis=1))[0]
            # add variances
            # we square because IQR approximates standard deviation
            var_epsilon_y += (
                aggregate_indicator_train[strata_indices] * (iqr_y_strata[tuple(strata_dummies)] ** 2)
            ).sum(axis=0)
            var_epsilon_z += (
                aggregate_indicator_train[strata_indices] * (iqr_z_strata[tuple(strata_dummies)] ** 2)
            ).sum(axis=0)

        # IQR constant for a normal random variable
        iqr_scale = 1.349
        var_epsilon_y /= iqr_scale**2
        var_epsilon_z /= iqr_scale**2
        var_epsilon_y /= aggregate_indicator_train.sum(axis=0)
        var_epsilon_z /= aggregate_indicator_train.sum(axis=0)

        # if we only have 1 unit in a contest we define the variance to be zero
        var_epsilon_y[aggregate_indicator_train.sum(axis=0) < 2] = 0
        var_epsilon_z[aggregate_indicator_train.sum(axis=0) < 2] = 0

        # sample B new epsilons
        epsilon_y_B = self.rng.multivariate_normal(
            mean=epsilon_y_hat.flatten(), cov=np.diag(var_epsilon_y), size=self.B
        ).T
        epsilon_z_B = self.rng.multivariate_normal(
            mean=epsilon_z_hat.flatten(), cov=np.diag(var_epsilon_z), size=self.B
        ).T
        return epsilon_y_B, epsilon_z_B

    def _bootstrap_errors(
        self,
        epsilon_y_hat: np.ndarray,
        epsilon_z_hat: np.ndarray,
        delta_y_hat: np.ndarray,
        delta_z_hat: np.ndarray,
        x_train_strata: pd.DataFrame,
        stratum_cdfs_y: dict,
        stratum_cdfs_z: dict,
        stratum_ppfs_delta_y: dict,
        stratum_ppfs_delta_z: dict,
        aggregate_indicator_train: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        Bootstrap the errors of our model (epsilon and delta)
        """
        # get unique strata that appea in the training data
        x_train_strata_unique = np.unique(x_train_strata, axis=0).astype(int)

        # bootstrap the epsilons. Uses the parametric bootstrap
        epsilon_y_B, epsilon_z_B = self._bootstrap_epsilons(
            epsilon_y_hat,
            epsilon_z_hat,
            x_train_strata,
            x_train_strata_unique,
            stratum_ppfs_delta_y,
            stratum_ppfs_delta_z,
            aggregate_indicator_train,
        )

        # turn deltas into percentiles (uniforms) so that we can re-sample those
        unifs_y = self._strata_pit(x_train_strata, x_train_strata_unique, delta_y_hat, stratum_cdfs_y)
        unifs_z = self._strata_pit(x_train_strata, x_train_strata_unique, delta_z_hat, stratum_cdfs_z)
        unifs = np.concatenate([unifs_y, unifs_z], axis=1)

        # re-sample uniforms and apply the inverse CDF (PPF) to convert back to re-sampled residuals
        delta_y_B, delta_z_B = self._bootstrap_deltas(
            unifs, x_train_strata, x_train_strata_unique, stratum_ppfs_delta_y, stratum_ppfs_delta_z
        )

        return (epsilon_y_B, epsilon_z_B), (delta_y_B, delta_z_B)

    def _sample_test_delta(
        self, x_test_strata: pd.DataFrame, stratum_ppfs_delta_y: dict, stratum_ppfs_delta_z: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function generates new test deltas (unit level errors)
        """
        # we previously estimated the distribution of the deltas per stratum
        # we can now generate a new set of uniform random variables (percentiles)
        # and then convert them to deltas using the deltas PPFs. We just need
        # to make sure that we use the correct distribution defined by the strata

        n_test = x_test_strata.shape[0]

        # sample a new set of uniforms
        test_unifs = self.rng.uniform(low=0, high=1, size=(n_test, self.B, 2))

        test_delta_y = np.zeros((n_test, self.B))
        test_delta_z = np.zeros((n_test, self.B))

        x_test_strata_unique = np.unique(x_test_strata, axis=0).astype(int)
        for strata_dummies in x_test_strata_unique:
            # get indices of the nonreporting data that are in strata strata_dummies
            strata_indices = np.where((strata_dummies == x_test_strata).all(axis=1))[0]
            # get their corresponding newly sampled uniforms
            unifs_strata = test_unifs[strata_indices]
            # use PPF to generate new deltas from the new uniforms
            test_delta_y[strata_indices] = stratum_ppfs_delta_y[tuple(strata_dummies)](unifs_strata[:, :, 0])
            test_delta_z[strata_indices] = stratum_ppfs_delta_z[tuple(strata_dummies)](unifs_strata[:, :, 1])

        return test_delta_y, test_delta_z

    def _sample_test_epsilon(
        self,
        residuals_y: np.ndarray,
        residuals_z: np.ndarray,
        epsilon_y_hat: np.ndarray,
        epsilon_z_hat: np.ndarray,
        aggregate_indicator_train: np.ndarray,
        aggregate_indicator_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function generates new test epsilons (contest level random effects)
        NOTE: for now we are sampling from a normal with mean zero and sample variance.
        """
        # the contests that need a sampled contest level random effect are those that still have outstanding
        # units (since for the others, the contest level random effect is a known fixed quantity) but no
        # current estimate for the epsilon in that contest (so ones with NO training examples)

        # this gives us indices of contests for which we have no samples in the training set
        contests_not_in_reporting_units = np.where(np.all(aggregate_indicator_train == 0, axis=0))[0]
        # gives us contests for which there is at least one county not reporting
        contests_in_nonreporting_units = np.where(np.any(aggregate_indicator_test > 0, axis=0))[0]
        # contests that need a random effect are:
        #   contests that we want to make predictions on (ie. there are still outstanding units)
        #   BUT no units in that contest are reporting
        contests_that_need_random_effect = np.intersect1d(
            contests_not_in_reporting_units, contests_in_nonreporting_units
        )

        # \epsilon ~ N(0, \Sigma)
        mu_hat = [0, 0]

        # the variance is the sample variance of epsilons for which we have an estimate
        non_zero_epsilon_indices = np.nonzero(epsilon_y_hat)[0]
        # if there is only one non zero contest OR if the contest is a state level election only
        # then just sample from nearly zero since no variance
        if non_zero_epsilon_indices.shape[0] == 1:
            return np.zeros((1, self.B)), np.zeros((1, self.B))

        sigma_hat = np.cov(np.concatenate([epsilon_y_hat, epsilon_z_hat], axis=1)[np.nonzero(epsilon_y_hat)[0]].T)

        # sample new test epsilons, but only for states that need tem
        test_epsilon = self.rng.multivariate_normal(
            mu_hat, sigma_hat, size=(len(contests_that_need_random_effect), self.B)
        )

        test_epsilon_y = aggregate_indicator_test[:, contests_that_need_random_effect] @ test_epsilon[:, :, 0]
        test_epsilon_z = aggregate_indicator_test[:, contests_that_need_random_effect] @ test_epsilon[:, :, 1]
        return test_epsilon_y, test_epsilon_z

    def _sample_test_errors(
        self,
        residuals_y: np.ndarray,
        residuals_z: np.ndarray,
        epsilon_y_hat: np.ndarray,
        epsilon_z_hat: np.ndarray,
        x_test_strata: pd.DataFrame,
        stratum_ppfs_delta_y: dict,
        stratum_ppfs_delta_z: dict,
        aggregate_indicator_train: np.ndarray,
        aggregate_indicator_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function samples new test errors for our model (ie. new test residuals)
        """
        # we generate new test residuals by generating new test epsilons
        # and new test deltas and then adding them together
        test_epsilon_y, test_epsilon_z = self._sample_test_epsilon(
            residuals_y, residuals_z, epsilon_y_hat, epsilon_z_hat, aggregate_indicator_train, aggregate_indicator_test
        )
        test_delta_y, test_delta_z = self._sample_test_delta(x_test_strata, stratum_ppfs_delta_y, stratum_ppfs_delta_z)
        test_error_y = test_epsilon_y + test_delta_y
        test_error_z = test_epsilon_z + test_delta_z
        return test_error_y, test_error_z

    # TODO: potentially generalize binning features for strata
    def _get_strata(
        self, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gets strata for stratified bootstrap sampling
        """
        n_train = reporting_units.shape[0]
        # we can use the featurizer, since strata are defined by dummy variables in the same way that
        # fixed effects are (ie. rural could be (1, 0, 0) while urban could be (0, 1, 0) while suburban
        # could be (0, 0, 1))
        # but like with fixed effects we drop one strata category and use the intercept instead so the
        # example would be
        # rural: 0, 0 urban: 1, 0 and rural: 0, 1
        strata_featurizer = Featurizer([], self.strata)
        all_units = pd.concat([reporting_units, nonreporting_units], axis=0)

        strata_all = strata_featurizer.prepare_data(
            all_units, center_features=False, scale_features=False, add_intercept=self.add_intercept
        )
        x_train_strata = strata_all[:n_train]
        x_test_strata = strata_all[n_train:]
        return x_train_strata, x_test_strata

    def compute_bootstrap_errors(
        self, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, unexpected_units: pd.DataFrame
    ):
        """
        Computes unit level point predictions and runs the bootstrap to generate quantities needed for
        prediction intervals.

        The bootstrap generally re-samples the observed data with replacement in order to generate synthentic
            "bootstrap" datasets, which can be used to estimate the sampling distribution of a quantity that we
            are interested in.
        Our implementation is the stratified residual bootstrap. The residual bootstrap samples residuals of a
            model (instead of the original dataset).
            This is preferable in a regression setting because it removes the the component of the observation that is
            not random.
        We use the stratified bootstrap because our units are not independent and identically distributed, which means
            that we cannot assign the error of any unit to any other unit (e.g. the residual for an urban unit would
            likely not fit for a rural unit). For now, this model stratifies on county classification
            (rural/urban/suburban).

        Generally we are interested in predicting functions of:
                w * y * z = weights * normalized_margin * turnout_factor = unnormalized_margin

        There are three cases:
            1) In the unit case we are interested in the unnormalized margin:
                    w_i * y_i * z_i
            2) In the aggregate (e.g. state aggregation) case we are interested in the normalized sum of the
               unnormalized margin of units
                    (sum_{i = 1}^N w_i * y_i * z_i) / (sum_{i = 1}^N w_i * z_i)
            3) In the national case we are interested in an interval over the sum of electoral votes generated by
               the predictions
                    sum_{s = 1}^{51} sigmoid{sum_{i = 1}^{N_s} w_i * y_i * z_i} * ev_s

        Our point prediction for each is:
            1) w_i * hat{y_i} * hat{z_i}
            2) (sum_{i = 1}^N w_i * hat{y_i} * hat_{z_i}) / (sum_{i = 1}^N w_i hat{z_i})
            3) sum_{s = 1}^{51} sigmoid{sum_{i=1}^{N_s} w_i * hat{y_i} * hat{z_i}} * ev_s
        remember that our model for y_i, z_i is OLS plus a contest random effect:
                y_i = f_y(x) + epsilon_y(x)
                z_i = f_z(x) + epsilon_z(x)
        this means that:
                hat{y_i} = hat{f_y(x)} + hat{epsilon_y(x)}
                hat{z_i} = hat{f_z(x)} + hat{epsilon_z(x)}

        As point predictions, this function only computes:
            1) w_i * hat{y_i} * hat{z_i}
            2) w_i * hat{z_i}
        which are then used in the respective prediction functions

        We are also interested in generating prediction intervals for the quantities, we do that by bootstrapping
            the error in our predictions and then taking the appropriate percentiles of those errors.
            The errors we are interested in are between the true quantity and our prediction:

        There are three cases that mirror the cases above (error between prediction and true quantity)
            1) w_i * hat{y_i} * hat{z_i} - w_i * y_i * z_i
            2) ((sum_{i = 1}^N w_i * hat{y_i} * hat{z_i}) / (sum_{i = 1}^n w_i hat{z_i}) -
               (sum_{i = 1}^N w_i * y_i * z_i) / (sum_{i = 1}^n w_i z_i))
            3) (sum_{s = 1}^{51} sigmoid{sum_{i=1}^{N_s} w_i * hat{y_i} * hat{z_i}} *
                ev_s - sum_{s = 1}^{51} sigmoid{sum_{i=1}^{N_s} w_i * y_i * z_i} * ev_s)

        In order to keep this model as flexible as possible for all potential cases, this function generates
            bootstrap estimates for:
            1) w_i * hat{y_i} * hat{z_i}
            2) w_i * y_i * z_i
            3) w_i * hat{z_i}
            4) w_i * z_i
        and store those so that we can later compute prediction interevals for any function of these quantities in their
            respective functions.

        In a normal setting the bootstrap works by assuming that our fitted predictions
            (e.g. w_i * hat{y_i} * hat{z_i}) is now the true value. Then using the bootstrap to generate synthentic
            samples (e.g. w_i hat{y_i}^b * hat{z_i}^b) and computing the error between the two.
            This would give us a confidence interval (ie. the error between a quantity and it's mean),
            but we are interested in a prediction interval, which means we also need to take into account
            the additional uncertainty in sampling new y_i and z_i.

        This means that our new "true" quantity (the equivalent of w_i * y_i * z_i)
            needs a new fresh sampled uncertainty, so we sample new test errors
                residuals_{y, i}^{b}, residuals_{z, i}^{b}
            in order to compute:
                hat{y_i} + residuals_{y, i}^{b}
                hat{z_i} + residuals_{z, i}^{b}
            so that:
                w_i * y_i * z_i     -->     w_i * (hat{y_i} + residuals_{y, i}^{b}) * (hat{z_i} + residuals_{z, i}^{b})

        We also need new "estimated" quantities (the equivalent of w_i * hat{y_i} * hat{z_i}), these are the outcome
            of the stratified residual bootstrap:
                w_i * hat{y_i} * hat{z_i}    -->     w_i * tilde{y_i}^{b} * tilde{z_i}^{b}

        For completeness, we also generate estimates for the other two quantities:
                w_i * z_i   ->      w_i * (hat{z_i} + epsilon_{z, i}^{b})
                w_i * hat{z_i}     ->      w_i * tilde{z_i}^{b}
        """
        # prepare data (generate fixed effects, add intercept etc.)
        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)
        x_all = self.featurizer.prepare_data(
            all_units, center_features=False, scale_features=False, add_intercept=self.add_intercept
        )
        n_train = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]

        x_train_df = self.featurizer.filter_to_active_features(x_all[:n_train])
        x_train = x_train_df.values
        y_train = reporting_units["results_normalized_margin"].values.reshape(-1, 1)
        z_train = reporting_units["turnout_factor"].values.reshape(-1, 1)
        weights_train = reporting_units["baseline_weights"].values.reshape(-1, 1)

        x_test_df = self.featurizer.generate_holdout_data(x_all[n_train : (n_train + n_test)])  # noqa: 203
        x_test = x_test_df.values
        weights_test = nonreporting_units["baseline_weights"].values.reshape(-1, 1)

        # Create a matrix of size (n_contests, n_total_units) which acts as a crosswalk
        # between unit and contest (ie. a 1 in i,j says that unit j belongs to contest i)
        # in case district election we need to create a variable that defines the state, district
        # which is what the contest is
        if self.district_election:
            all_units["postal_code-district"] = all_units[["postal_code", "district"]].agg("_".join, axis=1)
            aggregate_indicator = pd.get_dummies(all_units["postal_code-district"]).values
        else:
            aggregate_indicator = pd.get_dummies(all_units["postal_code"]).values

        aggregate_indicator_expected = aggregate_indicator[: (n_train + n_test)]
        aggregate_indicator_train = aggregate_indicator_expected[:n_train]
        aggregate_indicator_test = aggregate_indicator_expected[n_train:]

        # we compute bounds for normalized margin and turnout factor
        # based on our results providers current estimate for expected vote
        # ie. if 95% of the votes of a unit are in,
        # what is the max/min the normalized_margin and turnout factor could still reach?
        y_partial_reporting_lower, y_partial_reporting_upper = self._generate_nonreporting_bounds(
            nonreporting_units, "results_normalized_margin"
        )
        z_partial_reporting_lower, z_partial_reporting_upper = self._generate_nonreporting_bounds(
            nonreporting_units, "turnout_factor"
        )

        # we use k-fold cross validation to find the optimal lambda for our OLS regression
        optimal_lambda_y = self.cv_lambda(x_train, y_train, np.logspace(-3, 2, 20), weights=weights_train)
        optimal_lambda_z = self.cv_lambda(x_train, z_train, np.logspace(-3, 2, 20), weights=weights_train)

        # step 1) fit the initial model
        # we don't want to regularize the intercept or the coefficient for baseline_normalized_margin
        ols_y = OLSRegressionSolver()
        ols_y.fit(
            x_train,
            y_train,
            weights=weights_train,
            lambda_=optimal_lambda_y,
            fit_intercept=True,
            regularize_intercept=False,
            n_feat_ignore_reg=1,
        )
        ols_z = OLSRegressionSolver()
        ols_z.fit(
            x_train,
            z_train,
            weights=weights_train,
            lambda_=optimal_lambda_z,
            fit_intercept=True,
            regularize_intercept=False,
            n_feat_ignore_reg=1,
        )

        # step 2) calculate the fitted values
        y_train_pred = ols_y.predict(x_train)
        z_train_pred = ols_z.predict(x_train)

        # step 3) calculate residuals
        #   residuals are the residuals from OLS (ie. the amount of error that our regression does not explain)
        #       they have been inflated by (1 - hat_matrix) to be leave-one-out residuals so they are an estimate of
        #       the test residuals
        #   epsilon is the contest level effect
        #   (it is estiamted as the average error in OLS over all the units in a contest)
        #   delta is the unit level error that is unaccounted for by the contest level effect
        residuals_y, epsilon_y_hat, delta_y_hat = self._estimate_model_errors(
            ols_y, x_train, y_train, aggregate_indicator_train
        )
        residuals_z, epsilon_z_hat, delta_z_hat = self._estimate_model_errors(
            ols_z, x_train, z_train, aggregate_indicator_train
        )

        # As part of the residual bootstrap we now need to generate B synthetic versions of residuals_y and residuals_z
        # residuals are broken into an epsilon (contest level error) and delta (unit level error component). This means
        # in order to generate bootstrapped residuals we need to bootstrap samples for epsilon and for delta.
        # We bootstrap epsilons using the parametric bootstrap. Our procedure for bootstrapping the deltas is more
        # complicated.

        # The stratified bootstrap assumes that the final model residuals (delta) are independent and identically
        # distributed per strata. So we now need to generate new bootstrapped deltas for each unit in each strata.

        # Instead of sampling the deltas directly with replacement we generate a probability distributionsfor each
        # strata. We convert the deltas into uniform random variables using the distributions CDF
        # (probability integral transform), we then re-sample the uniforms with replacement and then convert the
        # uniform random variables back into deltas using the PPF of the distribution (inverse CDF).
        # We do this because we have two deltas to sample (delta_z and delta_y) by sampling the observed CDFs we
        # can maintain the correlation between the y and z deltas.
        #   E.g. take a rural delta (delta_y, delta_z), evaluate CDF -> get percentile for each residual (0.75, 0.85)
        #   If you do this for every pair of points, you will notice that these two percentiles are correlated
        #   (a big y error and a big z error co-occur)
        # This approach using uniform random variables allows us to smooth over the distribution in cases where we
        # have only seen very few obserations per strata. We can also impose a worst/best case scenario by adding an
        # additional datapoint for each stratum when generating the distribution.

        # we only want re-sample deltas in each strata (ie. rural counties should only receive errors from rural
        # counties, same for suburban and urban). this means that we need to generate error distributions
        # conditional on each strata value (ie. conditional on urban, rural and suburban) to do this, we first
        # need to get the strata.
        # x_train_strata is an array where each row defines the strata for that training unit (e.g. 00, 01, 10)
        # x_test_strata is equivalent
        x_train_strata, x_test_strata = self._get_strata(reporting_units, nonreporting_units)

        # we then compute the probability distribution (CDF/PPF) for the deltas given each strata, this will
        # allow us to move from the residual space to the percentile space [0, 1] and back again after re-sampling
        stratum_ppfs_delta_y, stratum_cdfs_delta_y = self._estimate_strata_dist(
            x_train, x_train_strata, x_test, x_test_strata, delta_y_hat, self.y_LB, self.y_UB
        )
        stratum_ppfs_delta_z, stratum_cdfs_delta_z = self._estimate_strata_dist(
            x_train, x_train_strata, x_test, x_test_strata, delta_z_hat, self.z_LB, self.z_UB
        )

        # step 4) bootstrap resampling
        # step 4a) we resample B new epsilons and deltas
        epsilon_B, delta_B = self._bootstrap_errors(
            epsilon_y_hat,
            epsilon_z_hat,
            delta_y_hat,
            delta_z_hat,
            x_train_strata,
            stratum_cdfs_delta_y,
            stratum_cdfs_delta_z,
            stratum_ppfs_delta_y,
            stratum_ppfs_delta_z,
            aggregate_indicator_train,
        )
        epsilon_y_B, epsilon_z_B = epsilon_B
        delta_y_B, delta_z_B = delta_B

        # step 4b) create our bootrapped dataset by adding the bootstrapped errors
        # (epsilon + delta) to our fitted values
        y_train_B = y_train_pred + (aggregate_indicator_train @ epsilon_y_B) + delta_y_B
        z_train_B = z_train_pred + (aggregate_indicator_train @ epsilon_z_B) + delta_z_B

        # step 5) refit the model
        #   we need to generate bootstrapped test predictions and to do that we need to fit a bootstrap model
        #   we are using the normal equations from the original model since x_train has stayed the same and the normal
        #       equations are only dependent on x_train. This saves compute.
        ols_y_B = OLSRegressionSolver()
        ols_y_B.fit(
            x_train,
            y_train_B,
            weights_train,
            normal_eqs=ols_y.normal_eqs,
            fit_intercept=True,
            regularize_intercept=False,
            n_feat_ignore_reg=1,
        )
        ols_z_B = OLSRegressionSolver()
        ols_z_B.fit(
            x_train,
            z_train_B,
            weights_train,
            normal_eqs=ols_z.normal_eqs,
            fit_intercept=True,
            regularize_intercept=False,
            n_feat_ignore_reg=1,
        )

        LOG.info("orig. ols coefficients, normalized margin: \n %s", ols_y.coefficients.flatten())
        LOG.info("boot. ols coefficients, normalized margin: \n %s", ols_y_B.coefficients.mean(axis=-1))

        # we cannot just apply ols_y_B/old_z_B to the test units because that would be missing
        # our contest level random effect
        # so we need to compute an bootstrapped estimate of the contest level random effect (epsilon)
        # to do that we first compute the bootstrapped leave-one-out training residuals and then use that to
        # estimate bootstrapped epsilons
        y_train_pred_B = ols_y_B.predict(x_train)
        z_train_pred_B = ols_z_B.predict(x_train)
        residuals_y_B = ols_y_B.residuals(y_train_B, y_train_pred_B, loo=True, center=True)
        residuals_z_B = ols_z_B.residuals(z_train_B, z_train_pred_B, loo=True, center=True)
        epsilon_y_hat_B = self._estimate_epsilon(residuals_y_B, aggregate_indicator_train)
        epsilon_z_hat_B = self._estimate_epsilon(residuals_z_B, aggregate_indicator_train)

        # We can then use our bootstrapped ols_y_B/ols_z_B and our bootstrapped contest level effect
        # (epsilon) to make bootstrapped predictions on our non-reporting units
        # This is \tilde{y_i}^{b} and \tilde{z_i}^{b}
        y_test_pred_B = (ols_y_B.predict(x_test) + (aggregate_indicator_test @ epsilon_y_hat_B)).clip(
            min=y_partial_reporting_lower, max=y_partial_reporting_upper
        )
        z_test_pred_B = (ols_z_B.predict(x_test) + (aggregate_indicator_test @ epsilon_z_hat_B)).clip(
            min=z_partial_reporting_lower, max=z_partial_reporting_upper
        )

        # \tilde{y_i}^{b} * \tilde{z_i}^{b}
        yz_test_pred_B = y_test_pred_B * z_test_pred_B

        # In order to generate our point prediction, we take the bootstrap mean.
        # this is \hat{y_i} and \hat{z_i}
        y_test_pred = y_test_pred_B.mean(axis=1).reshape(-1, 1)
        z_test_pred = z_test_pred_B.mean(axis=1).reshape(-1, 1)
        yz_test_pred = y_test_pred * z_test_pred

        # we now need to generate our bootstrapped "true" quantities (in order to subtract the
        # bootstrapped estimates from these quantities to get an estimate for our error)
        # In a normal bootstrap setting we would replace y and z with \hat{y} and \hat{z}
        # however, we want to produce prediction intervals (rather than just confidence intervals)
        # so we need to take into account the extra uncertainty that comes with prediction
        # (ie. estimating the mean has some variance, but sampling from our mean estimate
        # introduces an additional error)
        # To do that we need an estimate for our prediction error
        # (ie. an estimate of the residuals = epsilon + delta)
        # we cannot use our original residuals_y, residuals_z because
        # they would cancel out our error used in the bootstrap
        test_residuals_y, test_residuals_z = self._sample_test_errors(
            residuals_y,
            residuals_z,
            epsilon_y_hat,
            epsilon_z_hat,
            x_test_strata,
            stratum_ppfs_delta_y,
            stratum_ppfs_delta_z,
            aggregate_indicator_train,
            aggregate_indicator_test,
        )

        # multiply by weights to turn into unnormalized margin
        # w_i * \hat{y_i} * \hat{z_i}
        self.errors_B_1 = yz_test_pred_B * weights_test

        # This is (\hat{y_i} + \residuals_{y, i}^{b}) * (\hat{z_i} + \residuals_{z, i}^{b})
        # we clip them based on bounds we generated earlier. These are defined by the estimates amount of
        # outstanding vote from our election results provider
        errors_B_2 = (y_test_pred + test_residuals_y).clip(min=y_partial_reporting_lower, max=y_partial_reporting_upper)
        errors_B_2 *= (z_test_pred + test_residuals_z).clip(
            min=z_partial_reporting_lower, max=z_partial_reporting_upper
        )

        # multiply by weights turn into unnormalized margin
        # this is w_i * (\hat{y_i} + \residuals_{y, i}^{b}) * (\hat{z_i} + \residuals_{z, i}^{b})
        self.errors_B_2 = errors_B_2 * weights_test

        # we also need errors for the denominator of the aggregate
        # this is \tilde{z_i}^{b}
        self.errors_B_3 = z_test_pred_B * weights_test  # has already been clipped above
        # this is (\hat{z_i} + \residuals_{z, i}^{b})
        self.errors_B_4 = (z_test_pred + test_residuals_z).clip(
            min=z_partial_reporting_lower, max=z_partial_reporting_upper
        ) * weights_test

        # this is for the unit point prediction. turn into unnormalized margin
        self.weighted_yz_test_pred = yz_test_pred * weights_test
        # and turn into turnout estimate
        self.weighted_z_test_pred = z_test_pred * weights_test
        self.ran_bootstrap = True
        self.n_contests = aggregate_indicator.shape[1]

    def get_unit_predictions(
        self, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, estimand: str, **kwargs
    ) -> np.ndarray:
        """
        Returns the unit predictions, if necessary also generates them
        The unit predictions are the *unnormalized margin*
            w_i * hat{y_i} * hat{z_i}
        """
        # if bootstrap hasn't been run yet, run it
        if not self.ran_bootstrap:
            unexpected_units = kwargs["unexpected_units"]
            self.compute_bootstrap_errors(reporting_units, nonreporting_units, unexpected_units)
        return self.weighted_yz_test_pred, self.weighted_z_test_pred

    def _is_top_level_aggregate(self, aggregate: list) -> bool:
        """
        Function to figure out whether we are at the top level aggregation
        (ie. postal code for state level model or postal code, district for district model)
        """
        # case 1:
        # top level aggregate is postal code (ie. we are generating up to a state level -> ECV or Senate).
        # We know this is the case because aggregate length is just 1 and postal code is the only aggregate
        # case 2:
        # top level aggregate is postal code and district
        # (ie. we are generating up to a district level -> House or State Senate).
        # We know this is the case because aggregate length is 2 and postal code
        # and district are the two aggregates.
        return (len(aggregate) == 1 and "postal_code" in aggregate) or (
            len(aggregate) == 2 and "postal_code" in aggregate and "district" in aggregate
        )

    def _format_called_contests(
        self,
        lhs_called_contests: list,
        rhs_called_contests: list,
        contests: list,
        lhs_value: int | bool | None,
        rhs_value: int | bool | None,
        fill_value: int | bool,
    ) -> np.ndarray:
        """
        Create called contest numpy array
        """
        lhs_rhs_intersection = set(lhs_called_contests) & set(rhs_called_contests)
        if len(lhs_rhs_intersection) > 0:
            raise BootstrapElectionModelException(
                f"You can only call a contest for one party, not for both. Currently these contests are called for both parties: {lhs_rhs_intersection}"
            )

        lhs_difference_with_contests = set(lhs_called_contests) - set(contests)
        if len(lhs_difference_with_contests) > 0:
            raise BootstrapElectionModelException(
                f"You can only call contests that are being run by the model. These LHS called contests do not exist: {lhs_difference_with_contests}"
            )

        rhs_difference_with_contests = set(rhs_called_contests) - set(contests)
        if len(rhs_difference_with_contests) > 0:
            raise BootstrapElectionModelException(
                f"You can only call contests that are being run by the model. These RHS called contests do not exist: {rhs_difference_with_contests}"
            )

        # the order in called_coteests need
        called_contests = np.full(len(contests), fill_value)
        for i, contest in enumerate(contests):
            if contest in lhs_called_contests:
                called_contests[i] = lhs_value
            elif contest in rhs_called_contests:
                called_contests[i] = rhs_value

        return called_contests

    def _adjust_called_contests(self, to_call: np.array, called_contests: list) -> np.array:
        """
        This functions applies race calls to the point prediction
        """
        to_call_mod = to_call.copy()
        to_call_mod[np.isclose(called_contests, 1)] = np.maximum(
            self.lhs_called_threshold, to_call[np.isclose(called_contests, 1)]
        )
        to_call_mod[np.isclose(called_contests, 0)] = np.minimum(
            self.rhs_called_threshold, to_call[np.isclose(called_contests, 0)]
        )
        return to_call_mod

    def get_aggregate_predictions(
        self,
        reporting_units: pd.DataFrame,
        nonreporting_units: pd.DataFrame,
        unexpected_units: pd.DataFrame,
        aggregate: list,
        estimand: str,
        **kwargs: dict,
    ) -> pd.DataFrame:
        """
        Generates and returns the normalized margin for arbitrary aggregates
            sum_{i = 1}^N (w_i * hat{y_i} * hat{z_i}) / sum_{i = 1}^N (w_i * hat{z_i})
        """
        n_train = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]

        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)

        # if we want to aggregate to something that isn't postal_code we need to generate a temporary
        # column so that we create a dummary variable for each level of the aggregate
        # aggreagate_1 * aggregate_2 rather than aggregate_1 + aggregate_2 which is what would happen otherwise
        if len(aggregate) > 1:
            aggregate_temp_column_name = "-".join(aggregate)
            all_units[aggregate_temp_column_name] = all_units[aggregate].agg("_".join, axis=1)
            dummies = pd.get_dummies(all_units[aggregate_temp_column_name])
        else:
            # since aggregate is of length zero we can grab the first element
            dummies = pd.get_dummies(all_units[aggregate[0]])
            aggregate_temp_column_name = aggregate
        aggregate_indicator = dummies.values
        contests = dummies.columns

        # the unit level predictions that come in through reporting_units and nonreporting_units
        # are unnormalized. Since we want the normalized margin for the aggregate predictions
        # we need to divide the sum of unnormalized aggregates by the total turnout predictions
        # so we first compute the total turnout predictions

        aggregate_indicator_expected = aggregate_indicator[: (n_train + n_test)]  # noqa: 203
        aggregate_indicator_unexpected = aggregate_indicator[(n_train + n_test) :]  # noqa: 203

        # two party turnout
        # results weights is unexpected_units["results_dem"] + unexpected_units["results_gop"]
        turnout_unexpected = (unexpected_units["results_weights"]).values.reshape(-1, 1)

        aggregate_indicator_train = aggregate_indicator_expected[:n_train]
        aggregate_indicator_test = aggregate_indicator_expected[n_train:]
        weights_train = reporting_units["baseline_weights"].values.reshape(-1, 1)
        z_train = reporting_units["turnout_factor"].values.reshape(-1, 1)

        # get turnout for aggregate (w_i * z_i)
        aggregate_z_train = aggregate_indicator_train.T @ (weights_train * z_train)
        aggregate_z_unexpected = aggregate_indicator_unexpected.T @ turnout_unexpected

        # total turnout predictions
        aggregate_z_total = (
            aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        ).flatten()

        # use get_aggregate_predictions from BaseElectionModel to sum unnormalized margin of all the units
        raw_margin_df = super().get_aggregate_predictions(
            reporting_units, nonreporting_units, unexpected_units, aggregate, estimand
        )

        # divide the unnormalized margin and results by the total turnout predictions
        # to get the normalized margin for the aggregate
        # turnout prediction could be zero, in which case predicted margin is also zero,
        # so replace NaNs with zero in that case
        raw_margin_df["pred_margin"] = np.nan_to_num(raw_margin_df.pred_margin / aggregate_z_total).reshape(-1, 1)
        raw_margin_df["results_margin"] = np.nan_to_num(raw_margin_df.results_margin / aggregate_z_total)
        raw_margin_df["pred_turnout"] = aggregate_z_total
        # if we are in the top level prediction, then save the aggregated baseline margin,
        # which we will need for the national summary (e.g. ecv) model
        if self._is_top_level_aggregate(aggregate):
            lhs_called_contests = kwargs.get("lhs_called_contests", [])
            rhs_called_contests = kwargs.get("rhs_called_contests", [])
            called_contests = self._format_called_contests(lhs_called_contests, rhs_called_contests, contests, 1, 0, -1)

            self.aggregate_pred_margin = self._adjust_called_contests(
                raw_margin_df.pred_margin.values, called_contests
            ).reshape(-1, 1)
            raw_margin_df["pred_margin"] = self.aggregate_pred_margin

            aggregate_sum = all_units.groupby(aggregate_temp_column_name).sum()
            self.aggregate_baseline_margin = (
                (aggregate_sum.baseline_dem - aggregate_sum.baseline_gop) / (aggregate_sum.baseline_turnout + 1)
            ).values

        return raw_margin_df

    def _get_quantiles(self, alpha):
        """
        Given a confidence level for the prediction interval, calculates the quantiles
        necessary to be computed
        """
        lower_alpha = (1 - alpha) / 2
        upper_alpha = 1 - lower_alpha

        # adjust percentiles to account for bootstrap
        lower_q = np.floor(lower_alpha * (self.B + 1)) / self.B
        upper_q = np.ceil(upper_alpha * (self.B - 1)) / self.B

        return lower_q, upper_q

    def get_unit_prediction_intervals(
        self, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, alpha: float, estimand: str
    ) -> PredictionIntervals:
        """
        Generate and return unit level prediction intervals

        In the unit case, the error in our prediciton is:
                w_i * hat{y_i} * hat{z_i} - w_i * y_i * z_i
        In the bootstrap setting this has been estimated as:
                w_i * tilde{y_i}^{b} * tilde{z_i}^{b} - w_i *
               (hat{y_i} + residual_{y, i}^{b}) * (hat{z_i} + residual_{z, i}^{b})

        The alpha% prediction interval is the (1 - alpha) / 2 and (1 + alpha) / 2
        percentiles over the bootstrap samples of this quantity
        """
        # error_B_1: w_i * \tilde{y_i}^{b} * \tilde{z_i}^{b}
        # error_B_2: w_i * (\hat{y_i} + \residual_{y, i}^{b}) * (\hat{z_i} + \residual_{z, i}^{b})
        errors_B = self.errors_B_1 - self.errors_B_2

        lower_q, upper_q = self._get_quantiles(alpha)

        # sum in the prediction to our lower and upper esimate of the error in our prediction
        interval_upper, interval_lower = (
            self.weighted_yz_test_pred - np.quantile(errors_B, q=[lower_q, upper_q], axis=-1).T
        ).T

        interval_upper = interval_upper.reshape(-1, 1)
        interval_lower = interval_lower.reshape(-1, 1)

        return PredictionIntervals(interval_lower.round(decimals=0), interval_upper.round(decimals=0))

    def get_aggregate_prediction_intervals(
        self,
        reporting_units: pd.DataFrame,
        nonreporting_units: pd.DataFrame,
        unexpected_units: pd.DataFrame,
        aggregate: list,
        alpha: float,
        unit_prediction_intervals: PredictionIntervals,
        estimand: str,
        **kwargs: dict,
    ) -> PredictionIntervals:
        """
        Generate and return aggregate prediction intervals for arbitrary aggregates

        In the aggregate case, the error in our prediction is:
                (sum_{i = 1}^N w_i * hat{y_i} * hat{z_i}) / (sum_{i = 1}^n w_i hat{z_i}) -
                (sum_{i = 1}^N w_i * y_i * z_i) / (sum_{i = 1}^n w_i z_i)
        In the bootstrap setting this has been estimated as:
                (sum_{i = 1}^N w_i * tilde_{y_i}^b * tilde_{z_i}^b) /
                (sum_{i = 1}^N w_i * tilde_{z_i}^b) -
                (sum_{i = 1}^N w_i * hat_{y_i} + residual_{y, i}^b) *
                (hat{z_i} + residual_{z, i}^b)) /
                (sum_{i = 1}^N w_i * (hat{z_i} + residual_{z, i}^b))

        The alpha% prediction interval is the (1 - alpha) / 2 and (1 + alpha) / 2 percentiles
        over the bootstrap samples of this quantity
        """
        n_train = reporting_units.shape[0]
        n_test = nonreporting_units.shape[0]

        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)

        if len(aggregate) > 1:
            aggregate_temp_column_name = "-".join(aggregate)
            all_units[aggregate_temp_column_name] = all_units[aggregate].agg("_".join, axis=1)
            dummies = pd.get_dummies(all_units[aggregate_temp_column_name])
        else:
            # since aggregate is of length one, we can grab the first element
            dummies = pd.get_dummies(all_units[aggregate[0]])

        aggregate_indicator = dummies.values
        contests = dummies.columns
        aggregate_indicator_expected = aggregate_indicator[: (n_train + n_test)]

        # first compute turnout and unnormalized margin for unexpected units.
        # this is a known quantity
        aggregate_indicator_unexpected = aggregate_indicator[(n_train + n_test) :]  # noqa: 1185
        margin_unexpected = unexpected_units["results_margin"].values.reshape(-1, 1)
        # results weights is unexpected_units["results_dem"] + unexpected_units["results_gop"]
        turnout_unexpected = (unexpected_units["results_weights"]).values.reshape(-1, 1)
        aggregate_z_unexpected = aggregate_indicator_unexpected.T @ turnout_unexpected
        aggregate_yz_unexpected = aggregate_indicator_unexpected.T @ margin_unexpected

        aggregate_indicator_train = aggregate_indicator_expected[:n_train]
        aggregate_indicator_test = aggregate_indicator_expected[n_train:]
        weights_train = reporting_units["baseline_weights"].values.reshape(-1, 1)

        # compute turnout and unnormalized margin for reporting units.
        # this is also a known quantity with no uncertainty
        y_train = reporting_units["results_normalized_margin"].values.reshape(-1, 1)
        z_train = reporting_units["turnout_factor"].values.reshape(-1, 1)
        yz_train = y_train * z_train
        aggregate_z_train = aggregate_indicator_train.T @ (weights_train * z_train)
        aggregate_yz_train = aggregate_indicator_train.T @ (weights_train * yz_train)

        # (sum_{i = 1}^N w_i * \tilde_{y_i}^b * \tilde_{z_i}^b)
        aggregate_yz_test_B = aggregate_indicator_test.T @ self.errors_B_1

        # (\sum_{i = 1}^N w_i * (\hat_{y_i} + \residual_{y, i}^b) * (\hat{z_i} + \residual_{z, i}^b))
        aggregate_yz_test_pred = aggregate_indicator_test.T @ self.errors_B_2

        # (\sum_{i = 1}^N w_i * \tilde_{z_i}^b)
        aggregate_z_test_B = aggregate_indicator_test.T @ self.errors_B_3

        # (\sum_{i = 1}^N w_i * (\hat{z_i} + \residual_{z, i}^b))
        aggregate_z_test_pred = aggregate_indicator_test.T @ self.errors_B_4

        # sum the aggregate error components with the known quantities from reporting and unexpected units
        aggregate_yz_total_B = aggregate_yz_train + aggregate_yz_test_B + aggregate_yz_unexpected
        aggregate_yz_total_pred = aggregate_yz_train + aggregate_yz_test_pred + aggregate_yz_unexpected
        aggregate_z_total_B = aggregate_z_train + aggregate_z_test_B + aggregate_z_unexpected
        aggregate_z_total_pred = aggregate_z_train + aggregate_z_test_pred + aggregate_z_unexpected

        aggregate_error_B_1 = aggregate_yz_total_B
        aggregate_error_B_2 = aggregate_yz_total_pred
        aggregate_error_B_3 = aggregate_z_total_B
        aggregate_error_B_4 = aggregate_z_total_pred

        # (sum_{i = 1}^N w_i * \tilde_{y_i}^b * \tilde_{z_i}^b) /  (\sum_{i = 1}^N w_i * \tilde_{z_i}^b)
        divided_error_B_1 = np.nan_to_num(aggregate_error_B_1 / aggregate_error_B_3)

        # (\sum_{i = 1}^N w_i * (\hat_{y_i} + \residual_{y, i}^b) *
        # (\hat{z_i} + \residual_{z, i}^b)) /
        # (\sum_{i = 1}^N w_i * (\hat{z_i} + \residual_{z, i}^b))
        divided_error_B_2 = np.nan_to_num(aggregate_error_B_2 / aggregate_error_B_4)

        # we also need to re-compute our aggregate prediction to add to our error to get the prediction interval
        # first the turnout component
        aggregate_z_total = (
            aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        )
        # then the unnormalized margin component
        aggregate_yz_total = (
            aggregate_yz_unexpected + aggregate_yz_train + aggregate_indicator_test.T @ self.weighted_yz_test_pred
        )
        # calculate normalized margin in the aggregate prediction
        # turnout prediction could be zero, so convert NaN -> 0
        aggregate_perc_margin_total = np.nan_to_num(aggregate_yz_total / aggregate_z_total).reshape(-1, 1)

        lower_q, upper_q = self._get_quantiles(alpha)

        error_diff = divided_error_B_1 - divided_error_B_2

        # saves the aggregate errors in case we want to generate somem form of national predictions (like ecv)
        if self._is_top_level_aggregate(aggregate):
            aggregate_perc_margin_total = self.aggregate_pred_margin

            lhs_called_contests = kwargs.get("lhs_called_contests", [])
            rhs_called_contests = kwargs.get("rhs_called_contests", [])
            called_contests = self._format_called_contests(lhs_called_contests, rhs_called_contests, contests, 1, 0, -1)

            stop_model_call = kwargs.get("stop_model_call", [])
            stop_model_call = self._format_called_contests(stop_model_call, [], contests, True, None, False)

            interval_upper, interval_lower = (
                aggregate_perc_margin_total - np.quantile(error_diff, q=[lower_q, upper_q], axis=-1).T
            ).T

            for i, (contest, call, stop_model_call_i) in enumerate(zip(contests, called_contests, stop_model_call)):
                interval_lower_i = interval_lower[i]
                interval_upper_i = interval_upper[i]

                if stop_model_call_i:
                    # if we don't allow the model call, then force the lower interval to be below zero and the upper interval to be above zero
                    if interval_lower_i > 0:
                        # if interval_lower_i > 0 then our model thinks the race is called for the LHS party.
                        # error_diff > 0 means that the lower bound is smaller than the prediction, so for those we set error_diff to be the gap between
                        # the prediction and the imposed lower bound. This forces the difference between error_diff and the prediction to be exactly the imposed
                        # lower bound
                        error_diff[i, error_diff[i] > 0] = (
                            aggregate_perc_margin_total[i] - self.rhs_called_threshold
                        ).flatten()
                        # we are pushing divided_error_B_2 such that divided_error_B_2 is less than zero in 10% of cases (ie. that LHS party wins some simulations)
                        # there will be a chance that this impacts our electoral college lower bound
                        divided_error_B_2_quantile = np.quantile(divided_error_B_2[i, :], q=0.1)
                        if divided_error_B_2_quantile > 0:
                            divided_error_B_2[i, :] += self.rhs_called_threshold - divided_error_B_2_quantile
                    if interval_upper_i < 0:
                        # if interval_upper_i < 0 then our model thinks the race has been called for the RHS party.
                        # error_diff < 0 means that the upper bound is larger than the prediction, so for those we set error_diff to be the gap between the prediction
                        # and the imposed upper bound. This forces the difference between error_diff and the prediction to be exactly the imposed upper bound
                        error_diff[i, error_diff[i] < 0] = (
                            aggregate_perc_margin_total[i] - self.lhs_called_threshold
                        ).flatten()
                        # we are pushing divided error_B_2 such that divided_error_B_2 is greater than zero in 10% of cases (ie. that LHS party wins some simulations)
                        # there will be some chance that this impacts our electoral college prediction upper bound
                        divided_error_B_2_quantile = np.quantile(divided_error_B_2[i, :], q=0.9)
                        if divided_error_B_2_quantile < 0:
                            divided_error_B_2[i, :] += self.lhs_called_threshold - divided_error_B_2_quantile

                if np.isclose(call, 1):
                    if interval_lower_i < 0:
                        # if a contest has been called for the LHS party but the interval_lower is below zero (ie. our model does not think that this is called)
                        # error_diff > 0 means that lower bound is smaller than the prediction, so for those we set the error_diff to be the gap between the prediction
                        # and the imposed lower bound. This forces the difference between the error_diff and the prediction to be exactly the imposed lower bound
                        error_diff[i, error_diff[i] > 0] = (
                            aggregate_perc_margin_total[i] - self.lhs_called_threshold
                        ).flatten()
                    # for error_B_1 and error_B_2 we can set all of them to the imposed lower bound, because we no longer care about doing inference on the intervals
                    # ie. the winner is fixed now so we no longer care about doing inference on the margin
                    # NOTE: we cannot do this for error_diff because we still want the upper bound in case to be what it would be without the race call
                    divided_error_B_1[i, :] = self.lhs_called_threshold
                    divided_error_B_2[i, :] = self.lhs_called_threshold
                elif np.isclose(call, 0):
                    if interval_upper_i > 0:
                        # if a contest has been called for the RHS party but the interval_upper is larger than zero (ie. our model does not think this is called)
                        # error_diff < 0 means that the upper bound is larger than the prediction, so for those we set error_diff to be the gap between the prediction
                        # and the imposed upper bound. This forces the difference between the error diff and the prediction to be the imposed upper bound
                        error_diff[i, error_diff[i] < 0] = (
                            self.rhs_called_threshold - aggregate_perc_margin_total[i]
                        ).flatten()
                    divided_error_B_1[i, :] = self.rhs_called_threshold
                    divided_error_B_2[i, :] = self.rhs_called_threshold

            self.divided_error_B_1 = divided_error_B_1
            self.divided_error_B_2 = divided_error_B_2

        interval_upper, interval_lower = (
            aggregate_perc_margin_total - np.quantile(error_diff, q=[lower_q, upper_q], axis=-1).T
        ).T

        interval_upper = interval_upper.reshape(-1, 1)
        interval_lower = interval_lower.reshape(-1, 1)

        return PredictionIntervals(interval_lower, interval_upper)

    def get_national_summary_estimates(self, nat_sum_data_dict: dict, base_to_add: int | float, alpha: float) -> list:
        """
        Generates and returns a national summary estimate (ie. electoral votes or total number of senate seats).
        This function does both the point prediction and the lower and upper estimates.

        First element in the list is the prediction, second is the lower end of the interval,
        and third is the upper end of the interval.

        The point prediction and prediction intervals are very similar to
        get_aggregate_prediction / get_aggregate_prediction_intervals
        except that we pass our bootstrapped preditions (and our stand-in for the "true" value)
        through a sigmoid (or a threshold) and assign weights. This creates gives us bootstrapped
        national summary estimate (e.g. electoral votes), which we can use to generate
        a prediction interval.
        """
        # if nat_sum_data_dict is None then we assign 1 for every contest (ie. Senate or House)
        if nat_sum_data_dict is None:
            # the order does not matter since all contests have the same weight,
            # so we can use anything as the key when sorting
            nat_sum_data_dict = {i: 1 for i in range(self.divided_error_B_1.shape[0])}

        # if we didn't pass the right number of national summary weights
        # (ie. the number of contests) then raise an exception
        if len(nat_sum_data_dict) != self.divided_error_B_1.shape[0]:
            raise BootstrapElectionModelException(
                f"nat_sum_data_dict is of length {len(nat_sum_data_dict)} but there are {self.divided_error_B_1.shape[0]} contests"
            )

        # NOTE: This assumes that pd.get_dummies does alphabetical ordering
        # sort in order to get in the same order as the contests,
        # which have been sorted when getting dummies for aggregate indicators
        # in get_aggregate_prediction_intervals
        nat_sum_data_dict_sorted = sorted(nat_sum_data_dict.items())
        nat_sum_data_dict_sorted_vals = np.asarray([x[1] for x in nat_sum_data_dict_sorted]).reshape(-1, 1)

        if self.hard_threshold:
            aggregate_dem_prob_B_1 = self.divided_error_B_1 > 0
            aggregate_dem_prob_B_2 = self.divided_error_B_2 > 0
        else:
            aggregate_dem_prob_B_1 = expit(self.T * self.divided_error_B_1)
            aggregate_dem_prob_B_2 = expit(self.T * self.divided_error_B_2)

        # multiply by weights of each contest
        aggregate_dem_vals_B_1 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_1
        aggregate_dem_vals_B_2 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_2

        # calculate the error in our national aggregate prediction
        aggregate_dem_vals_B = np.sum(aggregate_dem_vals_B_1, axis=0) - np.sum(aggregate_dem_vals_B_2, axis=0)

        # we also need a national aggregate point prediction
        if self.hard_threshold:
            aggregate_dem_probs_total = self.aggregate_pred_margin > 0.5
        else:
            aggregate_dem_probs_total = expit(self.T * self.aggregate_pred_margin)

        aggregate_dem_vals_pred = np.sum(nat_sum_data_dict_sorted_vals * aggregate_dem_probs_total)

        lower_q, upper_q = self._get_quantiles(alpha)

        interval_upper, interval_lower = (
            aggregate_dem_vals_pred - np.quantile(aggregate_dem_vals_B, q=[lower_q, upper_q], axis=-1).T
        ).T

        # B_1 and B_2 outcomes should respect called races
        # because we create independent samples for B_1 and B_2 their difference can exaggerate the possible outcomes
        # in the predicted lower and upper bounds.
        # to undo this, we take the lower bound for B_1 and B_2 and the upper bound for B_1 and B_2 to max/min those
        # with the predicted lower and upper bounds.
        lower_bound = min(aggregate_dem_vals_B_1.sum(axis=0).min(), aggregate_dem_vals_B_2.sum(axis=0).min())
        upper_bound = max(aggregate_dem_vals_B_1.sum(axis=0).max(), aggregate_dem_vals_B_2.sum(axis=0).max())

        agg_pred = round(aggregate_dem_vals_pred + base_to_add, 2)
        agg_lower = round(max(interval_lower, lower_bound) + base_to_add, 2)
        agg_upper = round(min(interval_upper, upper_bound) + base_to_add, 2)
        national_summary_estimates = {"margin": [agg_pred, agg_lower, agg_upper]}
        return national_summary_estimates
