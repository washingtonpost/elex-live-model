from __future__ import annotations  # pylint: disable=too-many-lines

import logging

import numpy as np
import pandas as pd
# from conditionalconformal import CondConf
from elexsolver.OLSRegressionSolver import OLSRegressionSolver
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver
from elexsolver.LinearSolver import LinearSolver
from scipy.special import expit

from elexmodel.handlers.data.Featurizer import Featurizer
from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals

LOG = logging.getLogger(__name__)

UNIT_FIXED_EFFECTS = {'county_classification'}

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
        y_i = f_y(x) + residuals_y(x)
        z_i = f_z(x) + residuals_z(x)
    where f_y(x) and f_z(x) are the model. We assume that residuals_y and residuals_z are correlated
    """

    def __init__(self, model_settings={}):
        super().__init__(model_settings)
        self.B = model_settings.get("B", 200)  # number of bootstrap samples
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
        self.model_type = model_settings.get("model_type", 'OLS') # TODO: pass into model as parameter
        self.ran_bootstrap = False

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
                ols_kwargs = {
                    'lambda_': lambda_,
                    'fit_intercept': True,
                    'regularize_intercept': False,
                    'n_feat_ignore_reg': 1
                }
                ols_lambda.fit(
                    x_train,
                    y_train,
                    weights=w_train,
                    **ols_kwargs
                )
                y_hat_lambda = ols_lambda.predict(x_test)
                # error is the weighted sum of squares of the residual between
                # the actual heldout y and the predicted y on the heldout set
                errors[i] += np.sum(
                    w_test * ols_lambda.residuals(x_test, y_test, w_test, K=None, center=False, **ols_kwargs) ** 2
                ) / np.sum(w_test)
        # return lambda that minimizes the k-fold error
        # np.argmin returns the first occurence if multiple minimum values
        return lambdas_[np.argmin(errors)]

    def get_minimum_reporting_units(self, alpha: float) -> int:
        # arbitrary, just enough to fit coefficients
        return 10

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

    def _bootstrap_errors(
        self,
        residuals_y: np.ndarray,
        residuals_z: np.ndarray,
        residuals_y_cdf: dict,
        residuals_z_cdf: dict,
        residuals_y_ppf: dict,
        residuals_z_ppf: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap the errors of our model (residuals)
        """

        n_train = residuals_y.shape[0]

        # convert residuals into uniform random variables
        unifs_y = np.asarray([residuals_y_cdf[i](r) for i, r in enumerate(residuals_y)]).flatten()
        unifs_z = np.asarray([residuals_z_cdf[i](r) for i, r in enumerate(residuals_z)]).flatten()

        # re-sample B bootstrap errors
        # TODO: we currently sample y and z independently, which loses potential correlation
        unifs_y_B = self.rng.choice(unifs_y, (n_train, self.B), replace=True)
        unifs_z_B = self.rng.choice(unifs_z, (n_train, self.B), replace=True)

        residuals_y_B = np.zeros((n_train, self.B))
        residuals_z_B = np.zeros((n_train, self.B))

        # probability integral transform: revert the sampled uniforms back
        # into residual space
        for i, (unif_y_B, unif_z_B) in enumerate(zip(unifs_y_B, unifs_z_B)):
            residuals_y_B[i] = residuals_y_ppf[i](unif_y_B)
            residuals_z_B[i] = residuals_z_ppf[i](unif_z_B)

        return residuals_y_B, residuals_z_B

    def _sample_test_errors(
        self,
        residual_ppfs_y: dict,
        residual_ppfs_z: dict,
        n_train: int,
        n_test: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function samples new test errors for our model (ie. new test residuals)
        """
        # sample a new set of uniforms
        # TODO: this currently also re-samples uniforms independently
        test_unifs_y = self.rng.uniform(low=0, high=1, size=(n_test, self.B))
        test_unifs_z = self.rng.uniform(low=0, high=1, size=(n_test, self.B))

        test_residuals_y = np.zeros((n_test, self.B))
        test_residuals_z = np.zeros((n_test, self.B))

        # PIT: convert uniforms back into residual space
        for i, (test_unif_y, test_unif_z) in enumerate(zip(test_unifs_y, test_unifs_z)):
            # this is at [n_train + i] since residual_ppfs_y/z starts includes reporting
            # units which take on the first n_train elements
            test_residuals_y[i] = residual_ppfs_y[n_train + i](test_unif_y)
            test_residuals_z[i] = residual_ppfs_z[n_train + i](test_unif_z)
        return test_residuals_y, test_residuals_z

    def fit_model(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, **model_kwargs) -> LinearSolver:
        """
        Fit the actual models This is abstracted away for us to more easily test other kinds of models
        """

        if self.model_type == 'OLS':
            model = OLSRegressionSolver()
            model.fit(
                x,
                y,
                weights=weights,
                **model_kwargs
            )

        elif self.model_type == 'QR':
            model = QuantileRegressionSolver()
            model.fit(
                x,
                y,
                weights=weights,
                taus=0.5,
                **model_kwargs
            )

        return model

    def _get_residual_features(
        self, 
        reporting_units: pd.DataFrame, 
        nonreporting_units: pd.DataFrame, 
        aggregates: list, 
        agg_features: list,
        unit_features: list
    ) -> np.ndarray:
        """
        Generate features for fitting the residual model. The residual model is used to estimate
        the residual distribution. The features should be at each level of aggregation
        e.g. state level covariates, district level covariates and unit level covariates
        """
        # As an example, the residuals might be correlated within the same state. Ideally, the
        # state level covariates would help explain that and be used in the regression.

        # first we have to separate out the fixed effects and the feature. A feature is actually a fixed
        # effect if it is an aggregate or if it is in a pre-determined list of UNIT_FIXED_EFFECTS
        # TODO: CAN WE NOT JUST CREATE TWO LISTS HERE TO PASS INTO THIS FUNCTION, ONE BEING FEATURE AND ONE BEING FE

        # fixed effects are the combined aggregate and unit fixed effects
        fixed_effects_agg = [feat for feat in agg_features if feat in aggregates]
        fixed_effects_unit = [feat for feat in unit_features if feat in UNIT_FIXED_EFFECTS]
        fixed_effects = fixed_effects_agg + fixed_effects_unit
        
        # features are the combined aggregate and unit level features
        features_agg_og_name = [feat for feat in agg_features if feat not in aggregates] # features original name
        features_agg_placeholder = [feat + "_agg" for feat in features_agg_og_name] # names used in the unit DF so not overwrite original names
        features_agg = [
            [feat + f'_{agg}' for feat in features_agg_og_name] # feature name are things like "gender_f_postal_code" to show what the feature has been aggregated up to
            for agg in aggregates
        ]
        features_unit = [
            feat for feat in unit_features if feat not in UNIT_FIXED_EFFECTS
        ]
        features = sum(features_agg, []) + features_unit # sum trick combines a list of lists into one list
        
        residual_featurizer = Featurizer(features, fixed_effects)
        all_units = pd.concat([reporting_units, nonreporting_units], axis=0)

        # aggregate features is the population weighted average of the unit features within each aggregate
        if len(features_agg) > 0:
            # multiply by population weight
            all_units[features_agg_placeholder] = all_units[features_agg_og_name].multiply(all_units.baseline_turnout, axis=0)
            for agg, feats_agg in zip(aggregates, features_agg): 
                # for each feature and aggregate sum and divide by population weight to get weighted average
                agg_features = all_units.groupby(agg).sum().reset_index(drop=False)
                agg_features[feats_agg] = agg_features[features_agg_placeholder].div(agg_features.baseline_turnout, axis=0)
                all_units = all_units.merge(agg_features[[agg] + feats_agg], on=agg, how='left')
            all_units.drop(columns=features_agg_placeholder, inplace=True)

        resid_features = residual_featurizer.prepare_data(
            all_units, center_features=False, scale_features=False, add_intercept=self.add_intercept
        )

        n_train = reporting_units.shape[0]

        resid_features_reporting = residual_featurizer.filter_to_active_features(resid_features[:n_train])
        resid_features_nonreporting = residual_featurizer.generate_holdout_data(resid_features[n_train:])
        # resid_features = pd.concat([resid_features_reporting, resid_features_nonreporting], axis=0)
        return resid_features_reporting, resid_features_nonreporting


    def _get_residual_mask(
        self, features_agg_indic: pd.DataFrame, features_agg_feat: pd.DataFrame, aggregates: list
    ) -> pd.DataFrame:
        """
        Get a dataframe which tells us which units are from aggregates that are unobserved and which
        ones are observed. If a the element in the dataframe in True that means it's unobserved. 
        """

        # get all sets of aggregate indicator columns
        # e.g. [[postal_code_CA, postal_code_VA], [postal_code-district_CA-6, postal_code-district_CA-8]]
        indic_columns_list = [
            features_agg_indic.columns[[c.startswith(agg) for c in features_agg_indic.columns]]
            for agg in aggregates
        ]
        # get all seats of feature columns 
        feat_columns_list = [
            features_agg_feat.columns[[c.endswith(agg) for c in features_agg_feat.columns]]
            for agg in aggregates
        ]

        # TODO: this isn't exactly what we want but okay for now
        # we should only use the features one aggregate at a time (i.e., if we're in an unobserved
        # district, but observed state), we still want to use the state indicator...

        unobs_mask = np.zeros((features_agg_indic.shape[0],)).astype(bool)
        for indic_cols, feat_cols in zip(indic_columns_list, feat_columns_list):
            indic_agg = features_agg_indic[indic_cols]
            unobs_mask |= (~np.isclose(indic_agg, 0) & ~np.isclose(indic_agg, 1)).any(axis=1)
            # swap out indicators for aggregate features when it's not observed

        return unobs_mask

    def _estimate_residual_dist(
        self, residuals, features_agg_indic, features_agg_feat, features_mask,
        taus, randomize, lb, ub
    ) -> Tuple[dict, dict]:
        if residuals.ndim > 1 and residuals.shape[1] > 1:
            residuals_ppf = []
            residuals_cdf = []
            for r_b in residuals.T:
                r_ppf, r_cdf = self._estimate_residual_dist(
                    r_b, features_agg_indic, features_agg_feat, features_mask, taus, randomize, lb, ub
                )
                residuals_ppf.append(r_ppf)
                residuals_cdf.append(r_cdf)

            print("Done fitting residual distributions!")
            return residuals_ppf, residuals_cdf

        def ppf_creator(quantiles: np.ndarray, taus: np.ndarray, lb: float, ub: float) -> float:
            """
            Creates a probability point function (inverse of a cumulative distribution function -- CDF)
            Given a percentile, provides the value of the CDF at that point
            """
            # we interpolate, because we want to return smooth betas
            return lambda p: np.interp(p, taus, quantiles, lb, ub)

        def cdf_creator(quantiles: np.ndarray, taus: np.ndarray) -> float:
            """
            Creates a cumulative distribution function (CDF)
            Provides the probability that a value is at most x
            """
            # interpolates because we want to provide smooth probabilites
            return lambda x: np.interp(x, quantiles, taus, right=1)

        features_agg_indic_concat = pd.concat(features_agg_indic, axis=0).to_numpy()
        features_agg_feat_concat = pd.concat(features_agg_feat, axis=0).to_numpy()

        x_train_indic = features_agg_indic[0].to_numpy()
        x_train_feat = features_agg_feat[0].to_numpy()

        features_cache = {} # maps feature to index at which it was first observed
        residuals_ppf = {} # maps from unit index to ppf function; output 1
        residuals_cdf = {} # maps from unit index to cdf function; output 2

        taus = np.asarray(taus)
        taus_lower, taus_upper = taus[taus < 0.5], taus[taus >= 0.5]

        for idx in range(len(features_mask)):
            if features_mask[idx]:
                x_train = x_train_feat
                x_test = features_agg_feat_concat[idx]
            else:
                x_train = x_train_indic
                x_test = features_agg_indic_concat[idx]
            
            if tuple(x_test) in features_cache: # if we've already seen this before...
                residuals_ppf[idx] = residuals_ppf[features_cache[tuple(x_test)]]
                residuals_cdf[idx] = residuals_cdf[features_cache[tuple(x_test)]]
                continue
            else:
                features_cache[tuple(x_test)] = idx

            x_train_aug = np.concatenate([x_train, x_test.reshape(1, -1)], axis=0)

            residuals_aug_lb = np.concatenate([residuals.flatten(), [lb]])
            residuals_aug_ub = np.concatenate([residuals.flatten(), [ub]])

            # fit the regressions to create the probability distributions
            # for a single regression beta[i] is the tau-th (e.g. median or 30th percentile)
            # for where dummy variable position i is equal to 1
            # since we are fitting many quantile regressions at the same time, our beta is
            # beta[tau, i] where tau stretches from 0.01 to 0.99
            betas = np.zeros((len(taus), x_train_aug.shape[1]))
            import pdb; pdb.set_trace()
            if len(taus_lower) > 0:
                qr_lower = QuantileRegressionSolver()
                qr_lower.fit(x_train_aug, residuals_aug_lb, taus_lower, fit_intercept=False)
                betas[0:len(taus_lower)] = qr_lower.coefficients.T

            if len(taus_upper) > 0:
                qr_upper = QuantileRegressionSolver()
                qr_upper.fit(x_train_aug, residuals_aug_ub, taus_upper, fit_intercept=False)
                betas[len(taus_lower):] = qr_upper.coefficients.T

            quantiles = betas @ x_test

            # condconf = CondConf(score_fn=lambda x, y: y, Phi_fn=lambda x: x)
            # condconf.setup_problem(x_train, residuals)
            # quantiles = np.zeros(len(taus))
            # import IPython; IPython.embed()
            # for i, tau in enumerate(taus): # TODO: something is wrong here....
            #     q = condconf.predict(tau, x_test, lambda c, x: c, randomize=randomize)
            #     quantiles[i] = np.clip(q, lb, ub)

            quantiles = np.sort(quantiles) # if there's quantile crossing just resort...

            # for this stratum value create ppf
            # we want the lower bounds and upper bounds to be the actual lower and upper values taken from beta
            residuals_ppf[idx] = ppf_creator(
                quantiles, taus, np.min(quantiles), np.max(quantiles)
            )

            # for this stratum value create cdf
            residuals_cdf[idx] = cdf_creator(quantiles, taus)
        
        return residuals_ppf, residuals_cdf

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
                y_i = f_y(x) + residual_y(x)
                z_i = f_z(x) + residual_z(x)
        this means that:
                hat{y_i} = hat{f_y(x)} + hat{residual_y(x)}
                hat{z_i} = hat{f_z(x)} + hat{residual_z(x)}

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
        and store those so that we can later compute prediction intervals for any function of these quantities in their
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
                w_i * z_i   ->      w_i * (hat{z_i} + residual{z, i}^{b})
                w_i * hat{z_i}     ->      w_i * tilde{z_i}^{b}
        """
        # prepare data (generate fixed effects, add intercept etc.)
        all_units = pd.concat([reporting_units, nonreporting_units, unexpected_units], axis=0)

        # TODO: this should not be hard-coded
        all_features = [
            'age_le_30',
            'age_geq_30_le_45',
            'age_geq_45_le_65',
            'ethnicity_east_and_south_asian',
            'ethnicity_european',
            'ethnicity_hispanic_and_portuguese',
            'ethnicity_likely_african_american',
            'gender_f',
            'median_household_income',
            'percent_bachelor_or_higher',
            'baseline_normalized_margin'
        ]
        self.featurizer.select_LASSO_features(reporting_units, all_features)
        print(f"Features selected by the LASSO are: {self.featurizer.features}")
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
        # optimal_lambda_y = self.cv_lambda(x_train, y_train, np.logspace(-3, 2, 20), weights=weights_train)
        # optimal_lambda_z = self.cv_lambda(x_train, z_train, np.logspace(-3, 2, 20), weights=weights_train)

        # step 1) fit the initial model
        # we don't want to regularize the intercept or the coefficient for baseline_normalized_margin
        model_kwargs = {
            'fit_intercept': True,
            'regularize_intercept': False,
            'n_feat_ignore_reg': 1
        }
        if self.model_type == 'OLS':
            # if we run OLS we use cross validation to find an optimal lambda
            # TODO: WHY DO WE NOT DO THIS FOR OTHER MODELS ALSO (E.G QR)?
            optimal_lambda_y = self.cv_lambda(x_train, y_train, np.logspace(-3, 2, 20), weights=weights_train)
            optimal_lambda_z = self.cv_lambda(x_train, z_train, np.logspace(-3, 2, 20), weights=weights_train)

            model_kwargs_y = {'lambda_': optimal_lambda_y, **model_kwargs}
            model_kwargs_z = {'lambda_': optimal_lambda_y, **model_kwargs}
        else:
            model_kwargs_y = model_kwargs
            model_kwargs_z = model_kwargs
        
        model_y = self.fit_model(x_train, y_train, weights_train, **model_kwargs_y)
        model_z = self.fit_model(x_train, z_train, weights_train, **model_kwargs_z)

        # step 2) calculate the fitted values
        y_train_pred = model_y.predict(x_train)
        z_train_pred = model_z.predict(x_train)

        # step 3) calculate residuals and their distribution
        #   residuals_{y/z} are the residuals from the model (ie. the amount of error that our regression does not explain)
        #       they are out-of-sample residuals (obtained by computing hold-out residuals in K-fold CV)
        #       K = n is feasible if model is OLS, otherwise, choose K = 5 (or something sensible)
        residuals_y = model_y.residuals(x_train, y_train, weights_train, K=5, center=False, **model_kwargs_y)
        residuals_z = model_z.residuals(x_train, z_train, weights_train, K=5, center=False, **model_kwargs_y)

        # There are two different models which we use for estimating the distribution of the residuals.
        # 1) Indicator model
        #       For units that are in aggregates that have been observed (ie. a county is in a state for which we have observed
        #       units, whether or not this specific unit is observed or not) our model is indicator based. Ie. We have a model
        #       that has an effect for that aggregate, which we were able to learn and then we apply it.
        # 2) Feature model
        #       For units that are in aggregates that have not been observed (ie. a county in a state for which we have *not*
        #       observed units. So clearly this unit is unobserved also) our model is feature based. Ie. We have tried to learn
        #       the relationship between the aggregate's features (ie. education in a state) and the residual and we then apply 
        #       that to other units in unobserved aggregates.
        # For this reason there are also two resid_features, one for each model.

        #   resid_features is the set of features we are using to calibrate the residual distribution
        #       it should include features at each level of aggregation that might plausibly affect the residual
        #       e.g., state-level covariates, district-level covariates, unit-level covariates
        #       if we are fitting aggregate features, by default we will use the same features we used to fit the 
        #       original unit-level prediction model (but aggregated up)
        resid_aggregates = ["postal_code-district", "postal_code"] if self.district_election else ["postal_code"]
        resid_features_agg_indic = self._get_residual_features(
            reporting_units, nonreporting_units, aggregates=resid_aggregates, agg_features=resid_aggregates,
            unit_features=["county_classification"]
        ) 
        resid_features_agg_feat = self._get_residual_features(
            reporting_units, nonreporting_units, aggregates=resid_aggregates, agg_features=self.featurizer.features,
            unit_features=["county_classification"]
        ) 

        # we need a mask which tells us which model applies to which unit. NOTE: we only need this mask for unobserved
        # units (ie. resid_features_agg_indic[1] not resid_features_agg_indic[0] because all units that are observed
        # clearly also have observed aggregates).
        resid_mask = self._get_residual_mask(
            resid_features_agg_indic[1], resid_features_agg_feat[1], aggregates=resid_aggregates
        )
        # all units that have been observed (resid_features_agg_indic[0]) clearly have observed aggregates so
        # the residual mask is false for those. We then concatenate with resid_mask to create the resid_mask
        # for all features
        resid_mask = np.concatenate((np.full(resid_features_agg_indic[0].shape[0], False), resid_mask))

        # we now fit the distribution (cdf/pdf) for the residuals. This allows us to move the residual into
        # the uniform space, where bootstrapping them and sampling new ones is easier. 

        #   residuals_{y/z}_ppf, residuals_{y/z}_cdf are dictionaries that map each training and test unit 
        #   (keys: geographic_unit_fips?) to the estimated CDF and PPF of the residual distribution in that unit
        residuals_y_ppf, residuals_y_cdf = self._estimate_residual_dist(
            residuals_y, resid_features_agg_indic, resid_features_agg_feat, resid_mask,
            taus=self.taus, randomize=False, lb=self.y_LB, ub=self.y_UB
        )

        residuals_z_ppf, residuals_z_cdf = self._estimate_residual_dist(
            residuals_z, resid_features_agg_indic, resid_features_agg_feat, resid_mask,
            taus=self.taus, randomize=False, lb=self.z_LB, ub=self.z_UB
        )

        # step 4) bootstrap resampling
        # step 4a) we resample B new residuals
        residuals_B = self._bootstrap_errors(
            residuals_y,
            residuals_z,
            residuals_y_ppf,
            residuals_z_ppf,
            residuals_y_cdf,
            residuals_z_cdf
        )
        residuals_y_B, residuals_z_B = residuals_B

        # step 4b) create our bootrapped dataset by adding the bootstrapped errors
        # (epsilon + delta) to our fitted values
        y_train_B = y_train_pred + residuals_y_B
        z_train_B = z_train_pred + residuals_z_B

        # step 5) refit the model
        #   we need to generate bootstrapped test predictions and to do that we need to fit a bootstrap model
        #   we are using the normal equations from the original model since x_train has stayed the same and the normal
        #       equations are only dependent on x_train. This saves compute.
        
        # TODO: should there be regularization here also?
        if self.model_type == 'OLS':
            model_kwargs_y_B = {'normal_eqs': model_y.normal_eqs, **model_kwargs}
            model_kwargs_z_B = {'normal_eqs': model_z.normal_eqs, **model_kwargs}
        else:
            model_kwargs_y_B = model_kwargs
            model_kwargs_z_B = model_kwargs
            
        model_y_B = self.fit_model(x_train, y_train_B, weights_train, **model_kwargs_y_B)
        model_z_B = self.fit_model(x_train, z_train_B, weights_train, **model_kwargs_z_B)

        LOG.info("orig. coefficients, normalized margin: \n %s", model_y.coefficients.flatten())
        LOG.info("boot. coefficients, normalized margin: \n %s", model_y_B.coefficients.mean(axis=-1))

        # we cannot just apply ols_y_B/old_z_B to the test units because that would be missing
        # our contest level random effect (i.e., the median of the residual distributions)
        y_train_pred_B = model_y_B.predict(x_train)
        z_train_pred_B = model_z_B.predict(x_train)

        # We can then use our bootstrapped ols_y_B/ols_z_B and our (median) estimate of the residual distribution
        # to make bootstrapped predictions on our non-reporting units
        # This is \tilde{y_i}^{b} and \tilde{z_i}^{b}
        residuals_y_B = model_y_B.residuals(x_train, y_train_B, weights_train, K=5, center=False, **model_kwargs_y)
        residuals_z_B = model_z_B.residuals(x_train, z_train_B, weights_train, K=5, center=False, **model_kwargs_z)
        residuals_y_ppf_B, _ = self._estimate_residual_dist(
            residuals_y_B, resid_features_agg_indic, resid_features_agg_feat, resid_mask,
            taus=[0.5], randomize=True, lb=self.y_LB, ub=self.y_UB # [0.5] since median only
        ) 

        residuals_z_ppf_B, _ = self._estimate_residual_dist(
            residuals_z_B, resid_features_agg_indic, resid_features_agg_feat, resid_mask,
            taus=[0.5], randomize=True, lb=self.z_LB, ub=self.z_UB
        )
        median_y_B = np.asarray([[residuals_y_ppf_B[b][i](0.5) for b in range(self.B)] for i in range(n_train, n_train + n_test)])
        median_z_B = np.asarray([[residuals_z_ppf_B[b][i](0.5) for b in range(self.B)] for i in range(n_train, n_train + n_test)])
        y_test_pred_B = (model_y_B.predict(x_test) + median_y_B).clip(
            min=y_partial_reporting_lower, max=y_partial_reporting_upper
        )
        z_test_pred_B = (model_z_B.predict(x_test) + median_z_B).clip(
            min=z_partial_reporting_lower, max=z_partial_reporting_upper
        )

        # \tilde{y_i}^{b} * \tilde{z_i}^{b}
        yz_test_pred_B = y_test_pred_B * z_test_pred_B

        # In order to generate our point prediction, we also need to apply our non-bootstrapped model to the testset
        # this is \hat{y_i} and \hat{z_i}
        median_y_test = np.asarray([residuals_y_ppf[i](0.5) for i in range(n_train, n_train + n_test)]).reshape(-1,1)
        median_z_test = np.asarray([residuals_z_ppf[i](0.5) for i in range(n_train, n_train + n_test)]).reshape(-1,1)

        y_test_pred = (model_y.predict(x_test) + median_y_test).clip(
            min=y_partial_reporting_lower, max=y_partial_reporting_upper
        )
        z_test_pred = (model_z.predict(x_test) + median_z_test).clip(
            min=z_partial_reporting_lower, max=z_partial_reporting_upper
        )
        yz_test_pred = y_test_pred * z_test_pred

        # we now need to generate our bootstrapped "true" quantities (in order to subtract the
        # bootstrapped estimates from these quantities to get an estimate for our error)
        # In a normal bootstrap setting we would replace y and z with \hat{y} and \hat{z}
        # however, we want to produce prediction intervals (rather than just confidence intervals)
        # so we need to take into account the extra uncertainty that comes with prediction
        # (ie. estimating the mean has some variance, but sampling from our mean estimate
        # introduces an additional error)
        # To do that we need an estimate for our prediction error
        # we cannot use our original residuals_y, residuals_z because
        # they would cancel out our error used in the bootstrap
        test_residuals_y, test_residuals_z = self._sample_test_errors(
            residuals_y_ppf, residuals_z_ppf, n_train, n_test,
        )

        # multiply by weights to turn into unnormalized margin
        # w_i * \hat{y_i} * \hat{z_i}
        self.errors_B_1 = yz_test_pred_B * weights_test

        # This is (\hat{y_i} + \residuals_{y, i}^{b}) * (\hat{z_i} + \residuals_{z, i}^{b})
        # we clip them based on bounds we generated earlier. These are defined by the estimates amount of
        # outstanding vote from our election results provider
        errors_B_2 = (model_y.predict(x_test) + test_residuals_y).clip(min=y_partial_reporting_lower, max=y_partial_reporting_upper)
        errors_B_2 *= (model_z.predict(x_test) + test_residuals_z).clip(
            min=z_partial_reporting_lower, max=z_partial_reporting_upper
        )

        # multiply by weights turn into unnormalized margin
        # this is w_i * (\hat{y_i} + \residuals_{y, i}^{b}) * (\hat{z_i} + \residuals_{z, i}^{b})
        self.errors_B_2 = errors_B_2 * weights_test

        # we also need errors for the denominator of the aggregate
        # this is \tilde{z_i}^{b}
        self.errors_B_3 = z_test_pred_B * weights_test  # has already been clipped above
        # this is (\hat{z_i} + \residuals_{z, i}^{b})
        self.errors_B_4 = (model_z.predict(x_test) + test_residuals_z).clip(
            min=z_partial_reporting_lower, max=z_partial_reporting_upper
        ) * weights_test

        # this is for the unit point prediction. turn into unnormalized margin
        self.weighted_yz_test_pred = yz_test_pred * weights_test
        # and turn into turnout estimate
        self.weighted_z_test_pred = z_test_pred * weights_test
        self.ran_bootstrap = True

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

    def get_aggregate_predictions(
        self,
        reporting_units: pd.DataFrame,
        nonreporting_units: pd.DataFrame,
        unexpected_units: pd.DataFrame,
        aggregate: list,
        estimand: str,
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
            aggregate_indicator = pd.get_dummies(all_units[aggregate_temp_column_name]).values
        else:
            aggregate_indicator = pd.get_dummies(all_units[aggregate]).values
            aggregate_temp_column_name = aggregate

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
        )

        # use get_aggregate_predictions from BaseElectionModel to sum unnormalized margin of all the units
        raw_margin_df = super().get_aggregate_predictions(
            reporting_units, nonreporting_units, unexpected_units, aggregate, estimand
        )

        # divide the unnormalized margin and results by the total turnout predictions
        # to get the normalized margin for the aggregate
        # turnout prediction could be zero, in which case predicted margin is also zero,
        # so replace NaNs with zero in that case
        raw_margin_df["pred_margin"] = np.nan_to_num(raw_margin_df.pred_margin / aggregate_z_total.flatten())
        raw_margin_df["results_margin"] = np.nan_to_num(raw_margin_df.results_margin / aggregate_z_total.flatten())
        # if we are in the top level prediction, then save the aggregated baseline margin,
        # which we will need for the national summary (e.g. ecv) model
        if self._is_top_level_aggregate(aggregate):
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
            aggregate_indicator = pd.get_dummies(all_units[aggregate_temp_column_name]).values
        else:
            aggregate_indicator = pd.get_dummies(all_units[aggregate]).values
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

        # subtract to get bootstrap error for estimate in our predictions
        aggregate_error_B = divided_error_B_1 - divided_error_B_2

        lower_q, upper_q = self._get_quantiles(alpha)

        # we also need to re-compute our aggregate prediction to add to our error to get the prediction interval
        # first the turnout component
        aggregate_z_total = (
            aggregate_z_unexpected + aggregate_z_train + aggregate_indicator_test.T @ self.weighted_z_test_pred
        )
        # then the unnormalied margin component
        aggregate_yz_total = (
            aggregate_yz_unexpected + aggregate_yz_train + aggregate_indicator_test.T @ self.weighted_yz_test_pred
        )
        # calculate normalized margin in the aggregate prediction
        # turnout prediction could be zero, so convert NaN -> 0
        aggregate_perc_margin_total = np.nan_to_num(aggregate_yz_total / aggregate_z_total)

        # saves the aggregate errors in case we want to generate somem form of national predictions (like ecv)
        if self._is_top_level_aggregate(aggregate):
            self.aggregate_error_B_1 = aggregate_error_B_1
            self.aggregate_error_B_2 = aggregate_error_B_2
            self.aggregate_error_B_3 = aggregate_error_B_3
            self.aggregate_error_B_4 = aggregate_error_B_4
            self.aggregate_perc_margin_total = aggregate_perc_margin_total

        interval_upper, interval_lower = (
            aggregate_perc_margin_total - np.quantile(aggregate_error_B, q=[lower_q, upper_q], axis=-1).T
        ).T
        interval_upper = interval_upper.reshape(-1, 1)
        interval_lower = interval_lower.reshape(-1, 1)

        return PredictionIntervals(interval_lower, interval_upper)

    def get_national_summary_estimates(
        self, nat_sum_data_dict: dict, called_states: dict, base_to_add: int | float, alpha: float
    ) -> list:
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
            nat_sum_data_dict = {i: 1 for i in range(self.aggregate_error_B_1.shape[0])}

        # if we didn't pass the right number of national summary weights
        # (ie. the number of contests) then raise an exception
        if len(nat_sum_data_dict) != self.aggregate_error_B_1.shape[0]:
            raise BootstrapElectionModelException(
                f"nat_sum_data_dict is of length {len(nat_sum_data_dict)} but there are {self.aggregate_error_B_1.shape[0]} contests"
            )

        # called states is a dictionary where 1 means that the LHS party has one, 0 means that the RHS party has won
        # and -1 means that the state is not called. If called_states is None, assume that all states are not called.
        if called_states is None:
            called_states = {i: -1 for i in range(self.aggregate_error_B_1.shape[0])}

        if len(called_states) != self.aggregate_error_B_1.shape[0]:
            raise BootstrapElectionModelException(
                f"called_states is of length {len(called_states)} but there are {self.aggregate_error_B_1.shape[0]} contests"
            )

        # sort in order to get in the same order as the contests,
        # which have been sorted when getting dummies for aggregate indicators
        # in get_aggregate_prediction_intervals
        nat_sum_data_dict_sorted = sorted(nat_sum_data_dict.items())
        nat_sum_data_dict_sorted_vals = np.asarray([x[1] for x in nat_sum_data_dict_sorted]).reshape(-1, 1)

        called_states_sorted = sorted(called_states.items())
        called_states_sorted_vals = (
            np.asarray([x[1] for x in called_states_sorted]).reshape(-1, 1) * 1.0
        )  # multiplying by 1.0 to turn into floats
        # since we max/min the state called values with contest win probabilities,
        # we don't want the uncalled states to have a number to max/min
        # in order for those states to keep their original computed win probability
        called_states_sorted_vals[called_states_sorted_vals == -1] = np.nan

        # technically we do not need to do this division, since the margin
        # (ie. aggregate_error_B_1 and aggregate_error_B_2)
        # are enough to know who has won a contest (we don't need the normalized margin)
        # but we normalize so that the temperature we use to set aggressiveness of sigmoid is in the right scale

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

        # since called_states_sorted_vals has value 1 if the state is called for the LHS party,
        # maxing the probabilities gives a probability of 1 for the LHS party
        # and called_states_sorted_vals has value 0 if the state is called for the RHS party,
        # so mining with probabilities gives a probability of 0 for the LHS party
        # and called_states_sorted_vals has value np.nan if the state is uncalled,
        # since we use fmax/fmin the actual number and not nan gets propagated, so we maintain the probability
        aggregate_dem_prob_B_1_called = np.fmin(
            np.fmax(aggregate_dem_prob_B_1, called_states_sorted_vals), called_states_sorted_vals
        )
        aggregate_dem_prob_B_2_called = np.fmin(
            np.fmax(aggregate_dem_prob_B_2, called_states_sorted_vals), called_states_sorted_vals
        )

        # multiply by weights of each contest
        aggregate_dem_vals_B_1 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_1_called
        aggregate_dem_vals_B_2 = nat_sum_data_dict_sorted_vals * aggregate_dem_prob_B_2_called

        # calculate the error in our national aggregate prediction
        aggregate_dem_vals_B = np.sum(aggregate_dem_vals_B_1, axis=0) - np.sum(aggregate_dem_vals_B_2, axis=0)

        # we also need a national aggregate point prediction
        if self.hard_threshold:
            aggregate_dem_probs_total = self.aggregate_perc_margin_total > 0.5
        else:
            aggregate_dem_probs_total = expit(self.T * self.aggregate_perc_margin_total)

        # same as for the intervals
        aggregate_dem_probs_total_called = np.fmin(
            np.fmax(aggregate_dem_probs_total, called_states_sorted_vals), called_states_sorted_vals
        )
        aggregate_dem_vals_pred = np.sum(nat_sum_data_dict_sorted_vals * aggregate_dem_probs_total_called)

        lower_q, upper_q = self._get_quantiles(alpha)

        interval_upper, interval_lower = (
            aggregate_dem_vals_pred - np.quantile(aggregate_dem_vals_B, q=[lower_q, upper_q], axis=-1).T
        ).T
        national_summary_estimates = {
            "margin": [
                aggregate_dem_vals_pred + base_to_add,
                interval_lower + base_to_add,
                interval_upper + base_to_add,
            ]
        }

        return national_summary_estimates
