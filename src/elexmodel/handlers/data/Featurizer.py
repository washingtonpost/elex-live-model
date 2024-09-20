import numpy as np
import pandas as pd


class Featurizer:
    """
    Featurizer. Normalizes features, add intercept, expands fixed effects
    """

    def __init__(self, features: list, fixed_effects: list):
        self.features = features
        # fixed effects can be a list, in which case every value of a fixed effect gets its own column
        if isinstance(fixed_effects, list):
            self.fixed_effect_cols = fixed_effects
            self.fixed_effect_params = {fe: ["all"] for fe in fixed_effects}
        # fixed effects can be a dictionary from fixed effect to values that get their own column
        # (or the string all, if we want all values)
        else:
            self.fixed_effect_cols = list(fixed_effects.keys())
            self.fixed_effect_params = {}
            for fe, params in fixed_effects.items():
                if params == "all":
                    self.fixed_effect_params[fe] = ["all"]
                else:
                    self.fixed_effect_params[fe] = params

        # we differentiate between expanded fixed effects and active fixed effect values
        # expanded fixed effects are those fixed effect values that appear in any part of the
        # the data (fitting or heldout) exlcluding those that have been dropped to avoid
        # multicolinearity when fitting.
        self.expanded_fixed_effects = []
        # complete features are features + expanded fixed effects
        self.complete_features = []

        # active fixed effects are those that appear in the fitting data (ie. ones for which
        # the model fitting computes a coefficient) but exluding those that we drop manually
        # to avoid multicolinearity when fitting
        self.active_fixed_effects = []
        # active features are features + active fixed effects
        self.active_features = []

    def _expand_fixed_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert fixed effect columns into dummy variables.
        """
        df = df.copy()
        # we want to keep the original fixed effect columns since we may need them later
        # for aggregation (ie. county fixed effect)
        original_fixed_effect_columns = df[self.fixed_effect_cols]
        # set non-included values to 'other' as needed since we don't want their values
        # to get a dummy variable
        for fe, params in self.fixed_effect_params.items():
            if "all" not in params:
                df[fe] = np.where(~df[fe].isin(params), "other", df[fe])

        expanded_fixed_effects = pd.get_dummies(
            df, columns=self.fixed_effect_cols, prefix=self.fixed_effect_cols, prefix_sep="_", dtype=np.int64
        )

        return pd.concat([original_fixed_effect_columns, expanded_fixed_effects], axis=1)

    def _get_categories_for_fe(self, list_: list, fe: str) -> list:
        """
        Return list of fixed effects values for a given fixed effect
        """
        return [x for x in list_ if x.startswith(fe)]

    def prepare_data(
        self, df: pd.DataFrame, center_features: bool = True, scale_features: bool = True, add_intercept: bool = True
    ) -> pd.DataFrame:
        """
        Prepares features.
        Adds dummy variables for fixed effects, also determines which fixed effects are expanded and active.
        if center_features is true we subtract the features by their average column value,
            which sets the average column value to zero
            this allows us to interpret the intercept as the mean response
            given all other covariates at their average value
        if scale_features is true we divide the features by their standard deviation,
            which gives them all the same scale
            this can improve the convergence of optimization algorithms
        if add_intercept is true an intercept column is added to the features and one fixed effect value is dropped
        """
        df = df.copy()  # create copy so we can do things to the values
        if center_features:
            df[self.features] -= df[self.features].mean()
        if scale_features:
            # this expects there to be some variation in the data, otherwise we are dividing by zero
            df[self.features] /= df[self.features].std()
        if add_intercept:
            self.complete_features += ["intercept"]
            self.active_features += ["intercept"]
            df["intercept"] = 1

        if len(self.fixed_effect_cols) > 0:
            df = self._expand_fixed_effects(df)

            # we save the expanded fixed effects to be able to add a zero column for those
            # fixed effect values if they are not in the heldout_data (nonreporting units).
            # Also we can use this to guarantee the order of the fixed effect columns
            # when fitting the model
            all_expanded_fixed_effects = [
                x
                for x in df.columns
                if x.startswith(tuple(fixed_effect + "_" for fixed_effect in self.fixed_effect_cols))
            ]

            df_fitting = df[(df.reporting) & (df.unit_category == "expected")]
            # get the indices of all expanded fixed effects in the fitting data
            # (active fixed effects + the fixed effect we will drop for multicolinearity)
            active_fixed_effect_boolean_df = df_fitting[all_expanded_fixed_effects].sum(axis=0) > 0
            # get the names of those fixed effects, since we we will want to know which fixed effect was dropped
            all_active_fixed_effects = np.asarray(all_expanded_fixed_effects)[active_fixed_effect_boolean_df]

            # if we add an intercept we need to drop a value/column per fixed effect
            # in order to avoid multicolinearity.
            # the intercept column is now a stand-in for the the dropped fixed effect value/column
            if add_intercept:
                active_fixed_effects = []  # fixed effects that exist in the fitting_data
                # (excluding one dropped column to avoid multicolinearity)
                intercept_column = (
                    []
                )  # we need to save the fixed effect categories that the intercept is now standing in for
                # we want to drop one value/column per fixed effect to avoid multicolinearity
                for fe in self.fixed_effect_cols:
                    # grab the potentially active fixed effect names for this fixed effect
                    fe_fixed_effect_filter = self._get_categories_for_fe(all_active_fixed_effects, fe)
                    # drop the first potentially active fixed effect
                    active_fixed_effects.extend(fe_fixed_effect_filter[1:])
                    # save the name of the fixed effect that we dropped
                    intercept_column.append(fe_fixed_effect_filter[0])

                self.active_fixed_effects = active_fixed_effects
                self.intercept_column = intercept_column
                # expanded fixed effects do not include the ones that we dropped to avoid multicolinearity
                self.expanded_fixed_effects = [x for x in all_expanded_fixed_effects if x not in intercept_column]
            else:
                self.active_fixed_effects = all_active_fixed_effects
                self.expanded_fixed_effects = all_expanded_fixed_effects

        # all features that the model will be fit on
        # these are all the features + the expanded fixed effects
        # (so all fixed effect values in the complete data excluding the ones dropped for multicolinearity)
        self.complete_features += self.features + self.expanded_fixed_effects
        self.active_features += self.features + self.active_fixed_effects
        df = df[self.complete_features]

        return df

    def filter_to_active_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get active features (ie. features + active fixed effects)
        """
        return df[self.active_features]

    def generate_holdout_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate fixed effects for the holdout data (ie. data that we will predict on)
        """
        df = df.copy()

        # if a unit has an inactive fixed effect value for some fixed effect category we need
        # to insert 1 / (number of fixed effect values) into each active fixed effect value for that unit
        # if we were to leave them as zero, then the model would apply the dropped fixed effect
        # value coefficient (since this is now what the intercept stands in for)
        # instead we want to apply all active fixed effect coefficients equally

        # get inactive fixed effects (ie expanded fixed effects that are not active)
        # these are fixed effects that exist only in the holdout set (ie. we do not have a covariate for them)
        inactive_fixed_effects = [x for x in self.expanded_fixed_effects if x not in self.active_fixed_effects]
        for fe in self.fixed_effect_cols:
            # active fixed effect values for this fixed effect
            fe_active_fixed_effects = self._get_categories_for_fe(self.active_fixed_effects, fe)
            # inactive fixed effect values for this fixed effect
            fe_inactive_fixed_effects = self._get_categories_for_fe(inactive_fixed_effects, fe)
            # get rows that have an inactive fixed effect
            rows_w_inactive_fixed_effects = df[fe_inactive_fixed_effects].sum(axis=1) > 0

            # set the values for active fixed effect in rows that have inactive fixed effect to be 1 / (n + 1)
            # rows that have an inactive fixed effect value need to receive the treat of the average fixed effects
            df[fe_active_fixed_effects] = df[fe_active_fixed_effects].astype("float64")
            df.loc[rows_w_inactive_fixed_effects, fe_active_fixed_effects] = 1 / (len(fe_active_fixed_effects) + 1)
            # This is correct because even rows with active fixed effects have an interept columns, so the coefficient
            # of the fixed effect value column is actually the *difference* between the dropped column
            # (for which the intercept is the stand in and the fixed effect column).
            # Another way to think about this is that for a fixed effect value that is present,
            # the fixed effect estimate is:
            # if there are three fixed effects r, u and s where s is dropped.
            # beta_0 + beta_r * indic{r}
            # beta_0 + beta_u * indic{u}
            # and the fixed effect estimate for the dropped value is beta_0, so the average is:
            # beta_0 + (beta_r / 3) + (beta_u / 3)

        return self.filter_to_active_features(df)
