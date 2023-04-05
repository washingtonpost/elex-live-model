import numpy as np
import pandas as pd


class Featurizer(object):
    """
    Featurizer. Normalizes features, add intercept, expands fixed effects
    """

    def __init__(self, features, fixed_effects):
        self.features = features
        self.fixed_effects = fixed_effects
        self.expanded_fixed_effects = []
        self.complete_features = None
        self.column_means = None

    def compute_means_for_centering(self, *arg):
        """
        Computes and saves the column mean of pandas dataframe passed as args.
        This is used for centering.
        """
        data = pd.concat(arg)
        self.column_means = data[self.features].mean()

    def _center_features(self, df):
        """
        Centers the features. This changes the interpretation of the intercept coefficient
        from conditional mean given covariates = 0, to conditional mean given covariates are
        their average value
        """
        df[self.features] = df[self.features] - self.column_means

    def _add_intercept(self, df):
        df["intercept"] = 1

    def _expand_fixed_effects(self, df, drop_first):
        """
        Convert fixed effect columns into dummy variables.
        """
        # we concatenate the dummy variables with the original fixed effects, since we need the original fixed
        # effect columns for aggregation.
        original_fixed_effect_columns = df[self.fixed_effects]
        return pd.concat(
            [
                pd.get_dummies(
                    df,
                    columns=self.fixed_effects,
                    prefix=self.fixed_effects,
                    prefix_sep="_",
                    dtype=np.int64,
                    drop_first=drop_first,
                ),
                original_fixed_effect_columns,
            ],
            axis=1,
        )

    def featurize_fitting_data(self, fitting_data, center_features=True, add_intercept=True):
        """
        Featurize the data that the model is fitted on.
        """
        # make copy of fitting_data, since we do not want to change the original data
        new_fitting_data = fitting_data.copy()
        self.center_features = center_features
        self.add_intercept = add_intercept

        if self.center_features:
            self._center_features(new_fitting_data)

        if self.add_intercept:
            self._add_intercept(new_fitting_data)

        if len(self.fixed_effects) > 0:
            # drop_first is True for fitting_data (reporting_units) since we want to avoid the design matrix with
            # expanded fixed effects to be linearly dependent
            new_fitting_data = self._expand_fixed_effects(new_fitting_data, drop_first=True)
            # we save the expanded fixed effects to be able to add fixed effects that are not in the heldout_data (nonreporting_units)
            # as a zero column and to be able to specify the order of the expanded fixed effect when fitting the model
            self.expanded_fixed_effects = [
                x
                for x in new_fitting_data.columns
                if x.startswith(tuple([fixed_effect + "_" for fixed_effect in self.fixed_effects]))
            ]

        # all features that the model will be fit on
        self.complete_features = ["intercept"] + self.features + self.expanded_fixed_effects

        return new_fitting_data[self.complete_features]

    def featurize_heldout_data(self, heldout_data):
        """
        Featurize the data that the model will be applied on.
        """
        new_heldout_data = heldout_data.copy()

        if self.center_features:
            self._center_features(new_heldout_data)

        if self.add_intercept:
            self._add_intercept(new_heldout_data)

        if len(self.fixed_effects) > 0:
            missing_expanded_fixed_effects = []
            new_heldout_data = self._expand_fixed_effects(new_heldout_data, drop_first=False)
            # if all units from one fixed effect are reporting they will not appear in the heldout_data (nonreporting_units) and won't
            # get a column when we expand the fixed effects on that dataframe. Therefore we add those columns with zero
            # fixed effects manually. As an example, if we are running a county model using state fixed effects, and
            # all of Delaware's counties are reporting, then no Delaware county will be in heldout_data (nonreporting_units), as a result
            # there will be no column for Delaware in the expanded fixed effects of heldout_data (nonreporting_units).
            for expanded_fixed_effect in self.expanded_fixed_effects:
                if expanded_fixed_effect not in new_heldout_data.columns:
                    missing_expanded_fixed_effects.append(expanded_fixed_effect)

            missing_expanded_fixed_effects_df = pd.DataFrame(
                np.zeros((new_heldout_data.shape[0], len(missing_expanded_fixed_effects))),
                columns=missing_expanded_fixed_effects,
            )
            # if we use this method to add the missing expanded fixed effects because doing it manually
            # ie. new_heldout_data[expanded_fixed_effect] = 0
            # can throw a fragmentation warning when there are many missing fixed effects.
            new_heldout_data = new_heldout_data.join(missing_expanded_fixed_effects_df)

        return new_heldout_data[self.complete_features]
