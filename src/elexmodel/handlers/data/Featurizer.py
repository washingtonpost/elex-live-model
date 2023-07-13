import numpy as np
import pandas as pd


class Featurizer:
    """
    Featurizer. Normalizes features, add intercept, expands fixed effects
    """

    def __init__(self, features, fixed_effects):
        self.features = features
        if isinstance(fixed_effects, list):
            self.fixed_effect_cols = fixed_effects
            self.fixed_effect_params = {fe: ["all"] for fe in fixed_effects}
        else:
            self.fixed_effect_cols = list(fixed_effects.keys())
            self.fixed_effect_params = {}
            for fe, params in fixed_effects.items():
                if params == "all":
                    self.fixed_effect_params[fe] = ["all"]
                else:
                    self.fixed_effect_params[fe] = params

        self.complete_features = []
        self.active_features = []
        self.active_fixed_effects = []
        self.expanded_fixed_effects = []

    def _expand_fixed_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert fixed effect columns into dummy variables.
        """
        original_fixed_effect_columns = df[self.fixed_effect_cols]
        # set non-included values to other as needed
        for fe, params in self.fixed_effect_params.items():
            if "all" not in params:
                df[fe] = np.where(~df[fe].isin(params), "other", df[fe])

        expanded_fixed_effects = pd.get_dummies(
            df, columns=self.fixed_effect_cols, prefix=self.fixed_effect_cols, prefix_sep="_", dtype=np.int64
        )

        # drop first column or "other" column if drop_first is true
        # cols_to_drop = []
        # if len(columns_to_drop) > 0:
            # for fixed_effect in self.fixed_effect_cols:
                # relevant_cols = [col for col in expanded_fixed_effects.columns if col.startswith(fixed_effect)]
                # if f"{fixed_effect}_other" in relevant_cols:
                #     cols_to_drop.append(f"{fixed_effect}_other")
                # else:
                #     cols_to_drop.append(relevant_cols[0])
        # the intercept column corresponds to the effect for the column that we dropped
        # self.intercept_column = cols_to_drop
        # we concatenate the dummy variables with the original fixed effects, since we need the original fixed
        # effect columns for aggregation.
        # return pd.concat([original_fixed_effect_columns, expanded_fixed_effects.drop(columns_to_drop, axis=1)], axis=1)
        return pd.concat([original_fixed_effect_columns, expanded_fixed_effects], axis=1)

    def prepare_data(self, df, center_features=True, scale_features=True, add_intercept=True):
        df = df.copy()
        if center_features:
            df[self.features] -= df[self.features].mean()
        if scale_features:
            df[self.features] /= df[self.features].std()
        if add_intercept:
            self.complete_features += ['intercept']
            self.active_features += ['intercept']
            df['intercept'] = 1

        if len(self.fixed_effect_cols) > 0:
            df = self._expand_fixed_effects(df)

            # we save the expanded fixed effects to be able to add fixed effects that are
            # not in the heldout_data (nonreporting_units) as a zero column and to be able
            # to specify the order of the expanded fixed effect when fitting the model
            all_expanded_fixed_effects = [
                x
                for x in df.columns
                if x.startswith(tuple([fixed_effect + "_" for fixed_effect in self.fixed_effect_cols]))
            ]

            df_fitting = df[(df.reporting == True) & (df.expected == True)]
            active_fixed_effect_boolean_df = df_fitting[all_expanded_fixed_effects].sum(axis=0) > 0
            all_active_fixed_effects = np.asarray(all_expanded_fixed_effects)[active_fixed_effect_boolean_df]
            if add_intercept:
                active_fixed_effects = [] # fixed effects that exist in the fitting_data (excluding one dropped column to avoid multicolinearity)
                intercept_column = [] # we need to save the fixed effect categories that the intercept is now standing in for
                # want to drop one category per fixed effect to avoid multicolinearity
                for fe in self.fixed_effect_cols:
                    fe_fixed_effect_filter = [x for x in all_active_fixed_effects if x.startswith(fe)]
                    active_fixed_effects.extend(fe_fixed_effect_filter[1:])
                    intercept_column.append(fe_fixed_effect_filter[0])
            
                self.active_fixed_effects = active_fixed_effects
                self.intercept_column = intercept_column
                self.expanded_fixed_effects = [x for x in all_expanded_fixed_effects if x not in intercept_column]
            else:
                self.active_fixed_effects = all_active_fixed_effects
                self.expanded_fixed_effects = all_expanded_fixed_effects

        # all features that the model will be fit on
        self.complete_features += self.features + self.expanded_fixed_effects
        self.active_features += self.features + self.active_fixed_effects
        df = df[self.complete_features]
        
        # self.expanded_fixed_effects_cols = [df.columns.get_loc(c) for c in self.expanded_fixed_effects]
        return df

    def filter_to_active_features(self, df):
        return df[self.active_features]
    
    def _get_categories_for_fe(self, list_, fe):
        return [x for x in list_ if x.startswith(fe)]
    
    def generate_holdout_data(self, df):
        df = df.copy()
        inactive_fixed_effects = [x for x in self.expanded_fixed_effects if x not in self.active_fixed_effects]
        for fe in self.fixed_effect_cols:
            fe_active_fixed_effects = self._get_categories_for_fe(self.active_fixed_effects, fe)
            fe_inactive_fixed_effects = self._get_categories_for_fe(inactive_fixed_effects, fe)
            bad_rows = df[fe_inactive_fixed_effects].sum(axis=1) > 0
            df.loc[bad_rows, fe_active_fixed_effects] = 1 / (len(fe_active_fixed_effects) + 1)
        return self.filter_to_active_features(df)
    # def featurize_fitting_data(self, fitting_data, center_features=True, add_intercept=True, scale_features=False):
    #     """
    #     Featurize the data that the model is fitted on.
    #     In our case fitting_data is either the reporting_units (when fitting a model for the point predictions)
    #     or training_data (when fitting the model for the prediction intervals)
    #     """
    #     # make copy of fitting_data, since we do not want to change the original data
    #     new_fitting_data = fitting_data.copy()
    #     self.center_features = center_features
    #     self.add_intercept = add_intercept
    #     self.scale_features = scale_features

    #     if self.center_features:
    #         self._center_features(new_fitting_data)

    #     if self.scale_features:
    #         self._scale_features(new_fitting_data)

    #     self.complete_features = []
    #     if self.add_intercept:
    #         self.complete_features += ["intercept"]
    #         self._add_intercept(new_fitting_data)

    #     if len(self.fixed_effect_cols) > 0:
    #         # drop_first is True for fitting_data (e.g. reporting_units) since we want to avoid the design matrix with
    #         # expanded fixed effects to be linearly dependent
    #         new_fitting_data = self._expand_fixed_effects(new_fitting_data, drop_first=True)
    #         # we save the expanded fixed effects to be able to add fixed effects that are
    #         # not in the heldout_data (nonreporting_units) as a zero column and to be able
    #         # to specify the order of the expanded fixed effect when fitting the model
    #         self.expanded_fixed_effects = [
    #             x
    #             for x in new_fitting_data.columns
    #             if x.startswith(tuple([fixed_effect + "_" for fixed_effect in self.fixed_effect_cols]))
    #         ]

    #     # all features that the model will be fit on
    #     self.complete_features += self.features + self.expanded_fixed_effects
    #     new_fitting_data = new_fitting_data[self.complete_features]
        
    #     self.expanded_fixed_effects_cols = [new_fitting_data.columns.get_loc(c) for c in self.expanded_fixed_effects]

    #     return new_fitting_data

    # def featurize_heldout_data(self, heldout_data):
    #     """
    #     Featurize the data that the model will be applied on.
    #     In our case the heldout_data is either the nonreporting_units
    #     (when applying the model for the point predictions)
    #     or conformalization_data/nonreporting_units
    #     (when applying the model for the prediction intervals)
    #     """
    #     new_heldout_data = heldout_data.copy()

    #     if self.center_features:
    #         self._center_features(new_heldout_data)

    #     if self.scale_features:
    #         self._scale_features(new_heldout_data)

    #     if self.add_intercept:
    #         self._add_intercept(new_heldout_data)

    #     if len(self.fixed_effect_cols) > 0:
    #         missing_expanded_fixed_effects = []
    #         new_heldout_data = self._expand_fixed_effects(new_heldout_data, drop_first=False)
    #         # if all units from one fixed effect are reporting they will not appear in the heldout_data
    #         # (e.g. nonreporting_units) and won't get a column when we expand the fixed effects
    #         # on that dataframe. Therefore we add those columns with zero fixed effects manually.
    #         # As an example, if we are running a county model using state fixed effects, and
    #         # all of Delaware's counties are reporting, then no Delaware county will be in
    #         # heldout_data (nonreporting_units), as a result there will be no column for Delaware
    #         # in the expanded fixed effects of heldout_data (nonreporting_units).
    #         for expanded_fixed_effect in self.expanded_fixed_effects:
    #             if expanded_fixed_effect not in new_heldout_data.columns:
    #                 missing_expanded_fixed_effects.append(expanded_fixed_effect)

    #         missing_expanded_fixed_effects_df = pd.DataFrame(
    #             np.zeros((new_heldout_data.shape[0], len(missing_expanded_fixed_effects))),
    #             columns=missing_expanded_fixed_effects,
    #         )
    #         # if we use this method to add the missing expanded fixed effects because doing it manually
    #         # ie. new_heldout_data[expanded_fixed_effect] = 0
    #         # can throw a fragmentation warning when there are many missing fixed effects.
    #         new_heldout_data = new_heldout_data.join(missing_expanded_fixed_effects_df)

    #     return new_heldout_data[self.complete_features]