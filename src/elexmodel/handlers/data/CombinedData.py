import numpy as np
import pandas as pd

from elexmodel.handlers import s3
from elexmodel.utils.file_utils import S3_FILE_PATH, TARGET_BUCKET, convert_df_to_csv


class CombinedDataHandler(object):
    """
    Combined data handler. Combines preprocessed and live data
    """

    def __init__(
        self,
        preprocessed_data,
        current_data,
        estimands,
        geographic_unit_type,
        fixed_effects=[],
        handle_unreporting="drop",
    ):
        self.estimands = estimands
        # if we're running this for a past election, drop results columns from preprocessed data
        # so we use results_{estimand} numbers from current_data
        preprocessed_results_columns = list(filter(lambda col: col.startswith("results_"), preprocessed_data.columns))
        preprocessed_data = preprocessed_data.drop(preprocessed_results_columns, axis=1)
        self.preprocessed_data = preprocessed_data
        self.current_data = current_data
        self.geographic_unit_type = geographic_unit_type
        data = preprocessed_data.merge(current_data, how="left", on=["postal_code", "geographic_unit_fips"])
        # if unreporting is 'drop' then drop units that are not reporting (ie. units where results are na)
        # this is necessary if units will not be returning results in this election, but we didn't know that (ie. townships)
        result_cols = [f"results_{estimand}" for estimand in estimands]
        if handle_unreporting == "drop":
            # Drop the whole row if an estimand is not reporting
            data = data.dropna(axis=0, how="any", subset=result_cols)
        # if unreporting is 'zero' then we set the votes for non-reporting units to zero
        # this is necessary if we are worried that there is no zero state for units (ie. some precincts)
        elif handle_unreporting == "zero":
            indices_with_null_val = data[result_cols].isna().any(axis=1)
            data.update(data[result_cols].fillna(value=0))
            data.loc[indices_with_null_val, "percent_expected_vote"] = 0
        self.fixed_effects = fixed_effects
        self.expanded_fixed_effects = []

        self.data = data

    @classmethod
    def _expand_fixed_effects(self, data, fixed_effects, drop_first):
        """
        Turn fixed effect columns into dummy variables. Concatenates original columns also
        """
        # we concatenate the dummies with the original fixed effects, since we need the original fixed effect
        # columns in order to potentially aggregate on them.
        original_fixed_effect_columns = data[fixed_effects]
        return pd.concat(
            [
                pd.get_dummies(
                    data,
                    columns=fixed_effects,
                    prefix=fixed_effects,
                    prefix_sep="_",
                    dtype=np.int64,
                    drop_first=drop_first,
                ),
                original_fixed_effect_columns,
            ],
            axis=1,
        )

    def _normalize_features(self, df, features):
        """
        Normalize features. Columnize normalization, to make coefficients of model interpretable
        """
        return df[features] - df[features].mean()

    def get_reporting_units(self, percent_reporting_threshold, features_to_normalize=[], add_intercept=True):
        """
        Get reporting data. These are units where the expected vote is greater than the percent reporting threshold.
        """
        reporting_units = self.data[self.data.percent_expected_vote >= percent_reporting_threshold].reset_index(
            drop=True
        )

        # residualize + normalize
        for estimand in self.estimands:
            reporting_units[f"residuals_{estimand}"] = (
                reporting_units[f"results_{estimand}"] - reporting_units[f"last_election_results_{estimand}"]
            ) / reporting_units[f"total_voters_{estimand}"]

        if features_to_normalize:
            reporting_units[features_to_normalize] = self._normalize_features(reporting_units, features_to_normalize)

        if add_intercept:
            # we effectively always need the intercept for the model to work
            reporting_units["intercept"] = 1

        if len(self.fixed_effects) > 0:
            # drop_first is True for reporting units since we want to avoid the design matrix with
            # expanded fixed effects to be linearly dependent
            reporting_units = self._expand_fixed_effects(reporting_units, self.fixed_effects, drop_first=True)
            # we save the expanded fixed effects to be able to add fixed effects that are not in the non-reporting
            # units as a zero column and to be able to specify the order of the expanded fixed effect when fitting
            # the model
            self.expanded_fixed_effects = [
                x
                for x in reporting_units.columns
                if x.startswith(tuple([fixed_effect + "_" for fixed_effect in self.fixed_effects]))
            ]

        reporting_units["reporting"] = 1
        return reporting_units

    def get_nonreporting_units(self, percent_reporting_threshold, features_to_normalize=[], add_intercept=True):
        """
        Get nonreporting data. These are units where expected vote is less than the percent reporting threshold
        """
        nonreporting_units = (
            self.data.query(
                "percent_expected_vote < @percent_reporting_threshold"
            )  # not checking if results.isnull() anymore across multiple estimands
            .reset_index(drop=True)
            .assign(residuals=np.nan)
        )

        if features_to_normalize:
            nonreporting_units[features_to_normalize] = self._normalize_features(
                nonreporting_units, features_to_normalize
            )

        if add_intercept:
            nonreporting_units["intercept"] = 1

        if len(self.fixed_effects) > 0:
            missing_expanded_fixed_effects = {}
            nonreporting_units = self._expand_fixed_effects(nonreporting_units, self.fixed_effects, drop_first=False)
            # if all units from one fixed effect are reporting they will not appear in the nonreporting_units and won't
            # get a column when we expand the fixed effects on that dataframe. Therefore we add those columns with zero
            # fixed effects manually.
            for expanded_fixed_effect in self.expanded_fixed_effects:
                if expanded_fixed_effect not in nonreporting_units.columns:
                    missing_expanded_fixed_effects[expanded_fixed_effect] = [0]
            missing_expanded_fixed_effects_df = pd.DataFrame(missing_expanded_fixed_effects)
            # if we use this method to add the missing expanded fixed effects because doing it manually
            # can throw a fragmentation warning when there are many missing fixed effects.
            nonreporting_units = pd.concat([nonreporting_units, missing_expanded_fixed_effects_df], axis=1)
            # this is necessary because the concat above creates a row which has zeroes for the expanded
            # fixed effects and NaN for all other columns.
            nonreporting_units = nonreporting_units[~nonreporting_units.postal_code.isnull()].reset_index(drop=True)

        nonreporting_units["reporting"] = 0

        return nonreporting_units

    def _get_expected_geographic_unit_fips(self):
        """
        Get geographic unit fips for all expected units.
        """
        # data is only expected units since left join of preprocessed data in initialization
        return self.data.geographic_unit_fips

    def _get_county_fips_from_geographic_unit_fips(self, geographic_unit_fips):
        """
        Get county fips for geographic unit fips
        If district is part of the geographic units, we list it first to make house race parsing easier,
        otherwise county is first
        e.g. <district>_<county> or <district>_<county>_<precinct>
        """
        components = geographic_unit_fips.split("_")
        if "district" in self.geographic_unit_type:
            return components[1]
        else:
            return components[0]

    def _get_district_from_geographic_unit_fips(self, geographic_unit_fips):
        """
        Get district from geographic unit fips
        """
        components = geographic_unit_fips.split("_")
        return components[0]

    def get_unexpected_units(self, percent_reporting_threshold, aggregates):
        """
        Gets reporting but unexpected data. These are units that are may or may not be fully
        reporting, but we have no preprocessed data for them.
        """
        expected_geographic_units = self._get_expected_geographic_unit_fips().tolist()
        # Note: this uses current_data because self.data drops unexpected units
        unexpected_units = self.current_data[
            ~self.current_data["geographic_unit_fips"].isin(expected_geographic_units)
        ].reset_index(drop=True)

        # since we were not expecting them, we have don't have their county or district
        # from preprocessed data. so we have to add that back in.
        if "county_fips" in aggregates:
            unexpected_units["county_fips"] = unexpected_units["geographic_unit_fips"].apply(
                self._get_county_fips_from_geographic_unit_fips
            )

        if "district" in aggregates:
            unexpected_units["district"] = unexpected_units["geographic_unit_fips"].apply(
                self._get_district_from_geographic_unit_fips
            )

        unexpected_units["reporting"] = 1

        return unexpected_units

    def write_data(self, election_id, office):
        s3_client = s3.S3CsvUtil(TARGET_BUCKET)
        # convert df to csv
        csv_data = convert_df_to_csv(self.current_data)
        # put csv in s3
        path = f"{S3_FILE_PATH}/{election_id}/results/{office}/{self.geographic_unit_type}/current.csv"
        s3_client.put(path, csv_data)
