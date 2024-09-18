import numpy as np
import pandas as pd

from elexmodel.handlers import s3
from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.utils.file_utils import S3_FILE_PATH, TARGET_BUCKET, convert_df_to_csv


class CombinedDataHandler:
    """
    Combined data handler. Combines preprocessed and live data
    """

    def __init__(
        self,
        preprocessed_data,
        current_data,
        estimands,
        geographic_unit_type,
        handle_unreporting="drop",
    ):
        self.estimands = estimands

        estimandizer = Estimandizer()
        (current_data, _) = estimandizer.add_estimand_results(current_data.copy(), self.estimands, False)
        # if we're running this for a past election, drop results columns from preprocessed data
        # so we use results_{estimand} numbers from current_data
        preprocessed_results_columns = list(filter(lambda col: col.startswith("results_"), preprocessed_data.columns))
        preprocessed_data = preprocessed_data.drop(preprocessed_results_columns, axis=1)
        self.preprocessed_data = preprocessed_data
        self.current_data = current_data
        self.geographic_unit_type = geographic_unit_type
        data = preprocessed_data.merge(current_data, how="left", on=["postal_code", "geographic_unit_fips"])
        data = estimandizer.add_turnout_factor(data)
        # if unreporting is 'drop' then drop units that are not reporting (ie. units where results are na)
        # this is necessary if units will not be returning results in this election,
        # but we didn't know that (ie. townships)
        result_cols = [f"results_{estimand}" for estimand in self.estimands]
        if handle_unreporting == "drop":
            # Drop the whole row if an estimand is not reporting
            data = data.dropna(axis=0, how="any", subset=result_cols)
        # if unreporting is 'zero' then we set the votes for non-reporting units to zero
        # this is necessary if we are worried that there is no zero state for units (ie. some precinct states)
        elif handle_unreporting == "zero":
            indices_with_null_val = data[result_cols].isna().any(axis=1)
            data.update(data[result_cols].fillna(value=0))
            data.loc[indices_with_null_val, "percent_expected_vote"] = 0

        self.data = data

    def get_units(self, percent_reporting_threshold, turnout_factor_lower, turnout_factor_upper, aggregates):
        """
        Returns a tuple of:
        1. reporting data. These are units where the expected vote is greater than the percent reporting threshold.
        2. nonreporting data. These are units where expected vote is less than the percent reporting threshold.
        3. units for which we will not be making predictions:
            - unexpected units (ie. units for which we have no covariates prepared)
            - units for which the baseline results is zero (ie. units that are tiny)
            - units with strange turnout factors (ie. units that are likely precinct mismatches)
        """

        # units where the expected vote is greater than the percent reporting threshold
        reporting_units = self.data[self.data.percent_expected_vote >= percent_reporting_threshold].reset_index(
            drop=True
        )

        # identify unexpected and non-predictive units
        unexpected_units = self._get_unexpected_units(aggregates)
        non_modeled_units = self._get_non_modeled_units(
            percent_reporting_threshold, turnout_factor_lower, turnout_factor_upper
        )

        # remove these units from the reporting units
        reporting_units = reporting_units[
            ~reporting_units.geographic_unit_fips.isin(unexpected_units.geographic_unit_fips)
        ].reset_index(drop=True)
        reporting_units = reporting_units[
            ~reporting_units.geographic_unit_fips.isin(non_modeled_units.geographic_unit_fips)
        ].reset_index(drop=True)

        # residualize + normalize
        for estimand in self.estimands:
            reporting_units[f"residuals_{estimand}"] = (
                reporting_units[f"results_{estimand}"] - reporting_units[f"last_election_results_{estimand}"]
            ) / reporting_units[f"last_election_results_{estimand}"]

        reporting_units["reporting"] = int(1)
        reporting_units["unit_category"] = "expected"

        # units where expected vote is less than the percent reporting threshold
        nonreporting_units = self.data[self.data.percent_expected_vote < percent_reporting_threshold].reset_index(
            drop=True
        )

        # remove unexpected and non-predictive units
        nonreporting_units = nonreporting_units[
            ~nonreporting_units.geographic_unit_fips.isin(unexpected_units.geographic_unit_fips)
        ].reset_index(drop=True)
        nonreporting_units = nonreporting_units[
            ~nonreporting_units.geographic_unit_fips.isin(non_modeled_units.geographic_unit_fips)
        ].reset_index(drop=True)

        nonreporting_units["reporting"] = int(0)
        nonreporting_units["unit_category"] = "expected"

        # finalize all unexpected/non-modeled units
        unexpected_units["unit_category"] = "unexpected"
        non_modeled_units["unit_category"] = "non-modeled"
        all_unexpected_units = pd.concat([unexpected_units, non_modeled_units]).reset_index(drop=True)
        all_unexpected_units["reporting"] = int(0)

        return (reporting_units, nonreporting_units, all_unexpected_units)

    def _get_expected_geographic_unit_fips(self):
        """
        Get geographic unit fips for all expected units.
        """
        # data is only expected units since left join of preprocessed data in initialization
        return self.data.geographic_unit_fips

    def _get_units_with_baseline_of_zero(self):
        return self.data[np.isclose(self.data.baseline_weights, 0)].geographic_unit_fips

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
        return components[0]

    def _get_district_from_geographic_unit_fips(self, geographic_unit_fips):
        """
        Get district from geographic unit fips
        """
        components = geographic_unit_fips.split("_")
        return components[0]

    def _get_unexpected_units(self, aggregates):
        expected_geographic_units = self._get_expected_geographic_unit_fips().tolist()
        # Note: this uses current_data because self.data drops unexpected units
        unexpected_units = (
            self.current_data[~self.current_data["geographic_unit_fips"].isin(expected_geographic_units)]
            .reset_index(drop=True)
            .drop_duplicates(subset="geographic_unit_fips")
            .copy()
        )

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

        return unexpected_units

    def _get_non_modeled_units(self, percent_reporting_threshold, turnout_factor_lower, turnout_factor_upper):
        expected_geographic_units = self._get_expected_geographic_unit_fips().tolist()
        zero_baseline_units = self._get_units_with_baseline_of_zero()

        units_with_strange_turnout_factor = (
            self.data[
                (self.data.percent_expected_vote >= percent_reporting_threshold)
                & (self.data["geographic_unit_fips"].isin(expected_geographic_units))
                & (
                    (self.data["geographic_unit_fips"].isin(zero_baseline_units))
                    | (self.data.turnout_factor <= turnout_factor_lower)
                    | (self.data.turnout_factor >= turnout_factor_upper)
                )
            ]
            .drop_duplicates(subset="geographic_unit_fips")
            .copy()
        )
        return units_with_strange_turnout_factor

    def write_data(self, election_id, office):
        s3_client = s3.S3CsvUtil(TARGET_BUCKET)
        # convert df to csv
        csv_data = convert_df_to_csv(self.current_data)
        # put csv in s3
        path = f"{S3_FILE_PATH}/{election_id}/results/{office}/{self.geographic_unit_type}/current.csv"
        s3_client.put(path, csv_data)
