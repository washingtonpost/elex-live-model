import numpy as np
import pandas as pd
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver

from elexmodel.handlers import s3
from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.handlers.data.Featurizer import Featurizer
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

        self.n_minimum_for_outlier_detection_model = 20
        self.data = data

    def get_units(
        self,
        percent_reporting_threshold,
        turnout_factor_lower,
        turnout_factor_upper,
        unit_blocklist,
        postal_code_blocklist,
        fit_margin_outlier_model,
        fit_turnout_outlier_model,
        outlier_z_threshold,
        aggregates,
    ):
        """
        Returns a tuple of:
        1. reporting data. These are units where the expected vote is greater than the percent reporting threshold.
        2. nonreporting data. These are units where expected vote is less than the percent reporting threshold.
        3. units for which we will not be making predictions:
            - unexpected units (ie. units for which we have no covariates prepared)
            - units for which the baseline results is zero (ie. units that are tiny)
            - units with strange turnout factors (ie. units that are likely precinct mismatches)
            - units that have been blocklisted
            - if margin is an estimand, units with strange margin changes
            - if fit_margin_outlier_model is True, units whose margins are outliers
            - if fit_turnout_outlier_model is True, units whose turnout factors are outliers
        """

        # units where the expected vote is greater than the percent reporting threshold
        reporting_units = self.data[self.data.percent_expected_vote >= percent_reporting_threshold].reset_index(
            drop=True
        )

        # identify unexpected and non-predictive units
        unexpected_units = self._get_unexpected_units(aggregates)

        # remove unexpected units from reporting units
        reporting_units = reporting_units[
            ~reporting_units.geographic_unit_fips.isin(unexpected_units.geographic_unit_fips)
        ].reset_index(drop=True)

        # this will be overwritten if we have any non-modeled units
        reporting_units["reporting"] = int(1)
        reporting_units["unit_category"] = "expected"

        non_modeled_units = self._get_non_modeled_units(
            reporting_units,
            turnout_factor_lower,
            turnout_factor_upper,
            unit_blocklist,
            postal_code_blocklist,
            fit_margin_outlier_model,
            fit_turnout_outlier_model,
            outlier_z_threshold,
        )

        # remove non-modeled units from reporting units
        reporting_units = reporting_units[
            ~reporting_units.geographic_unit_fips.isin(non_modeled_units.geographic_unit_fips)
        ].reset_index(drop=True)

        # residualize + normalize
        for estimand in self.estimands:
            reporting_units[f"residuals_{estimand}"] = (
                reporting_units[f"results_{estimand}"] - reporting_units[f"last_election_results_{estimand}"]
            ) / reporting_units[f"last_election_results_{estimand}"]

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
        unexpected_units["unit_category"] = "unexpected"

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

    def _fit_outlier_detection_model(self, reporting_units, response_variable, outlier_z_threshold):
        features = [
            "baseline_normalized_margin",
            "race_black_or_african_american",
            "ethnicity_likely_african_american",
            "percent_bachelor_or_higher",
            "education_bachelors_or_higher",
        ]
        features_to_use = [feature for feature in features if feature in reporting_units.columns]
        fixed_effects = ["postal_code"]
        fixed_effects_to_use = [
            fixed_effect for fixed_effect in fixed_effects if fixed_effect in reporting_units.columns
        ]
        featurizer = Featurizer(features=features_to_use, fixed_effects=fixed_effects_to_use)
        x_data = featurizer.prepare_data(
            reporting_units, center_features=False, scale_features=False, add_intercept=True
        )
        y = reporting_units[response_variable]
        qr = QuantileRegressionSolver()
        qr.fit(x_data.values, y.values, weights=reporting_units["baseline_weights"].values, taus=[0.5])
        y_hat = qr.predict(x_data.values)
        residuals = y.values - y_hat
        abs_residuals = np.abs(residuals)
        threshold = abs_residuals.mean() + outlier_z_threshold * abs_residuals.std()
        return reporting_units[(abs_residuals > threshold).flatten()].copy()

    def _get_non_modeled_units(
        self,
        reporting_units,
        turnout_factor_lower,
        turnout_factor_upper,
        unit_blocklist,
        postal_code_blocklist,
        fit_margin_outlier_model,
        fit_turnout_outlier_model,
        outlier_z_threshold,
    ):
        units_blocklisted = self.data[
            (self.data["geographic_unit_fips"].isin(unit_blocklist))
            | (self.data["postal_code"].isin(postal_code_blocklist))
        ].copy()
        units_blocklisted["unit_category"] = "non-modeled: blocklisted"

        zero_baseline_units = self._get_units_with_baseline_of_zero()
        units_with_zero_baseline = self.data[self.data["geographic_unit_fips"].isin(zero_baseline_units)].copy()
        units_with_zero_baseline["unit_category"] = "non-modeled: zero baseline"

        units_with_strange_turnout_factor = (
            reporting_units[  # these are all already reporting and expected units
                (reporting_units.turnout_factor <= turnout_factor_lower)
                | (reporting_units.turnout_factor >= turnout_factor_upper)
            ]
        ).copy()
        units_with_strange_turnout_factor["unit_category"] = "non-modeled: strange turnout factor"

        non_modeled_units_list = [units_blocklisted, units_with_zero_baseline, units_with_strange_turnout_factor]

        if fit_turnout_outlier_model and reporting_units.shape[0] > self.n_minimum_for_outlier_detection_model:
            units_with_strange_turnout_factor_modeled = self._fit_outlier_detection_model(
                reporting_units, "turnout_factor", outlier_z_threshold
            )
            units_with_strange_turnout_factor_modeled["unit_category"] = "non-modeled: strange turnout factor modeled"
            non_modeled_units_list.append(units_with_strange_turnout_factor_modeled)

        if "margin" in self.estimands:
            if fit_margin_outlier_model and reporting_units.shape[0] > self.n_minimum_for_outlier_detection_model:
                units_with_strange_margin_change_modeled = self._fit_outlier_detection_model(
                    reporting_units, "results_normalized_margin", outlier_z_threshold
                )
                units_with_strange_margin_change_modeled["unit_category"] = "non-modeled: strange margin change modeled"
                non_modeled_units_list.append(units_with_strange_margin_change_modeled)

        non_modeled_units = (
            pd.concat(non_modeled_units_list).reset_index(drop=True).drop_duplicates(subset="geographic_unit_fips")
        )
        return non_modeled_units

    def write_data(self, election_id, office):
        s3_client = s3.S3CsvUtil(TARGET_BUCKET)
        # convert df to csv
        csv_data = convert_df_to_csv(self.current_data)
        # put csv in s3
        path = f"{S3_FILE_PATH}/{election_id}/results/{office}/{self.geographic_unit_type}/current.csv"
        s3_client.put(path, csv_data)

        # save another file with no precincts
        _keep_counties = ~self.current_data["geographic_unit_fips"].str.contains(
            "_"
        )  # filter includes fips with no "_"
        _keep_counties |= self.current_data["geographic_unit_fips"].str.startswith("23")  # filter includes Maine CDs
        _keep_counties |= self.current_data["geographic_unit_fips"].str.startswith("31")  # filter includes Nebraska CDs

        csv_data_counties = self.current_data[_keep_counties]

        # convert df to csv
        csv_data_counties = convert_df_to_csv(csv_data_counties)
        # put csv in s3
        path = f"{S3_FILE_PATH}/{election_id}/results/{office}/{self.geographic_unit_type}/current_counties.csv"
        s3_client.put(path, csv_data_counties)
