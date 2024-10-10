from functools import reduce

import pandas as pd

from elexmodel.handlers import s3
from elexmodel.utils.constants import VALID_AGGREGATES_MAPPING
from elexmodel.utils.file_utils import S3_FILE_PATH, TARGET_BUCKET, convert_df_to_csv


class ModelResultsHandler:
    """
    Handler for model results
    """

    def __init__(
        self,
        aggregates,
        prediction_interval_alphas,
        reporting_units,
        nonreporting_units,
        unexpected_units,
    ):
        self.prediction_interval_alphas = prediction_interval_alphas
        self.include_unit_data = "unit" in aggregates
        self.aggregates = [agg for agg in aggregates if agg != "unit"]
        self.estimates = {agg: [] for agg in self.aggregates}
        self.unit_data = {}
        self.final_results = {}

        self.reporting_units = reporting_units
        self.nonreporting_units = nonreporting_units
        self.unexpected_units = unexpected_units

    def add_unit_predictions(self, estimand, unit_predictions):
        """
        unit_predictions: data frame with unit predictions, as produced by model.get_unit_predictions

        """
        self.reporting_units[f"pred_{estimand}"] = self.reporting_units[f"results_{estimand}"]
        self.nonreporting_units[f"pred_{estimand}"] = unit_predictions
        self.unexpected_units[f"pred_{estimand}"] = self.unexpected_units[f"results_{estimand}"]

    def add_unit_turnout_predictions(self, unit_turnout_predictions):
        self.reporting_units["pred_turnout"] = self.reporting_units["results_weights"]
        self.nonreporting_units["pred_turnout"] = unit_turnout_predictions
        self.unexpected_units["pred_turnout"] = self.unexpected_units["results_weights"]

    def add_unit_intervals(self, estimand, prediction_intervals_unit):
        """
        estimand: str
        prediction_intervals_unit: dict of the PredicitonIntervals class as produced
            by model.get_unit_prediction_intervals(); keys are alphas (for prediction confidence intervals)

        """
        interval_cols = []
        for alpha in self.prediction_interval_alphas:
            lower_string = f"lower_{alpha}_{estimand}"
            upper_string = f"upper_{alpha}_{estimand}"
            interval_cols.extend([lower_string, upper_string])
            self.reporting_units[lower_string] = self.reporting_units[f"results_{estimand}"]
            self.reporting_units[upper_string] = self.reporting_units[f"results_{estimand}"]
            self.nonreporting_units[lower_string] = prediction_intervals_unit[alpha].lower
            self.nonreporting_units[upper_string] = prediction_intervals_unit[alpha].upper
            self.unexpected_units[lower_string] = self.unexpected_units[f"results_{estimand}"]
            self.unexpected_units[upper_string] = self.unexpected_units[f"results_{estimand}"]
        self.unit_data[estimand] = pd.concat(
            [self.reporting_units, self.nonreporting_units, self.unexpected_units]
        ).sort_values("geographic_unit_fips")[
            ["postal_code", "geographic_unit_fips", f"pred_{estimand}", "reporting", "unit_category"]
            + interval_cols
            + [f"results_{estimand}"]
            + (["pred_turnout"] if estimand == "margin" else [])
        ]

    def add_agg_predictions(self, estimand, aggregate, estimates_df, agg_interval_predictions):
        """
        Adds a set of aggregate predictions for a given estimand

        estimand: str
        aggregate: str
        estimates: data frame with aggregate predictions, as produced by model.get_aggregate_predictions;
        agg_interval_predictions: dict of tuples of lower and upper prediction intervals as produced by
            model.get_aggregate_prediction_intervals(); keys are alphas (prediction interval)
        """
        # require that unit data already be added
        assert estimand in self.unit_data, "Need to first add unit predictions with add_unit_predictions()"

        for alpha in self.prediction_interval_alphas:
            estimates_df[f"lower_{alpha}_{estimand}"] = agg_interval_predictions[alpha][0]
            estimates_df[f"upper_{alpha}_{estimand}"] = agg_interval_predictions[alpha][1]
        self.estimates[aggregate].append(estimates_df)

    def process_final_results(self):
        """
        Create final data frames of results
        """
        for agg in self.aggregates:
            merge_on = ["postal_code", "reporting", agg]
            # joins together dfs of the same level of aggregation (different estimands)
            agg_df = reduce(lambda x, y: pd.merge(x, y, how="inner", on=merge_on), self.estimates[agg])
            self.final_results[VALID_AGGREGATES_MAPPING.get(agg)] = agg_df
        if self.include_unit_data:
            merge_on = ["postal_code", "reporting", "geographic_unit_fips"]
            # joins together unit data dfs (for different estimands)
            self.final_results["unit_data"] = reduce(
                lambda x, y: pd.merge(x, y, how="inner", on=merge_on), self.unit_data.values()
            )

    def add_national_summary_estimates(self, nat_sum_estimates_dict):
        df = pd.DataFrame(index=["margin"])
        for alpha, data in nat_sum_estimates_dict.items():
            if "agg_pred" not in df.columns:
                df["agg_pred"] = [data["margin"][0]]
            df[f"lower_{alpha}"] = [data["margin"][1]]
            df[f"upper_{alpha}"] = [data["margin"][2]]

        df.index.name = "estimand"
        self.final_results["nat_sum_data"] = df.reset_index()

    def write_data(self, election_id, office, geographic_unit_type, keys=None):
        """
        Saves dataframe of estimates for all estimands to S3
        Different file by aggregate level
        """
        if not self.final_results:
            self.process_final_results()
        s3_client = s3.S3CsvUtil(TARGET_BUCKET)
        for key, value in self.final_results.items():
            if keys is not None and key not in keys:
                continue
            path = f"{S3_FILE_PATH}/{election_id}/predictions/{office}/{geographic_unit_type}/{key}/current.csv"
            # convert df to csv
            csv_data = convert_df_to_csv(value)
            # put csv in s3
            s3_client.put(path, csv_data)
