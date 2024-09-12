import logging
from abc import ABC
from collections import namedtuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

PredictionIntervals = namedtuple("PredictionIntervals", ["lower", "upper"], defaults=(None,) * 2)


class BaseElectionModel(ABC):
    def __init__(self, model_settings: dict):
        self.features = model_settings.get("features", [])
        self.fixed_effects = model_settings.get("fixed_effects", {})
        self.model_settings = model_settings
        self.features_to_coefficients = {}
        self.add_intercept = True
        self.seed = model_settings.get("seed", 4191)

    def get_minimum_reporting_units(self, alpha: float) -> int:
        """
        Returns the minimum number of units necessary to run the model
        """
        return 10

    @classmethod
    def get_unit_predictions(
        cls, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, estimand: str, *kwargs
    ) -> np.ndarray:
        """
        Generates and returns unit level predictions
        """
        raise NotImplementedError

    def _get_reporting_aggregate_votes(
        self, reporting_units: pd.DataFrame, unexpected_units: pd.DataFrame, aggregate: list, estimand: str, *kwargs
    ) -> pd.DataFrame:
        """
        Aggregate reporting votes by aggregate (ie. postal_code, county_fips etc.). This function
        adds reporting data and reporting unexpected data by aggregate. Note that all unexpected units -
        whether or not they are fully reporting - are included in this function.
        """
        reporting_units_known_votes = reporting_units.groupby(aggregate).sum().reset_index(drop=False)

        # we cannot know the county classification of unexpected geographic units, so we can't add the votes back in
        if "county_classification" in aggregate:
            aggregate_votes = reporting_units_known_votes[aggregate + [f"results_{estimand}", "reporting"]]
        else:
            unexpected_units_known_votes = unexpected_units.groupby(aggregate).sum().reset_index(drop=False)

            # outer join to ensure that if entire districts of county classes are unexpectedly present, we
            # should still have them. Same reasoning to replace NA with zero
            # NA means there was no such geographic unit, so they don't capture any votes
            results_col = f"results_{estimand}"
            reporting_col = "reporting"
            aggregate_votes = (
                reporting_units_known_votes.merge(
                    unexpected_units_known_votes,
                    how="outer",
                    on=aggregate,
                    suffixes=("_expected", "_unexpected"),
                )
                .fillna(
                    {
                        f"results_{estimand}_expected": 0,
                        f"results_{estimand}_unexpected": 0,
                        "reporting_expected": 0,
                        "reporting_unexpected": 0,
                    }
                )
                .assign(
                    **{
                        results_col: lambda x: x[f"results_{estimand}_expected"] + x[f"results_{estimand}_unexpected"],
                        reporting_col: lambda x: x["reporting_expected"] + x["reporting_unexpected"],
                    },
                )[aggregate + [f"results_{estimand}", "reporting"]]
            )

        return aggregate_votes

    def _get_nonreporting_aggregate_votes(
        self, nonreporting_units: pd.DataFrame, aggregate: list, *kwargs
    ) -> pd.DataFrame:
        """
        Aggregate nonreporting votes by aggregate (ie. postal_code, county_fips etc.). Note that all unexpected
        units - whether or not they are fully reporting - are handled in "_get_reporting_aggregate_votes" above
        """
        aggregate_nonreporting_units_known_votes = nonreporting_units.groupby(aggregate).sum().reset_index(drop=False)

        return aggregate_nonreporting_units_known_votes

    def get_aggregate_predictions(
        self,
        reporting_units: pd.DataFrame,
        nonreporting_units: pd.DataFrame,
        unexpected_units: pd.DataFrame,
        aggregate: list,
        estimand: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Aggregate predictions and results by aggregate (ie. postal_code, county_fips etc.). Add results from reporting
        and reporting unexpected units and then sum in the predictions from nonreporting units.
        """
        # these are subunits that are already counted
        aggregate_votes = self._get_reporting_aggregate_votes(reporting_units, unexpected_units, aggregate, estimand)

        # these are subunits that are not already counted
        aggregate_preds = self._get_nonreporting_aggregate_votes(nonreporting_units, aggregate).rename(
            columns={
                f"pred_{estimand}": f"pred_only_{estimand}",
                f"results_{estimand}": f"results_only_{estimand}",
                "reporting": "reporting_only",
            }
        )
        aggregate_data = (
            aggregate_votes.merge(aggregate_preds, how="outer", on=aggregate)
            .fillna(
                {
                    f"results_{estimand}": 0,
                    f"pred_only_{estimand}": 0,
                    f"results_only_{estimand}": 0,
                    "reporting": 0,
                    "reporting_only": 0,
                }
            )
            .assign(
                # don't need to sum results_only for predictions since those are superceded by pred_only
                # preds can't be smaller than results, since we maxed between predictions and results in unit function.
                **{
                    f"pred_{estimand}": lambda x: x[f"results_{estimand}"] + x[f"pred_only_{estimand}"],
                    f"results_{estimand}": lambda x: x[f"results_{estimand}"] + x[f"results_only_{estimand}"],
                    "reporting": lambda x: x["reporting"] + x["reporting_only"],
                },
            )
            .sort_values(aggregate)[aggregate + [f"pred_{estimand}", f"results_{estimand}", "reporting"]]
            .reset_index(drop=True)
        )

        return aggregate_data

    @classmethod
    def get_unit_prediction_intervals(
        cls, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, alpha: float, estimand: str
    ) -> PredictionIntervals:
        """
        Generates and returns unit level prediction intervals
        """
        raise NotImplementedError

    @classmethod
    def get_aggregate_prediction_intervals(
        cls,
        reporting_units: pd.DataFrame,
        nonreporting_units: pd.DataFrame,
        unexpected_units: pd.DataFrame,
        aggregate: list,
        alpha: float,
        **kwargs,
    ) -> PredictionIntervals:
        """
        Generates and returns aggregate prediction intervals for arbitrary aggregates
        """
        raise NotImplementedError

    def get_coefficients(self) -> dict:
        """
        Returns a dictionary of feature/fixed effect names to the coefficients
        These coefficients are for the point prediciton only.
        """
        return self.features_to_coefficients

    def get_national_summary_estimates(self, nat_sum_data_dict, called_states, base_to_add, alpha):
        raise NotImplementedError()
