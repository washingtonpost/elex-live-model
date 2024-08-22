import math

import numpy as np
import pandas as pd

from elexmodel.models.ConformalElectionModel import ConformalElectionModel, PredictionIntervals


class NonparametricElectionModel(ConformalElectionModel):
    def __init__(self, model_settings: dict):
        super().__init__(model_settings)
        self.robust = model_settings.get("robust", False)
        self.conformalization_data_agg = None
        self.conformalization_data_unit = None

    def _compute_conf_frac(self, n_reporting_units: int, alpha: float) -> float:
        """
        Returns fraction of reporting units to be part of conformalization set
        """
        # what happens if this is negative, which it is alpha 0.95 and n_reporting_units is 38
        return round(min(1 + (alpha + 1) / (n_reporting_units * (alpha - 1)), 0.9), 2)

    def get_minimum_reporting_units(self, alpha: float) -> int:
        return math.ceil(-1 * (alpha + 1) / (alpha - 1))

    def _compute_population_correction(
        self, conformalization_data: pd.DataFrame, scores: pd.Series, correction_quantile: float, estimand: str
    ) -> float:
        """
        Compute population corrected conforalization correction.
        We care about larger units more than smaller units when computing aggregate prediction intervals.
        To accomplish this we weight the i-th conformalization score by the number of voters in that county the
        previous election
        """
        # calc weights
        weights = (
            conformalization_data[f"last_election_results_{estimand}"]
            / conformalization_data[f"last_election_results_{estimand}"].sum()
        )
        # sort scores and weights by scores
        population_correction = pd.DataFrame({"scores": scores, "weights": weights}).sort_values("scores")
        # percent of voters covered so far is cumulative sum of weights
        population_correction["percent"] = population_correction.weights.cumsum()
        # return minimum score such that the fraction of voters covered is larger than the correction quantile
        population_correction = population_correction.query("percent > @correction_quantile").reset_index(drop=True)
        population_correction = np.min(population_correction.scores)
        return population_correction

    def get_unit_prediction_intervals(
        self, reporting_units: pd.DataFrame, nonreporting_units: pd.DataFrame, alpha: float, estimand: str
    ) -> PredictionIntervals:
        """
        Get unit prediction intervals for non-parametric model. Adjust nonreporting unit prediction intervals based
        on conformalization.
        Returns upper/lower unit bounds and conformalization data (since we need that for aggregation)
        """
        conf_frac = self._compute_conf_frac(reporting_units.shape[0], alpha)
        # compute unadjusted upper/lower unit bounds and get conformalization data
        prediction_intervals = self.get_unit_prediction_interval_bounds(
            reporting_units, nonreporting_units, conf_frac, alpha, estimand
        )
        self.conformalization_data_unit = prediction_intervals.conformalization
        # compute conformity scores (e_j). This is how well the the lower/upper model cover the conformalization data.
        scores = np.maximum(
            prediction_intervals.conformalization.lower_bounds, prediction_intervals.conformalization.upper_bounds
        )

        # our desired coverage is: the alpha-% percentile of e_j than is less than zero
        # this is roughly equivalent to alpha-% of conformalization data that is covered by initial guess
        # to get there we need to add the correction, which is equal to alpha * (1 + 1 / conformalization_data.shape[0])
        correction_quantile = alpha * (1 + 1 / prediction_intervals.conformalization.shape[0])
        correction = np.quantile(scores, q=correction_quantile)

        # we care about larger units more than smaller units when computing aggregate
        # prediction intervals. To accomplish this, we will weight the i-th score by the
        # number of voters in that geographic unit in the previous election
        population_correction = self._compute_population_correction(
            prediction_intervals.conformalization, scores, correction_quantile, estimand
        )
        if self.robust:
            correction = max(correction, population_correction)  # if robust we take the larger of the two corrections
        else:
            correction = population_correction  # "unbiased" state prediction intervals

        # apply correction
        lower = prediction_intervals.lower - correction
        upper = prediction_intervals.upper + correction

        # save for later
        self.nonreporting_lower_bounds = lower
        self.nonreporting_upper_bounds = upper

        # un-normalize residuals
        lower *= nonreporting_units[f"last_election_results_{estimand}"]
        upper *= nonreporting_units[f"last_election_results_{estimand}"]

        # move from residual to vote space
        # max with nonreporting results so that bounds are at least as large as the # of votes seen
        lower = np.maximum(
            lower + nonreporting_units[f"last_election_results_{estimand}"], nonreporting_units[f"results_{estimand}"]
        )
        upper = np.maximum(
            upper + nonreporting_units[f"last_election_results_{estimand}"], nonreporting_units[f"results_{estimand}"]
        )

        return PredictionIntervals(
            lower.round(decimals=0), upper.round(decimals=0), prediction_intervals.conformalization
        )

    def get_all_conformalization_data_unit(self) -> tuple[None, pd.DataFrame]:
        """
        Returns the conformalization data for the unit adjustments
        """
        return None, self.conformalization_data_unit

    def get_all_conformalization_data_agg(self) -> tuple[None, pd.DataFrame]:
        """
        Returns the conformalization data for the aggregate adjustments
        """
        return None, self.conformalization_data_agg

    def get_aggregate_prediction_intervals(
        self,
        reporting_units: pd.DataFrame,
        nonreporting_units: pd.DataFrame,
        unexpected_units: pd.DataFrame,
        aggregate: list,
        alpha: float,
        unit_prediction_intervals: PredictionIntervals,
        estimand: str,
        **kwargs,
    ) -> PredictionIntervals:
        """
        Get aggregate prediction intervals. In the non-parametric case prediction intervals just sum.
        Compute results from reporting data and nonreporting data and then sum in the lower and upper prediction
        intervals from nonreporting data.
        """
        # get conformalization data out of unit prediction intervals
        conformalization_data = unit_prediction_intervals.conformalization

        # we're doing the same work as in get_aggregate_predictions here, can we just do this work once?
        aggregate_votes = self._get_reporting_aggregate_votes(reporting_units, unexpected_units, aggregate, estimand)

        lower_string = f"lower_{alpha}_{estimand}"
        upper_string = f"upper_{alpha}_{estimand}"

        # prediction intervals sum, kind of miraculous
        # Technically this is a conservative approach. This is equivalent to perfect correlation if
        # we assume that the prediction intervals are multivariate gaussian
        aggregate_prediction_intervals = (
            nonreporting_units.groupby(aggregate)
            .sum()
            .reset_index(drop=False)
            .rename(columns={lower_string: f"pi_lower_{estimand}", upper_string: f"pi_upper_{estimand}"})[
                aggregate + [f"pi_lower_{estimand}", f"pi_upper_{estimand}"]
            ]
        )

        # sum in prediction intervals and rename
        aggregate_data = (
            aggregate_votes.merge(aggregate_prediction_intervals, how="outer", on=aggregate)
            .fillna({f"results_{estimand}": 0, f"pi_lower_{estimand}": 0, f"pi_upper_{estimand}": 0})
            .assign(
                lower=lambda x: x[f"pi_lower_{estimand}"] + x[f"results_{estimand}"],
                upper=lambda x: x[f"pi_upper_{estimand}"] + x[f"results_{estimand}"],
            )
            .sort_values(aggregate)[aggregate + ["lower", "upper"]]
            .reset_index(drop=True)
        )

        self.conformalization_data_agg = conformalization_data
        return PredictionIntervals(aggregate_data.lower.round(decimals=0), aggregate_data.upper.round(decimals=0))
