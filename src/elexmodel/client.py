import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from elexmodel.handlers import s3
from elexmodel.handlers.config import ConfigHandler
from elexmodel.handlers.data.CombinedData import CombinedDataHandler
from elexmodel.handlers.data.ModelResults import ModelResultsHandler
from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler
from elexmodel.logging import initialize_logging
from elexmodel.models.BootstrapElectionModel import BootstrapElectionModel
from elexmodel.models.ConformalElectionModel import ConformalElectionModel
from elexmodel.models.GaussianElectionModel import GaussianElectionModel
from elexmodel.models.NonparametricElectionModel import NonparametricElectionModel
from elexmodel.utils.constants import AGGREGATE_ORDER, DEFAULT_AGGREGATES, VALID_AGGREGATES_MAPPING
from elexmodel.utils.file_utils import APP_ENV, S3_FILE_PATH, TARGET_BUCKET
from elexmodel.utils.math_utils import compute_error, compute_frac_within_pi, compute_mean_pi_length

initialize_logging()

LOG = logging.getLogger(__name__)


class ModelClientException(Exception):
    pass


class ModelNotEnoughSubunitsException(ModelClientException):
    pass


class ModelClient:
    """
    Client for generating vote estimates
    """

    def __init__(self):
        super().__init__()
        self.all_conformalization_data_unit_dict = defaultdict(dict)
        self.all_conformalization_data_agg_dict = defaultdict(dict)
        self.model = None
        self.results_handler = None
        self.election_id = None
        self.office = None
        self.geographic_unit_type = None
        self.save_results = None

    def _check_input_parameters(
        self,
        config_handler,
        office,
        estimands,
        geographic_unit_type,
        features,
        aggregates,
        fixed_effects,
        pi_method,
        model_parameters,
        handle_unreporting,
    ):
        offices = config_handler.get_offices()
        if office not in offices:
            raise ValueError(f"Office '{office}' is not valid. Please check config.")

        estimands_in_config = config_handler.get_estimands(office)
        extra_estimands = set(estimands).difference(set(estimands_in_config))
        if len(extra_estimands) > 0:
            # this is ok; later they'll all be created by the Estimandizer
            LOG.info("Found additional estimands that were not specified in the config file: %s", extra_estimands)

        geographic_unit_types = config_handler.get_geographic_unit_types(office)
        if geographic_unit_type not in geographic_unit_types:
            raise ValueError(f"Geographic unit type: '{geographic_unit_type}' is not valid. Please check config")

        model_features = config_handler.get_features(office)
        invalid_features = list(set(features).difference(set(model_features)))
        if len(invalid_features) > 0:
            raise ValueError(f"Feature(s): {invalid_features} not valid. Please check config")

        model_aggregates = config_handler.get_aggregates(office)
        invalid_aggregates = [aggregate for aggregate in aggregates if aggregate not in model_aggregates]
        if len(invalid_aggregates) > 0:
            raise ValueError(f"Aggregate(s): {invalid_aggregates} not valid. Please check config")

        model_fixed_effects = config_handler.get_fixed_effects(office)
        if isinstance(fixed_effects, dict):
            invalid_fixed_effects = [
                fixed_effect for fixed_effect in fixed_effects.keys() if fixed_effect not in model_fixed_effects
            ]
        else:
            invalid_fixed_effects = [
                fixed_effect for fixed_effect in fixed_effects if fixed_effect not in model_fixed_effects
            ]
        if len(invalid_fixed_effects) > 0:
            raise ValueError(f"Fixed effect(s): {invalid_fixed_effects} not valid. Please check config")
        if pi_method not in {"gaussian", "nonparametric", "bootstrap"}:
            raise ValueError(
                f"Prediction interval method: {pi_method} is not valid. \
                    pi_method has to be either `gaussian` or `nonparametric`."
            )

        if not isinstance(model_parameters, dict):
            raise ValueError("model_paramters is not valid. Has to be a dict.")
        if len(model_parameters) > 0:
            if "lambda_" in model_parameters and (
                not isinstance(model_parameters["lambda_"], (float, int)) or model_parameters["lambda_"] < 0
            ):
                raise ValueError("lambda is not valid. It has to be numeric and greater than zero.")
            if "turnout_factor_lower" in model_parameters and not isinstance(
                model_parameters["turnout_factor_lower"], (float, int)
            ):
                raise ValueError("turnout_factor_lower is not valid. Has to be a float.")
            if "turnout_factor_upper" in model_parameters and not isinstance(
                model_parameters["turnout_factor_upper"], (float, int)
            ):
                raise ValueError("turnout_factor_upper is not valid. Has to be a float.")
            if pi_method == "gaussian":
                if "beta" in model_parameters and not isinstance(model_parameters["beta"], (int, float)):
                    raise ValueError("beta is not valid. Has to be either an integer or a float.")
                if "winsorize" in model_parameters and not isinstance(model_parameters["winsorize"], bool):
                    raise ValueError("winsorize is not valid. Has to be an boolean.")
            elif pi_method == "nonparametric":
                if "robust" in model_parameters and not isinstance(model_parameters["robust"], bool):
                    raise ValueError("robust is not valid. Has to be a boolean.")
            elif pi_method == "bootstrap":
                if "B" in model_parameters and not isinstance(model_parameters["B"], int):
                    raise ValueError("B is not valid. Has to be either an integer.")
                if "T" in model_parameters and not isinstance(model_parameters["T"], (int, float)):
                    raise ValueError("T is not valid. Has to be either an integer or a float.")
                if "strata" in model_parameters and not isinstance(model_parameters["strata"], list):
                    raise ValueError("strata is not valid. Has to be a list.")
                if "agg_model_hard_threshold" in model_parameters and not isinstance(
                    model_parameters["agg_model_hard_threshold"], bool
                ):
                    raise ValueError("agg_model_hard_threshold is not valid. Has to be a boolean.")
                if "y_LB" in model_parameters and not isinstance(model_parameters["y_LB"], (float, int)):
                    raise ValueError("y_LB is not valid. Has to be a float.")
                if "y_UB" in model_parameters and not isinstance(model_parameters["y_UB"], (float, int)):
                    raise ValueError("y_UB is not valid. Has to be a float.")
                if "z_LB" in model_parameters and not isinstance(model_parameters["z_LB"], (float, int)):
                    raise ValueError("z_LB is not valid. Has to be a float.")
                if "z_UB" in model_parameters and not isinstance(model_parameters["z_UB"], (float, int)):
                    raise ValueError("z_UB is not valid. Has to be a float.")
                if "y_unobserved_upper_bound" in model_parameters and not isinstance(
                    model_parameters["y_unobserved_upper_bound"], (float, int)
                ):
                    raise ValueError("y_unobserved_upper_bound is not valid. Has to be a float.")
                if "y_unobserved_lower_bound" in model_parameters and not isinstance(
                    model_parameters["y_unobserved_lower_bound"], (float, int)
                ):
                    raise ValueError("y_unobserved_lower_bound is not valid. Has to be a float.")
                if "percent_expected_vote_error_bound" in model_parameters and not isinstance(
                    model_parameters["percent_expected_vote_error_bound"], (float, int)
                ):
                    raise ValueError("z_UB is not valid. Has to be a float.")
                if "z_unobserved_upper_bound" in model_parameters and not isinstance(
                    model_parameters["z_unobserved_upper_bound"], (float, int)
                ):
                    raise ValueError("z_unobserved_upper_bound is not valid. Has to be a float.")
                if "z_unobserved_lower_bound" in model_parameters and not isinstance(
                    model_parameters["z_unobserved_lower_bound"], (float, int)
                ):
                    raise ValueError("z_unobserved_lower_bound is not valid. Has to be a float.")
        if handle_unreporting not in {"drop", "zero"}:
            raise ValueError("handle_unreporting must be either `drop` or `zero`")

        return True

    def get_aggregate_list(self, office, aggregate):
        default_aggregate = DEFAULT_AGGREGATES[office]
        base_aggregate = default_aggregate[:-1]  # remove unit
        raw_aggregate_list = base_aggregate + [aggregate]
        return sorted(list(set(raw_aggregate_list)), key=lambda x: AGGREGATE_ORDER.index(x))

    def get_national_summary_votes_estimates(self, nat_sum_data_dict=None, base_to_add=0, alphas=[0.99]):
        if self.model is None:
            raise ModelClientException(
                "Must call the get_estimands() method before get_national_summary_votes_estimates()."
            )

        nat_sum_estimates_dict = {}
        for alpha in alphas:
            nat_sum_estimates = self.model.get_national_summary_estimates(nat_sum_data_dict, base_to_add, alpha)
            nat_sum_estimates_dict[alpha] = nat_sum_estimates
        self.results_handler.add_national_summary_estimates(nat_sum_estimates_dict)

        if APP_ENV != "local" and self.save_results:
            self.results_handler.write_data(
                self.election_id, self.office, self.geographic_unit_type, keys=["nat_sum_data"]
            )
        return self.results_handler.final_results["nat_sum_data"]

    def get_estimates(
        self,
        current_data,  # list of lists
        election_id,
        office,
        estimands,
        prediction_intervals=[0.7, 0.9],
        percent_reporting_threshold=100,
        geographic_unit_type="county",
        raw_config=None,
        preprocessed_data=None,
        model_parameters={},
        **kwargs,
    ):
        """
        Get model estimate for one election, office and estimand.
        This function assumes that election_id is valid and in the format <date>_<state_postal>_<race_type>
        """
        LOG.info("Getting estimates: %s, %s, %s", election_id, office, estimands)
        # If current_data isn't already a dataframe, convert to df
        if not isinstance(current_data, pd.DataFrame):
            # First element of current_data is list of column values
            column_values = current_data[0]
            current_data = pd.DataFrame(current_data[1:], columns=column_values)
        features = kwargs.get("features", [])
        aggregates = kwargs.get("aggregates", DEFAULT_AGGREGATES[office])
        fixed_effects = kwargs.get("fixed_effects", {})
        pi_method = kwargs.get("pi_method", "nonparametric")
        lhs_called_contests = kwargs.get("lhs_called_contests", [])
        rhs_called_contests = kwargs.get("rhs_called_contests", [])
        stop_model_call = kwargs.get("stop_model_call", [])
        save_output = kwargs.get("save_output", ["results"])
        self.save_results = "results" in save_output
        save_data = "data" in save_output
        save_config = "config" in save_output
        # saving conformalization data only makes sense if a ConformalElectionModel is used
        save_conformalization = "conformalization" in save_output
        handle_unreporting = kwargs.get("handle_unreporting", "drop")

        district_election = False
        if office in {"H", "Y", "Z"}:
            district_election = True

        model_settings = {
            "election_id": election_id,
            "office": office,
            "geographic_unit_type": geographic_unit_type,
            "district_election": district_election,
            "features": features,
            "fixed_effects": fixed_effects,
            "save_conformalization": save_conformalization,
        }
        model_settings.update(model_parameters)

        LOG.info("Getting config: %s", election_id)
        config_handler = ConfigHandler(
            election_id, config=raw_config, s3_client=s3.S3JsonUtil(TARGET_BUCKET), save=save_config
        )

        self._check_input_parameters(
            config_handler,
            office,
            estimands,
            geographic_unit_type,
            features,
            aggregates,
            fixed_effects,
            pi_method,
            model_parameters,
            handle_unreporting,
        )
        self.election_id = election_id
        self.office = office
        self.geographic_unit_type = geographic_unit_type

        states_with_election = config_handler.get_states(self.office)
        estimand_baselines = config_handler.get_estimand_baselines(self.office, estimands)

        LOG.info("Getting preprocessed data: %s", self.election_id)
        preprocessed_data_handler = PreprocessedDataHandler(
            self.election_id,
            self.office,
            self.geographic_unit_type,
            estimands,
            estimand_baselines,
            data=preprocessed_data,
            s3_client=s3.S3CsvUtil(TARGET_BUCKET),
        )
        preprocessed_data_handler.data = preprocessed_data_handler.select_rows_in_states(
            preprocessed_data_handler.data, states_with_election
        )
        preprocessed_data = preprocessed_data_handler.data
        if save_data:
            preprocessed_data_handler.save_data(preprocessed_data)

        LOG.info("Getting combined data for requested estimands")
        data = CombinedDataHandler(
            preprocessed_data,
            current_data,
            estimands,
            self.geographic_unit_type,
            handle_unreporting=handle_unreporting,
        )

        turnout_factor_lower = model_parameters.get("turnout_factor_lower", 0.2)
        turnout_factor_upper = model_parameters.get("turnout_factor_upper", 2.5)

        (reporting_units, nonreporting_units, unexpected_units) = data.get_units(
            percent_reporting_threshold, turnout_factor_lower, turnout_factor_upper, aggregates
        )

        LOG.info(
            "Model parameters: \n prediction intervals: %s, percent reporting threshold: %s, \
                pi_method: %s, aggregates: %s, model settings: %s",
            prediction_intervals,
            percent_reporting_threshold,
            pi_method,
            aggregates,
            model_settings,
        )

        if pi_method == "nonparametric":
            self.model = NonparametricElectionModel(model_settings=model_settings)
        elif pi_method == "gaussian":
            self.model = GaussianElectionModel(model_settings=model_settings)
        elif pi_method == "bootstrap":
            self.model = BootstrapElectionModel(model_settings=model_settings)

        minimum_reporting_units_max = 0
        for alpha in prediction_intervals:
            minimum_reporting_units = self.model.get_minimum_reporting_units(alpha)
            if minimum_reporting_units > minimum_reporting_units_max:
                minimum_reporting_units_max = minimum_reporting_units

        if APP_ENV != "local" and self.save_results:
            data.write_data(self.election_id, self.office)

        non_modeled_units = unexpected_units[unexpected_units["unit_category"] == "non-modeled"]
        n_reporting_expected_units = reporting_units.shape[0]
        n_unexpected_units = len(unexpected_units[unexpected_units["unit_category"] == "unexpected"])
        n_nonreporting_units = nonreporting_units.shape[0]
        n_non_modeled_units = len(non_modeled_units)
        LOG.info(
            f"""Running model
            There are {n_reporting_expected_units} reporting and expected units.
            There are {n_unexpected_units} unexpected units.
            There are {n_non_modeled_units} non-modeled units.
            There are {n_nonreporting_units} nonreporting units."""
        )
        if len(non_modeled_units) > 0:
            non_modeled_units = non_modeled_units.groupby("postal_code")["geographic_unit_fips"].apply(list).to_dict()
            LOG.info(f"non-modeled units:\n{non_modeled_units}")

        if n_reporting_expected_units < minimum_reporting_units_max:
            raise ModelNotEnoughSubunitsException(
                f"Currently {n_reporting_expected_units} reporting, need at least {minimum_reporting_units_max}"
            )

        units_by_count = reporting_units["geographic_unit_fips"].value_counts()
        duplicate_units = units_by_count[units_by_count > 1].to_dict()
        if len(duplicate_units) > 0:
            raise ModelClientException(f"At least one unit appears twice: {duplicate_units}")

        self.results_handler = ModelResultsHandler(
            aggregates, prediction_intervals, reporting_units, nonreporting_units, unexpected_units
        )

        for estimand in estimands:
            unit_predictions, unit_turnout_predictions = self.model.get_unit_predictions(
                reporting_units, nonreporting_units, estimand, unexpected_units=unexpected_units
            )
            self.results_handler.add_unit_predictions(estimand, unit_predictions)
            if unit_turnout_predictions is not None:
                self.results_handler.add_unit_turnout_predictions(unit_turnout_predictions)

            # gets prediciton intervals for each alpha
            alpha_to_unit_prediction_intervals = {}
            for alpha in prediction_intervals:
                alpha_to_unit_prediction_intervals[alpha] = self.model.get_unit_prediction_intervals(
                    self.results_handler.reporting_units, self.results_handler.nonreporting_units, alpha, estimand
                )
                if isinstance(self.model, ConformalElectionModel):
                    self.all_conformalization_data_unit_dict[alpha][
                        estimand
                    ] = self.model.get_all_conformalization_data_unit()

            self.results_handler.add_unit_intervals(estimand, alpha_to_unit_prediction_intervals)

            for aggregate in self.results_handler.aggregates:
                aggregate_list = self.get_aggregate_list(self.office, aggregate)
                estimates_df = self.model.get_aggregate_predictions(
                    self.results_handler.reporting_units,
                    self.results_handler.nonreporting_units,
                    self.results_handler.unexpected_units,
                    aggregate_list,
                    estimand,
                    lhs_called_contests=lhs_called_contests,
                    rhs_called_contests=rhs_called_contests,
                )
                alpha_to_agg_prediction_intervals = {}
                for alpha in prediction_intervals:
                    alpha_to_agg_prediction_intervals[alpha] = self.model.get_aggregate_prediction_intervals(
                        self.results_handler.reporting_units,
                        self.results_handler.nonreporting_units,
                        self.results_handler.unexpected_units,
                        aggregate_list,
                        alpha,
                        alpha_to_unit_prediction_intervals[alpha],
                        estimand,
                        lhs_called_contests=lhs_called_contests,
                        rhs_called_contests=rhs_called_contests,
                        stop_model_call=stop_model_call,
                    )
                    if isinstance(self.model, ConformalElectionModel):
                        self.all_conformalization_data_agg_dict[alpha][
                            estimand
                        ] = self.model.get_all_conformalization_data_agg()

                # get all of the prediction intervals here
                self.results_handler.add_agg_predictions(
                    estimand, aggregate, estimates_df, alpha_to_agg_prediction_intervals
                )

        self.results_handler.process_final_results()

        if APP_ENV != "local" and self.save_results:
            self.results_handler.write_data(self.election_id, self.office, self.geographic_unit_type)

        return self.results_handler.final_results


class HistoricalModelClient(ModelClient):
    def __init__(self):
        super().__init__()

    def get_historical_evaluation(
        self,
        current_data,
        election_id,
        office,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        **kwargs,
    ):
        config_handler = ConfigHandler(election_id, s3_client=s3.S3JsonUtil(TARGET_BUCKET))
        historical_election_ids = config_handler.get_historical_election_ids(office)
        estimand_baselines = config_handler.get_estimand_baselines(office, estimands)
        if len(historical_election_ids) == 0:
            raise ModelClientException("No historical elections prepared")
        historical_evaluation_and_estimates = {}
        self.aggregates = list(set(kwargs.get("aggregates", []) + ["unit"]))
        kwargs["aggregates"] = self.aggregates
        for historical_election_id in historical_election_ids:
            historical_current_data, preprocessed_data = self._format_historical_current_data(
                current_data,
                historical_election_id,
                office,
                geographic_unit_type,
                estimands,
                estimand_baselines,
                percent_reporting_threshold,
            )
            historical_estimates = self.get_estimates(
                historical_current_data,
                historical_election_id,
                office,
                estimands,
                prediction_intervals,
                percent_reporting_threshold,
                geographic_unit_type,
                **kwargs,
            )
            historical_evaluation_and_estimates[historical_election_id] = {
                "evaluation": self.evaluate_historical_estimates(
                    historical_estimates, preprocessed_data, self.aggregates, prediction_intervals, estimands
                ),
                "estimates": historical_estimates,
            }
        save_output = kwargs.get("save_output", False)
        if APP_ENV != "local" and "results" in save_output:
            self._write_evaluation(
                historical_evaluation_and_estimates, election_id, office, geographic_unit_type, estimands
            )
        return historical_evaluation_and_estimates

    def _format_historical_current_data(
        self,
        current_data,
        historical_election_id,
        office,
        geographic_unit_type,
        estimands,
        estimand_baselines,
        percent_reporting_threshold,
    ):
        """
        Formats data for historical model run
        """

        # What does the historical model client do?
        #     - If we are running the election in 2024 and 100 counties are reporting, we want to see what
        #         our model error would have been in 2020 with these counties reporting
        #     - To do that we need to merge the 2020 results onto the 2024 reporting counties

        #     - So for 2020 (cli) this means -> we have 2020 data and we pick 100 random counties reporting
        #       in the MockLiveDataHandler
        #     - in this function we get the 2016 results and merge that to the 100 reporting counties in 2020

        # running election id: 2020-11-03_USA_G --historical
        #     -> historical election id: 2016-11-08_USA_G, 2012, ...

        formatted_data = current_data[["postal_code", "geographic_unit_fips", "percent_expected_vote"]]
        LOG.info(f"Getting data for historical election: {historical_election_id}")
        preprocessed_data_handler = PreprocessedDataHandler(
            historical_election_id,
            office,
            geographic_unit_type,
            estimands,
            estimand_baselines,
            s3_client=s3.S3CsvUtil(TARGET_BUCKET),
            historical=True,
            include_results_estimand=True,
        )
        # we always want to pass turnout so that we can generate results weights
        results_to_return = list(set([f"results_{estimand}" for estimand in estimands] + ["results_turnout"]))
        geo_columns = set(["geographic_unit_fips", "postal_code"] + [a for a in self.aggregates if a != "unit"])
        preprocessed_data = preprocessed_data_handler.data[list(geo_columns) + results_to_return].copy()
        historical_current_data = preprocessed_data.merge(formatted_data, on=["postal_code", "geographic_unit_fips"])
        for estimand in estimands:
            column_name = f"results_{estimand}"
            historical_current_data = historical_current_data.assign(
                **{
                    column_name: lambda x: np.where(
                        x.percent_expected_vote >= percent_reporting_threshold, x[column_name], 0
                    )
                }
            )
        historical_current_data = historical_current_data[
            ["postal_code", "geographic_unit_fips", "percent_expected_vote"] + results_to_return
        ].copy()
        return historical_current_data, preprocessed_data

    def compute_evaluation(self, historical_estimates, results, merge_on, group_by, prediction_intervals, estimand):
        """
        This function compute the actual evaluation for one set of model outputs and all prediction intervals.
        merge_on has to be the same as the aggregate, group_by can be the aggregate or a lambda returning true
        lambda x: True, to create one group
        """
        intermed = historical_estimates.merge(results, on=merge_on).groupby(group_by)
        error_df = intermed.apply(
            lambda x: pd.Series(
                {
                    f"mae_{estimand}": compute_error(x[f"raw_results_{estimand}"], x[f"pred_{estimand}"], type_="mae"),
                    f"mape_{estimand}": compute_error(
                        x[f"raw_results_{estimand}"], x[f"pred_{estimand}"], type_="mape"
                    ),
                }
            ),
            include_groups=False,
        )

        for alpha in prediction_intervals:
            lower_string = f"lower_{alpha}_{estimand}"
            upper_string = f"upper_{alpha}_{estimand}"
            alpha_df = intermed.apply(
                lambda x: pd.Series(
                    {
                        f"frac_within_pi_{alpha}_{estimand}": compute_frac_within_pi(
                            x[lower_string], x[upper_string], x[f"raw_results_{estimand}"]
                        ),
                        f"mean_pi_length_{alpha}_{estimand}": compute_mean_pi_length(
                            x[lower_string], x[upper_string], x[f"raw_results_{estimand}"]
                        ),
                    }
                ),
                include_groups=False,
            )
            error_df = error_df.merge(alpha_df, left_index=True, right_index=True)

        return error_df.to_dict(orient="index")

    def evaluate_historical_estimates(
        self, historical_estimates, preprocessed_data, aggregates, prediction_intervals, estimands
    ):
        """
        This function evaluates the historical model.
        It evaluates over unit data and all aggregates and prediction interval levels. note that it doesn't
        differentiate between reported and non-reporting units so those metrics will be inflated
        """
        all_evaluations = {}
        # necessary because we don't want to edit self.aggregates because we need
        # to keep unit there if there are more races to do estimates for
        aggregates = [a for a in aggregates if a != "unit"]
        for estimand in estimands:
            results_unit = preprocessed_data[
                list(set(aggregates + ["postal_code", "geographic_unit_fips", f"results_{estimand}"]))
            ].rename(columns={f"results_{estimand}": f"raw_results_{estimand}"})
            evaluation = {}
            evaluation["unit_data"] = self.compute_evaluation(
                historical_estimates["unit_data"],
                results_unit,
                ["postal_code", "geographic_unit_fips"],
                ["postal_code"],
                prediction_intervals,
                estimand,
            )
            evaluation["unit_data"]["all"] = self.compute_evaluation(
                historical_estimates["unit_data"],
                results_unit,
                ["postal_code", "geographic_unit_fips"],
                lambda x: True,
                prediction_intervals,
                estimand,
            )[True]

            for aggregate in aggregates:
                aggregate_label = VALID_AGGREGATES_MAPPING.get(aggregate)
                aggregate_list = sorted(list(set(["postal_code", aggregate])), key=lambda x: AGGREGATE_ORDER.index(x))
                results_aggregate = results_unit.groupby(aggregate_list).sum().reset_index(drop=False)
                evaluation[aggregate_label] = self.compute_evaluation(
                    historical_estimates[aggregate_label],
                    results_aggregate,
                    aggregate_list,
                    aggregate_list,
                    prediction_intervals,
                    estimand,
                )
                evaluation[aggregate_label]["all"] = self.compute_evaluation(
                    historical_estimates[aggregate_label],
                    results_aggregate,
                    aggregate_list,
                    lambda x: True,
                    prediction_intervals,
                    estimand,
                )[True]
            all_evaluations[estimand] = evaluation

        return all_evaluations

    def _write_evaluation(self, evaluation, election_id, office, geographic_unit_type, estimand):
        s3_client = s3.S3JsonUtil(TARGET_BUCKET)
        path = f"{S3_FILE_PATH}/{election_id}/evaluation/{office}/{geographic_unit_type}/{estimand}/current.json"
        s3_client.put(path, evaluation)
