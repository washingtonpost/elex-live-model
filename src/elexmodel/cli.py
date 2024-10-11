import ast
import json

import click
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
if len(dotenv_path.strip()) == 0:
    dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

from elexmodel.client import HistoricalModelClient, ModelClient  # noqa: E402
from elexmodel.handlers import s3  # noqa: E402
from elexmodel.handlers.data.LiveData import MockLiveDataHandler  # noqa: E402
from elexmodel.utils.constants import VALID_AGGREGATES_MAPPING  # noqa: E402
from elexmodel.utils.file_utils import TARGET_BUCKET  # noqa: E402


class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except ValueError as e:
            raise click.BadParameter(value) from e


@click.command()
@click.argument("election_id")
@click.option("--estimands", "estimands", default=["turnout"], multiple=True)
@click.option("--office_id", "office_id")
@click.option("--fixed_effects", "fixed_effects", default={})
@click.option("--features", default=[], multiple=True)
@click.option("--aggregates", multiple=True)
@click.option(
    "--pi_method",
    "pi_method",
    default="nonparametric",
    type=click.Choice(["gaussian", "nonparametric", "bootstrap"]),
)
@click.option("--prediction_intervals", "prediction_intervals", default=[0.7, 0.9], multiple=True)
@click.option("--percent_reporting_threshold", "percent_reporting_threshold", default=100)
@click.option(
    "--geographic_unit_type",
    "geographic_unit_type",
    default="county",
    type=click.Choice(["county", "precinct", "county-district", "precinct-district"]),
)
@click.option(
    "--model_parameters",
    "model_parameters",
    default="{}",
    cls=PythonLiteralOption,
    help="A dictionary of model parameters",
)
@click.option(
    "--lhs_called_contests",
    "lhs_called_contests",
    help="contests called for the lhs party (ie. the party for which margin predictions > 0 are winners)",
    default=None,
    multiple=True,
)
@click.option(
    "--rhs_called_contests",
    "rhs_called_contests",
    help="contests called for the rhs party (ie. the party for which margin predictions < 0 are winners)",
    default=None,
    multiple=True,
)
@click.option(
    "--stop_model_call",
    "stop_model_call",
    default=None,
    multiple=True,
    help="contests for which we don't allow model calls",
)
@click.option(
    "--percent_reporting",
    "percent_reporting",
    default=100,
    type=click.IntRange(min=0, max=100),
    help="percent of units reporting. For testing purposes.",
)
@click.option(
    "--unexpected_units",
    "unexpected_units",
    default=0,
    type=int,
    help="number of reporting unexpected units to include. for testing purposes - does not work for historical runs",
)
@click.option("--historical", "historical", is_flag=True, help="run historical election")
@click.option(
    "--save_output",
    "save_output",
    default=[],
    multiple=True,
    type=click.Choice(["results", "data", "config", "conformalization"]),
    help="options: results, data, config",
)
@click.option("--handle_unreporting", "handle_unreporting", default="drop", type=click.Choice(["drop", "zero"]))
@click.option(
    "--national_summary",
    "national_summary",
    is_flag=True,
    help="When not running a historical election, output results aggregated to the national level.",
)
def cli(
    election_id, estimands, office_id, prediction_intervals, percent_reporting_threshold, geographic_unit_type, **kwargs
):
    """
    This tool accepts an election ID (e.g. "2021-11-02_VA_G) and the options below and outputs formatted model data.
    """
    # Read data
    estimands = list(estimands)
    historical = kwargs["historical"]
    percent_reporting = kwargs["percent_reporting"]
    unexpected_units = kwargs["unexpected_units"]

    kwargs["features"] = list(kwargs["features"])
    kwargs["aggregates"] = list(kwargs["aggregates"])
    if len(kwargs["aggregates"]) == 0:
        del kwargs["aggregates"]
    try:
        kwargs["fixed_effects"] = json.loads(kwargs["fixed_effects"])
    except json.decoder.JSONDecodeError:
        kwargs["fixed_effects"] = {kwargs["fixed_effects"]: ["all"]}

    prediction_intervals = list(prediction_intervals)

    # Read data
    data_handler = MockLiveDataHandler(
        election_id,
        office_id,
        geographic_unit_type,
        estimands,
        historical=historical,
        unexpected_units=unexpected_units,
        s3_client=s3.S3CsvUtil(TARGET_BUCKET),
    )

    data_handler.shuffle()
    data = data_handler.get_percent_fully_reported(percent_reporting)

    # Format arguments for get_estimates function
    if historical:
        model_client = HistoricalModelClient()
        historical_id_to_result = model_client.get_historical_evaluation(
            data,
            election_id,
            office_id,
            estimands,
            prediction_intervals,
            percent_reporting_threshold,
            geographic_unit_type,
            **kwargs
        )
        for historical_election_id, result in historical_id_to_result.items():
            for estimand in estimands:
                print("estimand: ", estimand, "\n")
                for aggregate in kwargs["aggregates"]:
                    print(aggregate)
                    print(
                        (
                            historical_id_to_result[historical_election_id]["evaluation"][estimand][
                                VALID_AGGREGATES_MAPPING.get(aggregate)
                            ]
                        ),
                        "\n",
                    )
                    print("unit_data")
                    print(historical_id_to_result[historical_election_id]["evaluation"][estimand]["unit_data"], "\n")
            if "state_data" in historical_id_to_result[historical_election_id]["estimates"]:
                print("state estimates")
                print(historical_id_to_result[historical_election_id]["estimates"]["state_data"])
    else:
        model_client = ModelClient()
        result = model_client.get_estimates(
            data,
            election_id,
            office_id,
            estimands,
            prediction_intervals,
            percent_reporting_threshold,
            geographic_unit_type,
            **kwargs
        )

        if kwargs.get("national_summary", False):
            # TODO: get_national_summary_votes_estimates() arguments via CLI
            model_client.get_national_summary_votes_estimates(None, 0, [0.99])

        for aggregate_level, estimates in result.items():
            print(aggregate_level, "\n", estimates, "\n")
