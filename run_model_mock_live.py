#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:06:09 2022

@author: goldd
"""

import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv("~/.clokta/elections.env")  # noqa: E402
load_dotenv(find_dotenv())  # noqa: E402

from elexmodel.client import HistoricalModelClient, ModelClient  # noqa: E402
from elexmodel.handlers import s3  # noqa: E402
from elexmodel.handlers.data.LiveData import MockLiveDataHandler  # noqa: E402
from elexmodel.utils.file_utils import TARGET_BUCKET  # noqa: E402

os.environ["AWS_PROFILE"] = "elections"

election_id = "2020-11-03_USA_G"  # "
office_id = "P"
estimands = ["dem", "gop"]
geographic_unit_type = "county"
historical = False
unexpected_units = 0
prediction_intervals = [0.9]
percent_reporting_threshold = 100
percent_reporting = 20
print("target bucket", TARGET_BUCKET)
aggregates = ["unit", "postal_code"]
fixed_effects = ["postal_code"]

# agg_model params:
agg_model_estimates = True
agg_model_states_not_used = ["AK"]
ci_method = "percentile"  # percentile or normal_dist_mean (this is CI for mean)
num_observations = 1  # if set to 1, result is same as one batch of draws, not bootstrapped
nat_sum_data = pd.read_csv("data_for_agg_model/national_summary_votes_by_state.csv").drop("state", axis=1)
nat_sum_data_dict = dict(zip(nat_sum_data["postal_code"], nat_sum_data["vote"]))
# if using standard preprocessed data for current or historical election
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

model_client = ModelClient()

if not historical:
    result = model_client.get_estimates(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        pi_method="gaussian",
        aggregates=aggregates,
        agg_model_estimates=agg_model_estimates,
        agg_model_states_not_used=agg_model_states_not_used,
        ci_method=ci_method,
        num_observations=num_observations,
        nat_sum_data_dict=nat_sum_data_dict,
    )

    if agg_model_estimates:
        print("Agg results", model_client.get_national_summary_votes_estimates())


if historical:
    model_client = HistoricalModelClient()
    result = model_client.get_historical_evaluation(
        data,
        election_id,
        office_id,
        estimands,
        prediction_intervals,
        percent_reporting_threshold,
        geographic_unit_type,
        pi_method="gaussian",
        aggregates=aggregates,
        fixed_effects=fixed_effects,
        features=["age_18_to_29", "age_over_65", "median_household_income"],
    )

for aggregate_level, estimates in result.items():
    print(aggregate_level, "\n", estimates, "\n")
