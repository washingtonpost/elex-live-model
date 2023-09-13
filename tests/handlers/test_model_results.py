import pandas as pd

from elexmodel.handlers.data.ModelResults import ModelResultsHandler
from elexmodel.models.BaseElectionModel import PredictionIntervals

reporting = pd.DataFrame(
    {
        "geographic_unit_fips": ["a", "b"],
        "postal_code": ["AB", "AB"],
        "district": ["9", "8"],
        "results_e1": [1000, 2000],
        "results_e2": [400, 1200],
        "reporting": [1, 1],
    }
)
nonreporting = pd.DataFrame({"geographic_unit_fips": ["c"], "postal_code": ["AB"], "district": ["8"], "reporting": [0]})
notexpected = pd.DataFrame(
    {
        "geographic_unit_fips": ["d"],
        "postal_code": ["AB"],
        "district": ["9"],
        "results_e1": [0],
        "results_e2": [0],
        "reporting": [0],
    }
)
predictions_e1 = [1200]
predictions_e2 = [500]
intervals_e1 = {0.7: PredictionIntervals([1000], [1500]), 0.9: PredictionIntervals([800], [1800])}
intervals_e2 = {0.7: PredictionIntervals([400], [600]), 0.9: PredictionIntervals([300], [750])}

agg1_e1 = pd.DataFrame(
    {
        "district": [9, 8],
        "geographic_unit_fips": ["e", "f"],
        "postal_code": ["AB", "AB"],
        "reporting": [1, 0],
        "pred_e1": [1000, 3200],
    }
)
agg2_e1 = pd.DataFrame({"geographic_unit_fips": ["g"], "postal_code": ["AB"], "reporting": [0], "pred_e1": [4200]})

agg1_e2 = pd.DataFrame(
    {
        "district": [9, 8],
        "geographic_unit_fips": ["e", "f"],
        "postal_code": ["AB", "AB"],
        "reporting": [1, 0],
        "pred_e2": [400, 1700],
    }
)
agg2_e2 = pd.DataFrame({"geographic_unit_fips": ["g"], "postal_code": ["AB"], "reporting": [0], "pred_e2": [2100]})
intervals1_e1 = {
    0.7: PredictionIntervals([1000, 2900], [1000, 3600]),
    0.9: PredictionIntervals([1000, 2600], [1000, 4000]),
}
intervals2_e1 = {0.7: PredictionIntervals([3900], [4600]), 0.9: PredictionIntervals([3600], [5000])}
intervals1_e2 = {0.7: PredictionIntervals([400, 1600], [400, 1800]), 0.9: PredictionIntervals([400, 1500], [400, 2000])}
intervals2_e2 = {0.7: PredictionIntervals([2000], [2200]), 0.9: PredictionIntervals([1800], [2400])}


def test_model_results_handler():
    # test unit predictions/intervals methods for two estimands
    handler = ModelResultsHandler(["district", "unit", "postal_code"], [0.7, 0.9], reporting, nonreporting, notexpected)
    handler.add_unit_predictions("e1", predictions_e1)
    handler.add_unit_intervals("e1", intervals_e1)
    handler.add_unit_predictions("e2", predictions_e2)
    handler.add_unit_intervals("e2", intervals_e2)
    expected_cols = [
        "pred_e1",
        "pred_e2",
        "lower_0.7_e1",
        "lower_0.9_e1",
        "upper_0.7_e1",
        "upper_0.9_e1",
        "lower_0.7_e1",
        "lower_0.9_e1",
        "upper_0.7_e1",
        "upper_0.9_e1",
    ]
    for df in [handler.reporting_units, handler.nonreporting_units, handler.unexpected_units]:
        assert set(expected_cols).issubset(set(df.columns))

    # test agg predictions for 2 aggs and 2 estimands
    handler.add_agg_predictions("e1", "district", agg1_e1, intervals1_e1)
    handler.add_agg_predictions("e1", "postal_code", agg2_e1, intervals2_e1)
    handler.add_agg_predictions("e2", "district", agg1_e2, intervals1_e2)
    handler.add_agg_predictions("e2", "postal_code", agg2_e2, intervals2_e2)

    assert len(handler.estimates["postal_code"]) == 2
    assert len(handler.estimates["district"]) == 2
    expected_cols_agg = ["pred", "lower_0.7", "lower_0.9", "upper_0.7", "upper_0.9"]
    for v in handler.estimates.values():
        for i, df in enumerate(v):
            expected_cols = [f"{x}_e{i+1}" for x in expected_cols_agg]
            assert set(expected_cols).issubset(set(df.columns))

    # test preparation of final results data
    handler.process_final_results()

    assert set(handler.final_results.keys()) == set(["unit_data", "state_data", "district_data"])
    assert len(handler.final_results["unit_data"]) == 4
    assert len(handler.final_results["state_data"]) == 1
    assert len(handler.final_results["district_data"]) == 2


def test_no_unit_data():
    handler = ModelResultsHandler(["postal_code"], [0.7, 0.9], reporting, nonreporting, notexpected)
    handler.add_unit_predictions("e1", predictions_e1)
    handler.add_unit_intervals("e1", intervals_e1)

    handler.add_agg_predictions("e1", "postal_code", agg1_e1, intervals1_e1)
    handler.process_final_results()

    assert "unit_data" not in handler.final_results
