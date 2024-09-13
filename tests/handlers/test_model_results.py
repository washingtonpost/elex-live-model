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
    handler = ModelResultsHandler(
        ["district", "unit", "postal_code"], [0.7, 0.9], reporting.copy(), nonreporting.copy(), notexpected.copy()
    )
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
            expected_cols = [f"{x}_e{i+1}" for x in expected_cols_agg]  # noqa: E226
            assert set(expected_cols).issubset(set(df.columns))

    # test preparation of final results data
    handler.process_final_results()

    assert set(handler.final_results.keys()) == set(["unit_data", "state_data", "district_data"])
    assert len(handler.final_results["unit_data"]) == 4
    assert len(handler.final_results["state_data"]) == 1
    assert len(handler.final_results["district_data"]) == 2


def test_no_unit_data():
    handler = ModelResultsHandler(
        ["postal_code"], [0.7, 0.9], reporting.copy(), nonreporting.copy(), notexpected.copy()
    )
    handler.add_unit_predictions("e1", predictions_e1)
    handler.add_unit_intervals("e1", intervals_e1)

    handler.add_agg_predictions("e1", "postal_code", agg1_e1, intervals1_e1)
    handler.process_final_results()

    assert "unit_data" not in handler.final_results


def test_add_turnout_results():
    reporting_with_turnout = (
        reporting.copy().drop(columns=["results_e2"]).rename(columns={"results_e1": "results_margin"})
    )
    reporting_with_turnout["results_weights"] = [100000, 200000]
    reporting_with_turnout["pred_margin"] = [0.1, 0.05]

    nonreporting_pred_turnout = [150]
    predictions = [0.2]
    intervals = {0.9: PredictionIntervals([0.1], [0.3])}

    notexpected_with_turnout = (
        notexpected.copy().drop(columns=["results_e2"]).rename(columns={"results_e1": "results_margin"})
    )
    notexpected_with_turnout["results_weights"] = [0]

    agg_margin_with_turnout = agg1_e1.copy().drop(columns=["pred_e1"])
    agg_margin_with_turnout["pred_margin"] = [0.1, 0.05]
    agg_margin_with_turnout["pred_turnout"] = [2500, 4500]

    intervals_margin = {0.9: PredictionIntervals([0.05, 0.15], [0, 0.10])}

    handler = ModelResultsHandler(
        ["district", "unit"],
        [0.9],
        reporting_with_turnout.copy(),
        nonreporting.copy(),
        notexpected_with_turnout.copy(),
    )
    handler.add_unit_predictions("margin", predictions)
    handler.add_unit_turnout_predictions(nonreporting_pred_turnout)
    handler.add_unit_intervals("margin", intervals)
    expected_cols = [
        "pred_margin",
        "lower_0.9_margin",
        "upper_0.9_margin",
        "results_weights",
        "pred_turnout",
    ]

    assert set(expected_cols).issubset(set(handler.reporting_units))
    assert set(expected_cols).difference({"results_weights"}).issubset(set(handler.nonreporting_units))
    assert set(expected_cols).issubset(set(handler.unexpected_units))

    # test agg predictions for 1 agg and 1 estimand
    handler.add_agg_predictions("margin", "district", agg_margin_with_turnout, intervals_margin)
    # test preparation of final results data
    handler.process_final_results()

    assert "pred_turnout" in handler.final_results["unit_data"].columns
    assert "pred_turnout" in handler.final_results["district_data"].columns
