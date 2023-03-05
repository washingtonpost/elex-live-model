from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals
 
from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver
from scipy.stats import bootstrap
import numpy as np
class EnsembleElectionModel(BaseElectionModel):
    def __init__(self):
        self.unit_predictions = None

    @classmethod
    def string_add_prefix(cls, string, prefix, sep='_'):
        return f'{prefix}{sep}{string}'

    def process_matrix_for_solving(self, dataframe, prefix='results'):
        dem_string = self.string_add_prefix('dem', prefix)
        gop_string = self.string_add_prefix('gop', prefix)
        nonvoter_string = self.string_add_prefix('nonvoters', prefix)
        matrix = dataframe[[dem_string, gop_string, 'total_gen_voters']].copy()
        matrix[nonvoter_string] = matrix.total_gen_voters - (matrix[dem_string] + matrix[gop_string])
        return matrix[[dem_string, gop_string, nonvoter_string]].copy().to_numpy()

    def get_samples(self, reporting_units, sampling_method, m):
        K = reporting_units.shape[0]
        n = round(0.2 * K)
        if sampling_method == 'random':
            return np.random.choice(K, size=(n, m), replace=True).transpose()


    def compute_predictions(self, reporting_units, nonreporting_units, method='transition', sampling_method = 'random'):
        m = 100
        if method == 'transition':
            # estimate transition matrices

            sample_indices = self.get_samples(reporting_units, sampling_method, m)
            # sample_indices_test = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).transpose()
            # reporting_matrix_current[sample_indices]
            reporting_matrix_current = self.process_matrix_for_solving(reporting_units, 'results')
            reporting_matrix_current_sampled = reporting_matrix_current[sample_indices]
            reporting_matrix_past = self.process_matrix_for_solving(reporting_units, 'baseline')
            reporting_matrix_past_sampled = reporting_matrix_past[sample_indices]

            nonreporting_matrix_past = self.process_matrix_for_solving(nonreporting_units, 'baseline')

            self.unit_predictions = []
            for i in range(m):
                reporting_matrix_current_i = reporting_matrix_current_sampled[i]
                reporting_matrix_past_i = reporting_matrix_past_sampled[i]

                transition_matrix_solver = TransitionMatrixSolver()
                transition_matrix_solver.fit(reporting_matrix_past_i, reporting_matrix_current_i, strict=False)

                preds_i = transition_matrix_solver.predict(nonreporting_matrix_past).round(decimals=0)[:,:2] # can drop nonvoters now
                # still need to max with current results
                self.unit_predictions.append(preds_i)
                print(f"{i} done")
        elif method == 'regression':
            # estimate many regressions with covariates
            pass
    
    def get_unit_predictions(self):
        median_predictions = np.median(self.unit_predictions, axis=0)
        dem_predictions = median_predictions[:,0]
        gop_predictions = median_predictions[:,1]
        return dem_predictions, gop_predictions

    def get_unit_prediction_intervals(self):
        max_predictions = np.max(self.unit_predictions, axis=0)
        min_predictions = np.min(self.unit_predictions, axis=0)
        dem_max = max_predictions[:,0]
        gop_max = max_predictions[:,1]
        dem_min = min_predictions[:,0]
        gop_min = min_predictions[:,1]
        return dem_min, dem_max, gop_min, gop_max

    def _get_reporting_aggregate_votes(self, reporting_units, unexpected_units, aggregate):
        reporting_units_known_votes = reporting_units.groupby(aggregate).sum().reset_index(drop=False)

        # we cannot know the county classification of unexpected geographic units, so we can't add the votes back in
        if "county_classification" in aggregate:
            aggregate_votes = reporting_units_known_votes[aggregate + [f"results_dem", "results_gop", "reporting"]]
        else:
            unexpected_units_known_votes = unexpected_units.groupby(aggregate).sum().reset_index(drop=False)

            # outer join to ensure that if entire districts of county classes are unexpectedly present, we
            # should still have them. Same reasoning to replace NA with zero
            # NA means there was no such geographic unit, so they don't capture any votes
            results_col_dem = f"results_dem"
            results_col_gop = f"results_gop"
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
                        f"results_dem_expected": 0,
                        f"results_dem_unexpected": 0,
                        "results_gop_expected": 0,
                        "results_gop_unexpected": 0,
                        "reporting_expected": 0,
                        "reporting_unexpected": 0,
                    }
                )
                .assign(
                    **{
                        results_col_dem: lambda x: x[f"results_dem_expected"] + x[f"results_dem_unexpected"],
                        results_col_gop: lambda x: x[f"results_gop_expected"] + x[f"results_gop_unexpected"],

                        reporting_col: lambda x: x["reporting_expected"] + x["reporting_unexpected"],
                    },
                )[aggregate + [f"results_dem", "results_gop", "reporting"]]
            )

        return aggregate_votes

    def get_aggregate_predictions(self, reporting_units, nonreporting_units, unexpected_units, aggregate):
        # already counted votes
        aggregate_votes = self._get_reporting_aggregate_votes(reporting_units, unexpected_units, aggregate)

        # these are subunits that are not already counted
        aggregate_preds = (
            nonreporting_units.groupby(aggregate)
            .sum()
            .reset_index(drop=False)
            .rename(
                columns={
                    f"pred_dem": f"pred_only_dem",
                    f"results_dem": f"results_only_dem",
                    "pred_gop": "pred_only_gop",
                    "results_gop": "results_only_gop",
                    "reporting": "reporting_only",
                }
            )
        )

        aggregate_data = (
            aggregate_votes.merge(aggregate_preds, how="outer", on=aggregate)
            .fillna(
                {
                    f"results_dem": 0,
                    f"pred_only_dem": 0,
                    f"results_only_dem": 0,
                    "results_gop": 0,
                    "pred_only_gop": 0,
                    "results_only_gop": 0,
                    "reporting": 0,
                    "reporting_only": 0,
                }
            )
            .assign(
                # don't need to sum results_only for predictions since those are superceded by pred_only
                # preds can't be smaller than results, since we maxed between predictions and results in unit function.
                **{
                    f"pred_dem": lambda x: x[f"results_dem"] + x[f"pred_only_dem"],
                    f"results_dem": lambda x: x[f"results_dem"] + x[f"results_only_dem"],
                    f"pred_gop": lambda x: x[f"results_gop"] + x[f"pred_only_gop"],
                    f"results_gop": lambda x: x[f"results_gop"] + x[f"results_only_gop"],
                    "reporting": lambda x: x["reporting"] + x["reporting_only"],
                },
            )
            .sort_values(aggregate)[aggregate + [f"pred_dem", f"results_dem", "pred_gop", "results_gop", "reporting"]]
            .reset_index(drop=True)
        )
        return aggregate_data

    def get_aggregate_prediction_intervals(self,
        reporting_units,
        nonreporting_units,
        unexpected_units,
        aggregate
    ):
        # we're doing the same work as in get_aggregate_predictions here, can we just do this work once?
        aggregate_votes = self._get_reporting_aggregate_votes(reporting_units, unexpected_units, aggregate)

        # prediction intervals sum, kind of miraculous
        # Technically this is a conservative approach. This is equivalent to perfect correlation if
        # we assume that the prediction intervals are multivariate gaussian
        aggregate_prediction_intervals = (
            nonreporting_units.groupby(aggregate)
            .sum()
            .reset_index(drop=False)
            .rename(columns={"lower_0.9_dem": f"pi_lower_dem", "upper_0.9_dem": f"pi_upper_dem", "lower_0.9_gop": "pi_lower_gop", "upper_0.9_gop": "pi_upper_gop"})[
                aggregate + [f"pi_lower_dem", f"pi_upper_dem", "pi_lower_gop", "pi_upper_gop"]
            ]
        )

        # sum in prediction intervals and rename
        aggregate_data = (
            aggregate_votes.merge(aggregate_prediction_intervals, how="outer", on=aggregate)
            .fillna({f"results_dem": 0, f"pi_lower_dem": 0, f"pi_upper_dem": 0, "results_gop": 0, "pi_lower_gop": 0, "pi_upper_gop": 0})
            .assign(
                lower_dem=lambda x: x[f"pi_lower_dem"] + x[f"results_dem"],
                upper_dem=lambda x: x[f"pi_upper_dem"] + x[f"results_dem"],
                lower_gop=lambda x: x["pi_lower_gop"] + x["results_gop"],
                upper_gop=lambda x: x["pi_upper_gop"] + x["results_gop"]
            )
            .sort_values(aggregate)[aggregate + ["lower_dem", "upper_dem", "lower_gop", "upper_gop"]]
            .reset_index(drop=True)
        )

        return PredictionIntervals(aggregate_data.lower_dem.round(decimals=0), aggregate_data.upper_dem.round(decimals=0)), PredictionIntervals(aggregate_data.lower_gop.round(decimals=0), aggregate_data.upper_gop.round(decimals=0))
    
    def get_all_conformalization_data_unit(self):
        return None
    
    def get_all_conformalization_data_agg(self):
        return None