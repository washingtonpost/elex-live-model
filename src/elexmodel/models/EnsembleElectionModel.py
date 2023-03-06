from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals
 
from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver
from scipy.stats import bootstrap
import numpy as np
import pandas as pd

class EnsembleElectionModel(BaseElectionModel):
    def __init__(self, estimands, alphas):
        self.unit_prediction_samples = []
        self.unit_predictions = None
        self.alphas = alphas
        self.unit_prediction_intervals_lower = {}
        self.unit_prediction_intervals_upper = {}
        self.estimands = estimands
        self.estimand_to_index = {estimand: i for i, estimand in enumerate(self.estimands)}
        self.method = 'transition'
        self.sampling_method = 'random'

    def get_minimum_reporting_units(self, alpha):
        return np.power(len(self.estimands), 2) + 1

    def get_estimand_index(self, estimand):
        return self.estimand_to_index[estimand]

    @classmethod
    def string_add_prefix(cls, string, prefix, sep='_'):
        return f'{prefix}{sep}{string}'

    @classmethod
    def get_total_people(cls, dataframe):
        if 'total_people' in dataframe:
            return dataframe.total_people
        elif 'total_gen_voters':
            return dataframe.total_gen_voters

    def process_matrix_for_solving(self, dataframe, prefix='results'):
        estimand_string_list = [self.string_add_prefix(estimand, prefix) for estimand in self.estimands]
        nonvoter_string = self.string_add_prefix('nonvoters', prefix)
        dataframe['total_people'] = self.get_total_people(dataframe)
        matrix = dataframe[estimand_string_list + ['total_people']].copy()
        matrix[nonvoter_string] = matrix.total_people - matrix[estimand_string_list].sum(1)
        return matrix[estimand_string_list + [nonvoter_string]].copy().to_numpy()

    def get_samples(self, reporting_units, m):
        K = reporting_units.shape[0]
        n = round(0.5 * K)
        if self.sampling_method == 'random':
            return np.random.choice(K, size=(n, m), replace=True).transpose()

    def sample_predictions(self, reporting_units, nonreporting_units):
        m = 100
        if self.method == 'transition':
            # estimate transition matrices

            sample_indices = self.get_samples(reporting_units, m)
            # sample_indices_test = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).transpose()
            # reporting_matrix_current[sample_indices]
            reporting_matrix_current = self.process_matrix_for_solving(reporting_units, 'results')
            reporting_matrix_current_sampled = reporting_matrix_current[sample_indices]
            reporting_matrix_past = self.process_matrix_for_solving(reporting_units, 'baseline')
            reporting_matrix_past_sampled = reporting_matrix_past[sample_indices]

            nonreporting_matrix_past = self.process_matrix_for_solving(nonreporting_units, 'baseline')
            unit_prediction_samples = []
            for i in range(m):
                reporting_matrix_current_i = reporting_matrix_current_sampled[i]
                reporting_matrix_past_i = reporting_matrix_past_sampled[i]

                transition_matrix_solver = TransitionMatrixSolver()
                transition_matrix_solver.fit(reporting_matrix_past_i, reporting_matrix_current_i, strict=False)

                preds_i = transition_matrix_solver.predict(nonreporting_matrix_past).round(decimals=0)[:,:2] # can drop nonvoters now
                preds_i = np.maximum(preds_i, nonreporting_units[['results_dem', 'results_gop']]).to_numpy()
                
                unit_prediction_samples.append(preds_i)
            self.unit_prediction_samples = np.asarray(unit_prediction_samples)

        elif self.method == 'regression':
            # estimate many regressions with covariates
            pass
    
    def compute_unit_predictions(self):
        self.unit_predictions = np.median(self.unit_prediction_samples, axis=0)

    def get_unit_predictions(self, reporting_units, nonreporting_units, estimand):
        if self.unit_predictions is None:
            self.sample_predictions(reporting_units, nonreporting_units)
            self.compute_unit_predictions()
        estimand_index = self.get_estimand_index(estimand)
        return self.unit_predictions[:,estimand_index]
    
    def compute_unit_prediction_intervals(self, alpha):
        lower_quantile = (1 - alpha) / 2
        upper_quantile = (1 + alpha) / 2
        self.unit_prediction_intervals_lower[alpha] = np.quantile(self.unit_prediction_samples, lower_quantile, axis=0)
        self.unit_prediction_intervals_upper[alpha] = np.quantile(self.unit_prediction_samples, upper_quantile, axis=0)

    def get_unit_prediction_intervals(self, reporting_units, nonreporting_units, alpha, estimand):
        if alpha not in self.unit_prediction_intervals_lower:
            self.compute_unit_prediction_intervals(alpha)
        estimand_index = self.get_estimand_index(estimand)
        return PredictionIntervals(
            self.unit_prediction_intervals_lower[alpha][:,estimand_index],
            self.unit_prediction_intervals_upper[alpha][:,estimand_index],
            None
        )

    # getting aggregate prediction is not correct. need to aggregate and then get median
    # same for prediction intervals (!)

    def get_aggregate_predictions(self, reporting_units, nonreporting_units, unexpected_units, aggregate, estimand):
        aggregate_votes = self._get_reporting_aggregate_votes(reporting_units, unexpected_units, aggregate, estimand)
        estimand_index = self.get_estimand_index(estimand)
        nonreporting_units_samples = pd.DataFrame(self.unit_prediction_samples[:, :, estimand_index].transpose())
        nonreporting_units_samples = (
            nonreporting_units[aggregate]
            .copy()
            .join(nonreporting_units_samples)
        )
        aggregate_preds = (
            nonreporting_units_samples
            .groupby(aggregate)
            .sum()
            .median(1)
            .reset_index(drop=False)
            .rename(columns={0: f'pred_only_{estimand}'})
        )
        import pdb; pdb.set_trace()

    def get_aggregate_prediction_intervals(self,
        reporting_units,
        nonreporting_units,
        unexpected_units,
        aggregate,
        alpha,
        conformalization,
        estimand,
        model_settings,
    ):
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

        return PredictionIntervals(aggregate_data.lower.round(decimals=0), aggregate_data.upper.round(decimals=0), None)
    
    def get_all_conformalization_data_unit(self):
        return None
    
    def get_all_conformalization_data_agg(self):
        return None