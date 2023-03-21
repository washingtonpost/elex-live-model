from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals
 
from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver
import numpy as np
import scipy
import pandas as pd

from arch.bootstrap import IIDBootstrap

import time

class EnsembleElectionModel(BaseElectionModel):
    def __init__(self, model_settings, estimands, alphas):
        super().__init__(model_settings)
        self.bootstraped_unit_predictions = None
        self.unit_predictions = None
        self.alphas = alphas
        self.unit_prediction_intervals_lower = {}
        self.unit_prediction_intervals_upper = {}
        self.estimands = estimands
        self.estimand_to_index = {estimand: i for i, estimand in enumerate(self.estimands)}
        self.method = 'regression'            
        self.sampling_method = 'random'

    def get_minimum_reporting_units(self, alpha):
        return np.power(len(self.estimands) + 1, 2)

    def get_estimand_index(self, estimand):
        return self.estimand_to_index[estimand]

    @classmethod
    def string_add_prefix(cls, string, prefix, sep='_'):
        return f'{prefix}{sep}{string}'

    @classmethod
    def get_total_people(cls, dataframe):
        if 'total_people' in dataframe:
            return dataframe.total_people
        elif 'total_gen_voters' in dataframe:
            return dataframe.total_gen_voters

    def get_model_fun(self):
        if self.method == 'transition':
            return  self.compute_transition_matrix
        elif self.method == 'regression':
            return self.compute_regression

    def process_dataframe_for_transition_model(self, dataframe, prefix='results'):
        estimand_string_list = [self.string_add_prefix(estimand, prefix) for estimand in self.estimands]
        nonvoter_string = self.string_add_prefix('nonvoters', prefix)
        dataframe['total_people'] = self.get_total_people(dataframe)
        matrix = dataframe[estimand_string_list + ['total_people']].copy()
        matrix[nonvoter_string] = matrix.total_people - matrix[estimand_string_list].sum(1)
        matrix[estimand_string_list + [nonvoter_string]] = matrix[estimand_string_list + [nonvoter_string]].divide(matrix.total_people, axis='index')
        return matrix[estimand_string_list + [nonvoter_string]].copy().to_numpy()

    def get_samples(self, reporting_units, b=1):
        K = reporting_units.shape[0]
        if self.sampling_method == 'random':
            shape = (K, b)
            if b == 1:
                shape = (K, )
            return np.random.choice(K, size=shape, replace=True).transpose()

    def stupid_function_to_test_bootstrap(self, reporting_units):
        import pdb; pdb.set_trace()

    def my_bootstrap(self, reporting_units, model_fun, alpha, B):
        import pdb; pdb.set_trace
        b = scipy.stats.bootstrap((reporting_units, ), self.stupid_function_to_test_bootstrap, n_resamples=B, confidence_level=(1 + alpha) / 2, vectorized=False)
        import pdb; pdb.set_trace

    def bootstrap_b(self, reporting_units, nonreporting_units, model_fun):
        sample = self.get_samples(reporting_units, 1)
        reporting_units_sample_b = reporting_units.loc[sample,:]
        preds_b = model_fun(reporting_units_sample_b, nonreporting_units)
        return preds_b
    
    def bootstrap_samples(self, reporting_units, nonreporting_units, model_fun, B=100):
        t0 = time.time()
        self.bootstraped_unit_predictions = np.zeros((B, nonreporting_units.shape[0], len(self.estimand_to_index)))
        for b in range(B):
            preds_b = self.bootstrap_b(reporting_units, nonreporting_units, model_fun)
            self.bootstraped_unit_predictions[b, :, :] = preds_b
        print(time.time() - t0)

    def compute_transition_matrix(self, reporting_units_sample_b, nonreporting_units):
        nonreporting_units_past = self.process_dataframe_for_transition_model(nonreporting_units, 'baseline')
        reporting_units_sample_i_current = self.process_dataframe_for_transition_model(reporting_units_sample_b, 'results')
        reporting_units_sample_i_past = self.process_dataframe_for_transition_model(reporting_units_sample_b, 'baseline')
        model = TransitionMatrixSolver()
        model.fit(reporting_units_sample_i_past, reporting_units_sample_i_current, strict=False)
        preds = model.predict(nonreporting_units_past)[:,:2] # can drop nonvoters now
        preds = np.maximum(preds, nonreporting_units[['results_dem', 'results_gop']]).to_numpy()
        preds = nonreporting_units.total_people.to_numpy().reshape(-1, 1) * preds # move predictions from % space to vote space
        return preds
    
    def compute_regression(self, reporting_units_sample_b):
        import pdb; pdb.set_trace()
        nonreporting_units_features = self.current_nonreporting_data[self.features]
        model = QuantileRegressionSolver(solver='ECOS')
        reporting_units_sample_b_features = reporting_units_sample_b[self.features]
        weights = self.get_total_people(reporting_units_sample_b)
        preds = np.zeros((self.current_nonreporting_data.shape[0], len(self.estimands)))
        for estimand, j in self.estimand_to_index.items():
            reporting_units_sample_i_residuals = reporting_units_sample_b[f"residuals_{estimand}"]
            self.fit_model(model, reporting_units_sample_b_features, reporting_units_sample_i_residuals, 0.5, weights, True)
            preds_j = model.predict(nonreporting_units_features)
            preds_j = preds_j * self.current_nonreporting_data[f"total_voters_{estimand}"] # move into vote difference space
            preds_j = preds_j + self.current_nonreporting_data[f"last_election_results_{estimand}"] # move into vote space
            preds_j = np.maximum(preds_j, self.current_nonreporting_data[f"results_{estimand}"])
            preds[:, j] = preds_j
        return preds
    
    def compute_unit_predictions(self, reporting_units, nonreporting_units, model_fun):
        self.current_nonreporting_data = nonreporting_units
        preds = model_fun(reporting_units)        
        self.unit_predictions = preds.round(decimals=0)

    def get_unit_predictions(self, reporting_units, nonreporting_units, estimand):
        model_fun = self.get_model_fun()
        if self.unit_predictions is None:
            self.compute_unit_predictions(reporting_units, nonreporting_units, model_fun)

        estimand_index = self.get_estimand_index(estimand)
        return self.unit_predictions[:,estimand_index]
    
    def compute_unit_prediction_intervals(self, alpha, reporting_units):
        se = self.bootstraped_unit_predictions.std(axis=0, ddof=1)
        df = reporting_units.shape[0] - len(self.features) - 1
        t_score = scipy.stats.t.ppf((1 + alpha) / 2, df)
        self.unit_prediction_intervals_lower[alpha] = np.maximum(self.unit_predictions - (t_score * se), 0).round(decimals=0) # should max with results
        self.unit_prediction_intervals_upper[alpha] = np.maximum(self.unit_predictions + (t_score * se), 0).round(decimals=0) # should max with results
        #lower_q = (1 - alpha) / 2
        #upper_q = (1 + alpha) / 2
        #self.unit_prediction_intervals_lower[alpha] = np.percentile(self.bootstraped_unit_predictions, lower_q, axis=0)
        #self.unit_prediction_intervals_upper[alpha] = np.percentile(self.bootstraped_unit_predictions, upper_q, axis=0)

    def get_unit_prediction_intervals(self, reporting_units, nonreporting_units, alpha, estimand):
        self.current_nonreporting_data = nonreporting_units
        self.my_bootstrap(reporting_units, self.get_model_fun(), alpha, B=1000)
        
        #if alpha not in self.unit_prediction_intervals_lower:
        #    self.compute_unit_prediction_intervals(alpha, reporting_units)

        estimand_index = self.get_estimand_index(estimand)
        return PredictionIntervals(
            self.unit_prediction_intervals_lower[alpha][:,estimand_index],
            self.unit_prediction_intervals_upper[alpha][:,estimand_index]
        )
    
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

        aggregate_predictions = (
            nonreporting_units
            .groupby(aggregate)
            .sum()
            .reset_index(drop=False)
            [aggregate + [f'pred_{estimand}']]
        )

        df = reporting_units.shape[0] - len(self.features) - 1
        t_score = scipy.stats.t.ppf((1 + alpha) / 2, df)

        estimand_index = self.get_estimand_index(estimand)
        nonreporting_units_samples = pd.DataFrame(self.bootstraped_unit_predictions[:, :, estimand_index].transpose())
        nonreporting_units_samples = (
            nonreporting_units[aggregate]
            .copy()
            .join(nonreporting_units_samples)
        )
        aggregate_std = (
            nonreporting_units_samples
            .groupby(aggregate)
            .sum()
            .std(axis=1)
            .rename('se')
            .reset_index(drop=False)
        )

        # sum in prediction intervals and rename
        aggregate_data = ( 
            aggregate_votes.merge(aggregate_predictions, how="outer", on=aggregate)
            .fillna({f"results_{estimand}": 0, f'pred_{estimand}': 0})
            .merge(aggregate_std, how='outer', on=aggregate)
            .fillna({"se": 0, "se_pred": 0})
            .assign(
                lower = lambda x: (x[f'pred_{estimand}'] - (t_score * x.se)) + x[f"results_{estimand}"],
                upper = lambda x: (x[f'pred_{estimand}'] + (t_score * x.se)) + x[f"results_{estimand}"],
            )
            .sort_values(aggregate)[aggregate + ["lower", "upper"]]
            .reset_index(drop=True)
        )

        return PredictionIntervals(aggregate_data.lower.round(decimals=0), aggregate_data.upper.round(decimals=0))
    
    def get_all_conformalization_data_unit(self):
        return None
    
    def get_all_conformalization_data_agg(self):
        return None