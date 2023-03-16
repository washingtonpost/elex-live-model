from elexmodel.models.BaseElectionModel import BaseElectionModel, PredictionIntervals
 
from elexsolver.TransitionMatrixSolver import TransitionMatrixSolver
from elexsolver.QuantileRegressionSolver import QuantileRegressionSolver
import numpy as np
import pandas as pd

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
        elif 'total_gen_voters':
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

    def bootstrap_b(self, reporting_units, nonreporting_units, model_fun):
        sample = self.get_samples(reporting_units, 1)
        reporting_units_sample_b = reporting_units.loc[sample,:]
        preds_b = model_fun(reporting_units_sample_b, nonreporting_units)
        return preds_b
    
    def bootstrap_unit_predictions(self, reporting_units, nonreporting_units, model_fun, B=100):
        self.bootstraped_unit_predictions = np.zeros((B, nonreporting_units.shape[0], len(self.estimand_to_index)))
        for b in range(B):
            preds_b = self.bootstrap_b(reporting_units, nonreporting_units, model_fun)
            self.bootstraped_unit_predictions[b, :, :] = preds_b

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
    
    def compute_regression(self, reporting_units_sample_b, nonreporting_units):
        nonreporting_units_features = nonreporting_units[self.features]
        model = QuantileRegressionSolver(solver='ECOS')
        reporting_units_sample_b_features = reporting_units_sample_b[self.features]
        weights = self.get_total_people(reporting_units_sample_b)
        preds = np.zeros((nonreporting_units.shape[0], len(self.estimands)))
        for estimand, j in self.estimand_to_index.items():
            reporting_units_sample_i_residuals = reporting_units_sample_b[f"residuals_{estimand}"]
            self.fit_model(model, reporting_units_sample_b_features, reporting_units_sample_i_residuals, 0.5, weights, True)
            preds_j = model.predict(nonreporting_units_features)
            preds_j = preds_j * nonreporting_units[f"total_voters_{estimand}"] # move into vote difference space
            preds_j = preds_j + nonreporting_units[f"last_election_results_{estimand}"] # move into vote space
            preds_j = np.maximum(preds_j, nonreporting_units[f"results_{estimand}"])
            preds[:, j] = preds_j
        return preds
    
    def compute_unit_predictions(self, reporting_units, nonreporting_units, model_fun):
        preds = model_fun(reporting_units, nonreporting_units)        
        self.unit_predictions = preds.round(decimals=0)

    def get_unit_predictions(self, reporting_units, nonreporting_units, estimand):
        model_fun = self.get_model_fun()
        if self.unit_predictions is None:
            self.compute_unit_predictions(reporting_units, nonreporting_units, model_fun)
        if self.bootstraped_unit_predictions is None:
            self.bootstrap_unit_predictions(reporting_units, nonreporting_units, model_fun, B=200)
        estimand_index = self.get_estimand_index(estimand)
        return self.unit_predictions[:,estimand_index]
    
    def compute_unit_prediction_intervals(self, alpha):
        lower_quantile = (1 - alpha) / 2
        upper_quantile = (1 + alpha) / 2
        self.unit_prediction_intervals_lower[alpha] = np.quantile(self.bootstraped_unit_predictions, lower_quantile, axis=0)
        self.unit_prediction_intervals_upper[alpha] = np.quantile(self.bootstraped_unit_predictions, upper_quantile, axis=0)

    def get_unit_prediction_intervals(self, reporting_units, nonreporting_units, alpha, estimand):
        if alpha not in self.unit_prediction_intervals_lower:
            self.compute_unit_prediction_intervals(alpha)

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

        lower_quantile = (1 - alpha) / 2
        upper_quantile = (1 + alpha) / 2
        
        estimand_index = self.get_estimand_index(estimand)
        nonreporting_units_samples = pd.DataFrame(self.bootstraped_unit_predictions[:, :, estimand_index].transpose())
        nonreporting_units_samples = (
            nonreporting_units[aggregate]
            .copy()
            .join(nonreporting_units_samples)
        )

        aggregate_prediction_intervals_lower = (
            nonreporting_units_samples
            .groupby(aggregate)
            .sum()
            .quantile(lower_quantile, axis=1)
            .rename('pi_lower')
            .reset_index(drop=False)
        )

        aggregate_prediction_intervals_upper = (
            nonreporting_units_samples
            .groupby(aggregate)
            .sum()
            .quantile(upper_quantile, axis=1)
            .rename('pi_upper')
            .reset_index(drop=False)
        )

        # sum in prediction intervals and rename
        aggregate_data = (
            aggregate_votes.merge(aggregate_prediction_intervals_lower, how="outer", on=aggregate)
            .fillna({f"results_{estimand}": 0, 'pi_lower': 0})
            .merge(aggregate_prediction_intervals_upper, how='outer', on=aggregate)
            .fillna({f"results_{estimand}": 0, 'pi_upper': 0})
            .assign(
                lower=lambda x: x[f"pi_lower"] + x[f"results_{estimand}"],
                upper=lambda x: x[f"pi_upper"] + x[f"results_{estimand}"],
            )
            .sort_values(aggregate)[aggregate + ["lower", "upper"]]
            .reset_index(drop=True)
        )
        return PredictionIntervals(aggregate_data.lower.round(decimals=0), aggregate_data.upper.round(decimals=0))
    
    def get_all_conformalization_data_unit(self):
        return None
    
    def get_all_conformalization_data_agg(self):
        return None