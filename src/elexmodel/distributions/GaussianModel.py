import numpy as np
import pandas as pd

from elexmodel.handlers import s3
from elexmodel.utils import math_utils, pandas_utils
from elexmodel.utils.file_utils import S3_FILE_PATH, TARGET_BUCKET, convert_df_to_csv


class GaussianModel:
    """
    Gaussian model for conformalization scores
    """

    def __init__(self, model_settings):
        super().__init__()
        self.save_conformalization = model_settings.get("save_conformalization", True)
        self.election_id = model_settings.get("election_id")
        self.office = model_settings.get("office")
        self.geographic_unit_type = model_settings.get("geographic_unit_type")
        self.winsorize = model_settings.get("winsorize", False)
        self.beta = model_settings.get("beta", 1)

    def _empty_gaussian_model(self, conformalization_data, aggregate):
        """
        Return empty Gaussian model. Everything is None
        """
        return (
            conformalization_data[aggregate]
            .assign(
                mu_lower_bound=None,
                mu_upper_bound=None,
                sigma_lower_bound=None,
                sigma_upper_bound=None,
                var_inflate=None,
            )
            .astype(
                dtype={
                    "mu_lower_bound": float,
                    "mu_upper_bound": float,
                    "sigma_lower_bound": float,
                    "sigma_upper_bound": float,
                    "var_inflate": float,
                }
            )
        )

    def _get_n_units_per_group(self, conformalization_data, nonreporting_units, aggregate):
        """
        Get number of units per group in aggregate
        """
        # if aggregate is None, we are fitting the model to the unit case
        # and all calibration sets fit into the same group
        if not aggregate:
            return {"n": conformalization_data.shape[0]}
        # if aggregate is not None, we are fitting a model for each group
        # we compute the number of units in each group.
        # This is a two step process, since we care about *all* groups, not
        # just the ones that may have units in the conformalization data.

        # get number units per group in the conformalization data
        conformalization_counts = conformalization_data.groupby(aggregate).size().reset_index(name="n")
        # get groups in nonreporting data (that is groupby + size + reset_index + drop)
        # outer join with conformalization counts to get *all* groups. If group was in
        # nonreporting data but not conformalization data then count will be NA, so replace with
        # zero
        return (
            nonreporting_units.groupby(aggregate)
            .size()
            .reset_index(drop=False)
            .drop(columns=0)
            .merge(conformalization_counts, how="outer", on=aggregate)
            .fillna({"n": 0})
        )

    def _fit(self, conformalization_data, estimand, aggregate, alpha):
        """
        Compute fit for Gaussian Model
        """
        # pandas does not support grouping by an empty list
        # need to return true for all if we want to have everything in the same group
        to_aggregate = aggregate
        drop_index = False
        if not aggregate:
            drop_index = True

            def to_aggregate(x):
                return True

        # fit gaussian model to conformalization data
        # use weighted median as center
        # bootstrap standard deviation
        gaussian_fit = (
            conformalization_data.groupby(to_aggregate)
            .apply(
                lambda x: pd.Series(
                    {
                        "var_inflate": math_utils.compute_inflate(x[f"last_election_results_{estimand}"]),
                        "mu_lower_bound": math_utils.weighted_median(
                            x.lower_bounds.values,
                            (
                                x[f"last_election_results_{estimand}"] / np.sum(x[f"last_election_results_{estimand}"])
                            ).to_numpy(),
                        ),
                        "mu_upper_bound": math_utils.weighted_median(
                            x.upper_bounds.values,
                            (
                                x[f"last_election_results_{estimand}"] / np.sum(x[f"last_election_results_{estimand}"])
                            ).to_numpy(),
                        ),
                        "sigma_lower_bound": self.beta
                        * math_utils.boot_sigma(x.lower_bounds.values, conf=(3 + alpha) / 4, winsorize=self.winsorize),
                        "sigma_upper_bound": self.beta
                        * math_utils.boot_sigma(x.upper_bounds.values, conf=(3 + alpha) / 4, winsorize=self.winsorize),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=drop_index)
        )
        return gaussian_fit

    def fit(
        self,
        conformalization_data,
        reporting_units,
        nonreporting_units,
        estimand,
        aggregate=[],
        alpha=0.9,
        reweight=False,
        top_level=True,
    ):
        """
        Fit lower/upper Gaussian models to conformalization data.
        Equivalent to computing mean and standard deviation for each model, since they are sufficient.
        Instead of mean, we use weighted median since it is more robust.
        """
        n_conformalization_data = conformalization_data.shape[0]
        if n_conformalization_data == 0:
            # if conformalization data is empty, return empty dataframe with nones
            return self._empty_gaussian_model(conformalization_data, aggregate)

        if reweight:
            # TODO: implement reweighting
            raise NotImplementedError
        counts = self._get_n_units_per_group(conformalization_data, nonreporting_units, aggregate)

        # if one group has fewer than MODEL_THRESHOLD observations to fit with,
        # remove one layer of aggregation and refit the parametric model
        # e.g. instead of fitting a parametric model for each county in a state,
        # produce a parametric model for the state
        MODEL_THRESHOLD = min(10, n_conformalization_data)
        if np.min(counts["n"]) < MODEL_THRESHOLD:
            # model for small groups (ie. ones where count < MODEL_THRESHOLD)
            gaussian_model_small_groups = self.fit(
                conformalization_data,
                reporting_units,
                nonreporting_units,
                estimand,
                aggregate=aggregate[:-1],  # remove the last (smallest) aggregate
                alpha=alpha,
                reweight=reweight,
                top_level=False,
            )

            # still construct the per-group parametric model when possible
            conformalization_data_for_large_groups = (
                counts.query("n >= @MODEL_THRESHOLD")
                .reset_index(drop=True)
                .merge(conformalization_data, how="inner", on=aggregate)
                # query("n < 0").reset_index(drop=True) # uncomment me to disable
                .drop(columns=["n"])
            )

            # select reporting/nonreporting data that are in groups
            # that are large enough only. semi_join does that
            reporting_units_for_large_groups = pandas_utils.semi_join(
                reporting_units, conformalization_data_for_large_groups, on=aggregate
            )

            nonreporting_units_for_large_groups = pandas_utils.semi_join(
                nonreporting_units, conformalization_data_for_large_groups, on=aggregate
            )

            # fit gaussian model for groups that have enough units per group
            gaussian_model_large_groups = self.fit(
                conformalization_data_for_large_groups,
                reporting_units_for_large_groups,
                nonreporting_units_for_large_groups,
                estimand,
                aggregate=aggregate,  # remove the last (smallest) aggregate
                alpha=alpha,
                reweight=reweight,
                top_level=False,
            )

            # combine large and small models
            x = pd.concat([gaussian_model_small_groups, gaussian_model_large_groups]).reset_index(drop=True)
        else:
            # when the group is large enough we can compute the Gaussian model for conformalization
            x = self._fit(conformalization_data, estimand, aggregate, alpha)

        # Write to s3 at the highest level of recursion before we exit GaussianModel
        # and return to GaussianElectionModel
        if top_level and aggregate and self.save_conformalization:
            # Write conformalization data
            gaussian_bounds = x.copy()
            self._write_conformalization_data(
                conformalization_data[
                    ["geographic_unit_fips"]
                    + aggregate
                    + [
                        f"last_election_results_{estimand}",
                        "lower_bounds",
                        "upper_bounds",
                    ]
                ],
                self.election_id,
                self.office,
                self.geographic_unit_type,
                estimand,
                aggregate,
                alpha,
            )
            # Write bounds
            self._write_gaussian_bounds(
                gaussian_bounds,
                self.election_id,
                self.office,
                self.geographic_unit_type,
                estimand,
                aggregate,
                alpha,
            )

        return x

    def _write_conformalization_data(
        self, conformalization_data, election_id, office, geographic_unit_type, estimand, aggregate, alpha
    ):
        """
        Write this data to S3 so we can examine if the upper and lower bounds look like
        they come from a Gaussian distribution.
        """
        s3_client = s3.S3CsvUtil(TARGET_BUCKET)
        # type(aggregate) == list
        aggregate_string = aggregate[-1]
        path = f"{S3_FILE_PATH}/{election_id}/gaussian/{office}/{geographic_unit_type}\
            /{estimand}-{aggregate_string}-{alpha}/conformalization_data"
        conformalization_data_csv = convert_df_to_csv(conformalization_data)
        s3_client.put(path, conformalization_data_csv)

    def _write_gaussian_bounds(
        self, gaussian_fit, election_id, office, geographic_unit_type, estimand, aggregate, alpha
    ):
        """
        Write this data to S3 so we can examine the gaussian mu and sigma bounds
        """
        s3_client = s3.S3CsvUtil(TARGET_BUCKET)
        # type(aggregate) == list
        aggregate_string = aggregate[-1]
        path = f"{S3_FILE_PATH}/{election_id}/gaussian/{office}/\
            {geographic_unit_type}/{estimand}-{aggregate_string}-{alpha}/bounds"
        gaussian_fit_csv = convert_df_to_csv(gaussian_fit)
        s3_client.put(path, gaussian_fit_csv)
