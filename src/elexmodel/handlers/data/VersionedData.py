from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import tz

from elexmodel.handlers import s3
from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.logger import getModelLogger
from elexmodel.utils.file_utils import S3_FILE_PATH, TARGET_BUCKET

LOG = getModelLogger()


class VersionedDataHandler:
    def __init__(
        self,
        election_id,
        office_id,
        geographic_unit_type,
        estimands=["margin"],
        start_date=None,
        end_date=None,
        sample=2,
        tzinfo="America/New_York",
    ):
        self.election_id = election_id
        self.office_id = office_id
        self.geographic_unit_type = geographic_unit_type
        self.estimands = estimands
        self.start_date = start_date  # in EST
        self.end_date = end_date  # in EST

        if self.election_id.startswith("2020-11-03_USA_G"):
            target_bucket = "elex-models-prod"
        else:
            target_bucket = TARGET_BUCKET
        start_date = datetime.fromisoformat(start_date).astimezone(tz=tz.gettz("UTC")) if start_date else None
        end_date = datetime.fromisoformat(end_date).astimezone(tz=tz.gettz("UTC")) if end_date else None
        # versioned results natively are in UTC but we'll convert it back to timezone in tzinfo
        self.s3_client = s3.S3VersionUtil(target_bucket, start_date, end_date, tzinfo)

        # Sample lets us skip every nth version, by default 2.
        self.sample = sample

        # This handles timezone conversion for us, by default to EST.
        self.tz = tzinfo

    def get_versioned_results(self, filepath=None):
        if filepath is not None:
            versioned_results_np = np.load(f"{filepath}/versioned_results.npy")
            versioned_results_cols = np.loadtxt(f"{filepath}/versioned_results.txt", dtype="U")
            versioned_results = pd.DataFrame(versioned_results_np, columns=versioned_results_cols.tolist())
            versioned_results = versioned_results[versioned_results.last_modified_idx <= self.end_date]
            versioned_results["last_modified"] = versioned_results["last_modified_idx"]
            versioned_results["geographic_unit_fips"] = versioned_results["geographic_unit_fips"].apply(
                lambda x: str(x).replace(".", "-")
            )
            versioned_results["geographic_unit_fips"] = versioned_results["geographic_unit_fips"].apply(
                lambda x: x.split("-")[0] if x.split("-")[1] == "0" else x
            )
            self.data = versioned_results.sort_values("last_modified")
            return self.data

        if self.election_id.startswith("2020-11-03_USA_G"):
            path = "elex-models-prod/2020-general/results/pres/current.csv"
        elif self.election_id.startswith("2024-11-05_USA_G"):
            path = f"{S3_FILE_PATH}/{self.election_id}/results/{self.office_id}/{self.geographic_unit_type}/current_counties.csv"
        else:
            path = f"{S3_FILE_PATH}/{self.election_id}/results/{self.office_id}/{self.geographic_unit_type}/current.csv"

        data = self.s3_client.get(path, self.sample)
        LOG.info("Loaded versioned results from S3")
        if data is None:
            self.data = data
            return data
        estimandizer = Estimandizer()
        data, _ = estimandizer.add_estimand_results(data, self.estimands, False)
        self.data = data.sort_values("last_modified")
        return self.data

    def compute_versioned_margin_estimate(self, data=None):
        """
        This function imputes the margin at each percent reporting for a versioned dataset.
        We only see the normalized_margin at the times at which someone updates the voter data, but
        we want to use this to estimate the margin at all percent reportings.

        We do this by linearly interpolating the normalized_margin at each percent reporting.
        So, let's say for example, that we want to estimate results_normalized_margin at 88% reporting
        but we only see the normalized_margin at 80% and 90% reporting. We would estimate the margin
        by computing the "batch margin" for the votes that were reported betwen 80 and 90% reporting
        and then estimate the margin at 88% by

        margin_88 = [margin_80 + (batch_margin) * (88 - 80)] / 88

        This function does this for all percent reportings and all geographic units in the
        versioned dataset.
        """
        # Fill NaNs with 0
        if data is None:
            results = self.data
        else:
            results = data
        results.fillna(0, inplace=True)

        # Function to compute estimated margins for each group defined by the geographic_unit_fips
        # so you can think of this df as a dataframe of versioned results for one particular county
        def compute_estimated_margin(df):
            # Convert columns to NumPy arrays for faster computation
            results_turnout = df["results_turnout"].values
            percent_expected_vote = df["percent_expected_vote"].values
            results_dem = df["results_dem"].values
            results_gop = df["results_gop"].values
            results_weights = df["results_weights"].values

            # sometimes the percent_expected_vote we have recorded is non-monotonic
            # because the AP adjusted its model after the fact. We correct for this here.
            # we recompute the percent_expected_vote using the last reported value as the max
            perc_expected_vote_corr = np.divide(
                results_turnout, results_turnout[-1], out=np.zeros_like(results_turnout), where=results_turnout[-1] != 0
            )

            # check if perc_expected_vote_corr is monotone increasing (if not, give up and don't try to estimate a margin)
            if not np.all(np.diff(perc_expected_vote_corr) >= 0):
                return pd.DataFrame(
                    {
                        "percent_expected_vote": np.arange(101),
                        "nearest_observed_vote": np.nan * np.ones(101),
                        "est_margin": np.nan * np.ones(101),
                        "est_correction": np.nan * np.ones(101),
                        "error_type": "non-monotone percent expected vote",
                    }
                )

            # perc_expected_vote_corr is a value living in the unit interval [0, 1] - we want to rescale it to [0, 100]
            # or [0, 93] or whatever the latest percent_expected_vote value is
            df["percent_expected_vote"] = perc_expected_vote_corr * percent_expected_vote[-1]

            # Compute batch_margin using NumPy
            # this is the difference in dem_votes - the difference in gop_votes divided by the difference in total votes
            # that is, this is the normalized margin in the batch of votes recorded between versions
            batch_margin = (
                np.diff(results_dem, append=results_dem[-1]) - np.diff(results_gop, append=results_gop[-1])
            ) / np.diff(results_weights, append=results_weights[-1])

            # nan values in batch_margin are due to div-by-zero since there's no change in votes
            batch_margin[np.isnan(batch_margin)] = 0  # Set NaN margins to 0
            df["batch_margin"] = batch_margin

            # batch_margins should be between -1 and 1 (otherwise, there was a data entry issue and we will not use this unit)
            if np.abs(batch_margin).max() > 1:
                return pd.DataFrame(
                    {
                        "percent_expected_vote": np.arange(101),
                        "nearest_observed_vote": np.nan * np.ones(101),
                        "est_margin": np.nan * np.ones(101),
                        "est_correction": np.nan * np.ones(101),
                        "error_type": "batch_margin",
                    }
                )

            # Compute margin estimates using the formula in the block comment above this function
            # this is a bit hard to read because it's vectorized, but note that "perc" here is the
            # percent_expected_vote we want to estimate the margin at and we are trying to identify
            # both the normalized margin at the closest percent_expected_vote less than perc
            # and the batch_margin that got us to the next percent_expected_vote (above perc)
            # there are some subtleties here about handling the case where perc < the earliest recorded
            # percent_expected_vote - I'll call them out below

            # Extract relevant data as NumPy arrays
            percent_vote = df["percent_expected_vote"].to_numpy()
            batch_margin = df["batch_margin"].to_numpy()
            norm_margin = df["results_normalized_margin"].to_numpy()

            # Create perc values (0, 1, ..., max_percent)
            max_perc = int(np.max(percent_vote))
            percs = np.arange(0, max_perc + 1)

            # Find the last index where percent_vote <= perc for all percs
            obs_indices = np.searchsorted(percent_vote, percs, side="right") - 1
            clipped_indices = np.clip(obs_indices, 0, len(percent_vote) - 1)  # Ensure valid indices

            # Vectorized calculation of est_margin
            # when obs_indices = -1, this means that perc < the earliest recorded percent_expected_vote
            # then we will "pretend" that the nearest_observed_vote is 0
            # and then the est_margin should just be equal to the earliest recorded norm_margin
            observed_vote = np.where(obs_indices == -1, 0, percent_vote[clipped_indices])
            observed_norm_margin = np.where(obs_indices == -1, norm_margin[0], norm_margin[clipped_indices])
            observed_batch_margin = np.where(obs_indices == -1, norm_margin[0], batch_margin[clipped_indices])

            est_margins = observed_norm_margin * observed_vote + observed_batch_margin * (percs - observed_vote)
            est_margins = np.divide(
                est_margins, percs, where=percs != 0, out=np.zeros_like(est_margins)
            )  # Handle div-by-zero

            # Return a DataFrame with the multi-index (geographic_unit_fips, perc)
            return pd.DataFrame(
                {
                    "percent_expected_vote": percs,
                    "nearest_observed_vote": percent_vote[np.clip(obs_indices + 1, 0, len(percent_vote) - 1)],
                    "est_margin": est_margins,
                    "est_correction": norm_margin[-1] - est_margins,
                    "error_type": "none",
                }
            )

        results = results.groupby("geographic_unit_fips").apply(compute_estimated_margin).reset_index()

        for error_type in sorted(set(results["error_type"])):
            if error_type == "none":
                continue
            category_error_type = results[results["error_type"] == error_type].geographic_unit_fips.unique()
            LOG.info(f"# of versioned units with {error_type} error: {len(category_error_type)}")
        return results

    def get_versioned_predictions(self, filepath=None):
        if filepath is not None:
            return pd.read_csv(filepath)

        if self.election_id.startswith("2020-11-03_USA_G"):
            path = "elex-models-prod/2020-general/prediction/pres/current.csv"
            raise ValueError("No versioned predictions available for this election.")
        else:
            path = f"{S3_FILE_PATH}/{self.election_id}/predictions/{self.office_id}/{self.geographic_unit_type}/current.csv"

        return self.s3_client.get(path, self.sample)
