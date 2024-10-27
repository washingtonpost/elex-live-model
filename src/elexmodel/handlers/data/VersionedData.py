import pandas as pd
import numpy as np

from dateutil import tz

from elexmodel.handlers import s3
from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.utils.file_utils import APP_ENV, S3_FILE_PATH, TARGET_BUCKET

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
        self.start_date = start_date # in EST
        self.end_date = end_date # in EST

        if self.election_id.startswith("2020-11-03_USA_G"):
            TARGET_BUCKET = "elex-models-prod"

        if APP_ENV != "local":
            start_date = start_date.astimezone(tz=tz.gettz("UTC")) if start_date else None
            end_date = end_date.astimezone(tz=tz.gettz("UTC")) if start_date else None
            # versioned results natively are in UTC but we'll convert it back to timezone in tzinfo
            self.s3_client = s3.S3VersionUtil(TARGET_BUCKET, start_date, end_date, tzinfo)

        # Sample lets us skip every nth version, by default 2.
        self.sample = sample

        # This handles timezone conversion for us, by default to EST.
        self.tz = tzinfo

    def get_versioned_results(self, filepath=None): 
        if filepath is not None:
            versioned_results_np = np.load(f"{filepath}/versioned_results.npy")
            versioned_results_cols = np.loadtxt(f"{filepath}/versioned_results.txt", dtype="U")
            versioned_results = pd.DataFrame(versioned_results_np, columns=versioned_results_cols.tolist())
            versioned_results = versioned_results[
                versioned_results.last_modified_idx <= self.end_date
            ]
            versioned_results["last_modified"] = versioned_results["last_modified_idx"]
            versioned_results["geographic_unit_fips"] = versioned_results["geographic_unit_fips"].apply(
                lambda x: str(x).replace('.', '-')
            )
            versioned_results["geographic_unit_fips"] = versioned_results['geographic_unit_fips'].apply(
                lambda x: x.split('-')[0] if x.split('-')[1] == "0" else x
            )
            self.data = versioned_results.sort_values("last_modified")
            return self.data
        
        if self.election_id.startswith("2020-11-03_USA_G"):
            path = "elex-models-prod/2020-general/results/pres/current.csv"
        else:
            path = f"{S3_FILE_PATH}/{self.election_id}/results/{self.office_id}/{self.geographic_unit_type}/current.csv"

        data = self.s3_client.get(path, self.sample)
        estimandizer = Estimandizer()
        data, _ = estimandizer.add_estimand_results(data, self.estimands, False)
        self.data = data.sort_values("last_modified")
        return self.data
    
    def compute_versioned_margin_estimate(self, data=None):
        # Fill NaNs with 0
        if data is None:
            results = self.data
        else:
            results = data
        results.fillna(0, inplace=True)

        # Function to compute estimated margins for each group
        def compute_estimated_margin(df):
            # Convert columns to NumPy arrays for faster computation
            results_turnout = df["results_turnout"].values
            percent_expected_vote = df["percent_expected_vote"].values
            results_dem = df["results_dem"].values
            results_gop = df["results_gop"].values
            results_weights = df["results_weights"].values

            # Percent expected vote correction using NumPy
            perc_expected_vote_corr = np.divide(results_turnout, results_turnout[-1], out=np.zeros_like(results_turnout), where=results_turnout[-1] != 0)

            # check if perc_expected_vote_corr is monotone increasing
            if not np.all(np.diff(perc_expected_vote_corr) >= 0):
                return pd.DataFrame({
                    "percent_expected_vote": np.arange(101),
                    "nearest_observed_vote": np.nan * np.ones(101),
                    "est_margin": np.nan * np.ones(101)
                })
            
            df["percent_expected_vote"] = perc_expected_vote_corr * percent_expected_vote[-1]

            # Compute batch_margin using NumPy
            batch_margin = (np.diff(results_dem, append=results_dem[-1]) -
                            np.diff(results_gop, append=results_gop[-1])) / np.diff(results_weights, append=results_weights[-1])
            batch_margin[np.isnan(batch_margin)] = 0  # Set NaN margins to 0
            df["batch_margin"] = batch_margin

            # batch_margins should be between -1 and 1 (otherwise, there was a data entry issue and we will not use this unit)
            if np.abs(batch_margin).max() > 1:
                return pd.DataFrame({
                    "percent_expected_vote": np.arange(101),
                    "nearest_observed_vote": np.nan * np.ones(101),
                    "est_margin": np.nan * np.ones(101)
                })
            
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
            observed_vote = np.where(
                obs_indices == -1,
                0,
                percent_vote[clipped_indices]
            )
            observed_norm_margin = np.where(
                obs_indices == -1,
                norm_margin[0], # TODO: test this logic
                norm_margin[clipped_indices]
            )
            observed_batch_margin = np.where(
                obs_indices == -1,
                norm_margin[0], # TODO: check this logic
                batch_margin[clipped_indices]
            )

            est_margins = (
                observed_norm_margin * observed_vote +
                observed_batch_margin * (percs - observed_vote)
            )
            est_margins = np.divide(est_margins, percs, where=percs != 0, out=np.zeros_like(est_margins))  # Handle div-by-zero

            # Return a DataFrame with the multi-index (geographic_unit_fips, perc)
            return pd.DataFrame({
                "percent_expected_vote": percs,
                "nearest_observed_vote": percent_vote[np.clip(obs_indices + 1, 0, len(percent_vote) - 1)],
                "est_margin": est_margins,
                "est_correction": norm_margin[-1] - est_margins
            })
        results = results.groupby("geographic_unit_fips").apply(compute_estimated_margin).reset_index()
        return results.reset_index()
    
    def get_versioned_predictions(self, filepath=None):
        if filepath is not None:
            return pd.read_csv(filepath)
        
        if self.election_id.startswith("2020-11-03_USA_G"):
            path = "elex-models-prod/2020-general/prediction/pres/current.csv"
            raise ValueError("No versioned predictions available for this election.")
        else:
            path = f"{S3_FILE_PATH}/{self.election_id}/predictions/{self.office_id}/{self.geographic_unit_type}/current.csv"

        return self.s3_client.get(path, self.sample)