import logging
from io import StringIO
from pathlib import Path

import pandas as pd

from elexmodel.utils.file_utils import create_directory, get_directory_path

LOG = logging.getLogger(__name__)


class PreprocessedDataHandler(object):
    """
    Handler for preprocessed data for model
    """

    def __init__(
        self,
        election_id,
        office,
        geographic_unit_type,
        estimands,
        estimand_baselines,
        s3_client=None,
        historical=False,
        data=None,
    ):
        """
        Initialize preprocessed data. If not present, download from s3.
        """
        self.election_id = election_id
        self.office = office
        self.geographic_unit_type = geographic_unit_type
        self.estimands = estimands
        self.s3_client = s3_client
        self.estimand_baselines = estimand_baselines
        self.historical = historical

        self.local_file_path = self.get_preprocessed_data_path()

        if data is not None:
            self.data = self.load_data(data, estimand_baselines)
        else:
            self.data = self.get_data()

    def get_data(self):
        # If local data file is not available, read data from s3
        if not Path(self.local_file_path).is_file():
            path_info = {
                "election_id": self.election_id,
                "office": self.office,
                "geographic_unit_type": self.geographic_unit_type,
            }
            file_path = self.s3_client.get_file_path("preprocessed", path_info)
            csv_data = self.s3_client.get(file_path)
            # read data as a buffer
            preprocessed_data = StringIO(csv_data)
        else:
            # read data as a filepath
            preprocessed_data = self.local_file_path

        data = pd.read_csv(preprocessed_data, dtype={"geographic_unit_fips": str, "county_fips": str, "district": str})
        return self.load_data(data, self.estimand_baselines)

    def get_preprocessed_data_path(self):
        directory_path = get_directory_path()
        path = f"{directory_path}/data/{self.election_id}/{self.office}/data_{self.geographic_unit_type}.csv"
        return path

    def select_rows_in_states(self, data, states_with_election):
        data = data.query(
            "postal_code in @states_with_election"
        ).reset_index(  # make sure to return results for relevant states only
            drop=True
        )
        return data

    def load_data(self, preprocessed_data, estimand_baselines):
        """
        Load preprocessed csv data as df
        """
        LOG.info("Loading preprocessed data: %s, %s, %s", self.election_id, self.office, self.geographic_unit_type)

        if self.historical:
            # if we are in a historical election we are only reading preprocessed data to get
            # the historical election results of the currently reporting units.
            # so we don't care about the total voters or the baseline election.
            return preprocessed_data

        for estimand, pointer in estimand_baselines.items():
            baseline_name = f"baseline_{pointer}"
            preprocessed_data[f"last_election_results_{estimand}"] = preprocessed_data[baseline_name].copy()
            # TODO: rename total voters column
            # Adding one to prevent zero divison
            preprocessed_data[f"total_voters_{estimand}"] = preprocessed_data[f"last_election_results_{estimand}"] + 1

        return preprocessed_data

    def save_data(self, preprocessed_data):
        if not Path(self.local_file_path).parent.exists():
            create_directory(str(Path(self.local_file_path).parent))
        preprocessed_data.to_csv(self.local_file_path, index=False)
