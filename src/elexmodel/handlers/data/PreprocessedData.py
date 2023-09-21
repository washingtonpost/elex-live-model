import logging
from io import StringIO
from pathlib import Path

import pandas as pd

from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.utils.file_utils import create_directory, get_directory_path

LOG = logging.getLogger(__name__)


class PreprocessedDataHandler:
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
        include_results_estimand=False,
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
        self.include_results_estimand = include_results_estimand
        self.estimandizer = Estimandizer()

        self.local_file_path = self.get_preprocessed_data_path()

        if data is not None:
            self.data = self.load_data(data)
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
        return self.load_data(data)

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

    def load_data(self, preprocessed_data):
        """
        Load preprocessed csv data as df
        """
        LOG.info("Loading preprocessed data: %s, %s, %s", self.election_id, self.office, self.geographic_unit_type)
        data = self.estimandizer.add_estimand_baselines(
            preprocessed_data,
            self.estimand_baselines,
            self.historical,
            include_results_estimand=self.include_results_estimand,
        )

        return data

    def save_data(self, preprocessed_data):
        if not Path(self.local_file_path).parent.exists():
            create_directory(str(Path(self.local_file_path).parent))
        preprocessed_data.to_csv(self.local_file_path, index=False)
