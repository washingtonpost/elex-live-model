import abc
from io import StringIO
from pathlib import Path

import pandas as pd

from elexmodel.handlers.data.Estimandizer import Estimandizer
from elexmodel.utils.file_utils import get_directory_path


class BaseDataHandler(abc.ABC):
    """
    Abstract base handler for model data
    """

    def __init__(
        self, election_id, office, geographic_unit_type, estimands, s3_client=None, historical=False, data=None
    ):
        self.election_id = election_id
        self.office = office
        self.geographic_unit_type = geographic_unit_type
        self.estimands = estimands
        self.s3_client = s3_client
        self.historical = historical
        self.estimandizer = Estimandizer()
        self.file_path = self.get_data_path()

        if data is not None:
            self.data = self.load_data(data)
        else:
            self.data = self.get_data()

    def get_data_path(self):
        directory_path = get_directory_path()
        path = f"{directory_path}/data/{self.election_id}/{self.office}/data_{self.geographic_unit_type}.csv"
        return path

    def get_data(self):
        # If local data file is not available, read data from s3
        if not Path(self.file_path).is_file():
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

    @abc.abstractmethod
    def load_data(self, data):
        pass
