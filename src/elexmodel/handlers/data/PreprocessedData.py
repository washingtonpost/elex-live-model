from pathlib import Path

from elexmodel.handlers.data.BaseDataHandler import BaseDataHandler
from elexmodel.logger import getModelLogger
from elexmodel.utils.file_utils import create_directory

LOG = getModelLogger()


class PreprocessedDataHandler(BaseDataHandler):
    """
    Handler for preprocessed data for model
    """

    def __init__(
        self,
        election_id,
        office_id,
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
        self.estimand_baselines = estimand_baselines
        self.include_results_estimand = include_results_estimand
        super().__init__(
            election_id,
            office_id,
            geographic_unit_type,
            estimands,
            s3_client=s3_client,
            historical=historical,
            data=data,
        )

    def select_rows_in_states(self, data, states_with_election):
        data = data.query(
            "postal_code in @states_with_election"
        ).reset_index(  # make sure to return results for relevant states only
            drop=True
        )
        return data

    def load_data(self, data):
        """
        Load preprocessed csv data as df
        """
        LOG.info("Loading preprocessed data: %s, %s, %s", self.election_id, self.office_id, self.geographic_unit_type)
        data = self.estimandizer.add_estimand_baselines(
            data,
            self.estimand_baselines,
            self.historical,
            include_results_estimand=self.include_results_estimand,
        )

        return data

    def save_data(self, preprocessed_data):
        if not Path(self.file_path).parent.exists():
            create_directory(str(Path(self.file_path).parent))
        preprocessed_data.to_csv(self.file_path, index=False)
