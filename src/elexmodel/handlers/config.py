import json
import logging
from pathlib import Path

from elexmodel.utils.file_utils import create_directory, get_directory_path

LOG = logging.getLogger(__name__)


class ConfigHandler:
    """
    Handler for model config
    """

    def __init__(self, election_id, s3_client=None, config=None, save=False):
        """
        Initialize config. If not present, download from s3
        """
        self.election_id = election_id
        self.s3_client = s3_client
        self.local_file_path = self.get_config_file_path()
        if config:
            self.config = config
        else:
            self.config = self.get_config()
        if save:
            self.save()

    def get_config_file_path(self):
        directory_path = get_directory_path()
        path = f"{directory_path}/config/{self.election_id}.json"
        return path

    def get_config(self):
        """
        Read config from file
        """
        LOG.info("Loading config: %s", self.election_id)

        # Read local config file if available
        if Path(self.local_file_path).is_file():
            with open(self.local_file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        # Else, get config from S3
        else:
            path_info = {"election_id": self.election_id}
            file_path = self.s3_client.get_file_path("config", path_info)
            config = self.s3_client.get(file_path)
        return config

    def _get_office_subconfig(self, office):
        """
        Get offices that we have model prepared for.
        This assumes that offices are unique per election, otherwise returns first
        """
        return list(filter(lambda x: x.get("office") == office, self.config.get(self.election_id)))[0]

    def get_offices(self):
        return [subconfig.get("office") for subconfig in self.config.get(self.election_id)]

    def get_baseline_pointer(self, office):
        # then we are using the old configs, without baseline pointers
        return self._get_office_subconfig(office).get(
            "baseline_pointer", {"dem": "dem", "gop": "gop", "turnout": "turnout"}
        )

    def get_estimand_baselines(self, office, estimands):
        """
        Return dict of baseline pointers for requested estimands
        """
        baseline_pointers = {estimand: self.get_baseline_pointer(office).get(estimand) for estimand in estimands}
        if "margin" in estimands:
            baseline_pointers["margin"] = "margin"
        return baseline_pointers

    def get_estimands(self, office):
        baseline_pointer = self.get_baseline_pointer(office)
        estimands = list(baseline_pointer.keys())
        if self.election_id.endswith("G"):
            estimands += ["margin"]  # would otherwise need to add margin to every single config
        return estimands

    def get_states(self, office):
        """
        Get states that office is being run for in election
        """
        return self._get_office_subconfig(office).get("states")

    def get_historical_election_ids(self, office):
        """
        Get election id for historical election, otherwise return None
        """
        return self._get_office_subconfig(office).get("historical_election")

    def get_geographic_unit_types(self, office):
        return self._get_office_subconfig(office).get("geographic_unit_types")

    def get_features(self, office):
        features = self._get_office_subconfig(office).get("features", [])
        if self.election_id.endswith("G"):
            features += [
                "baseline_normalized_margin"
            ]  # would otherwise need to add baseline_margin to every single config
        return features

    def get_aggregates(self, office):
        return self._get_office_subconfig(office).get("aggregates", [])

    def get_fixed_effects(self, office):
        return self._get_office_subconfig(office).get("fixed_effect", [])

    def save(self):
        if not Path(self.local_file_path).parent.exists():
            create_directory(str(Path(self.local_file_path).parent))
        with open(self.local_file_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f)
