class Estimandizer:
    """
    Estimandizer. Generate estimands explicitly.
    """

    def __init__(self, data_handler, office, estimands={}):
        self.data_handler = data_handler
        self.office = office
        self.estimands = estimands
        self.transformations = []
        self.transformation_map = {
            "party_vote_share": [self.calculate_party_vote_share],
            "candidate": [self.candidate],
        }

    def pre_check_estimands(self):
        """
        Ensure estimand isn't one of the pre-specified values that are already included
        """
        standard = ["dem_votes", "gop_votes", "total_votes"]
        if not self.check_input_columns(standard):
            self.create_estimand(None, self.standard)

    def check_input_columns(self, columns):
        """
        Check that input columns contain all neccessary values for a calculation
        """
        missing_columns = []
        if self.office == "G":
            missing_columns = [col for col in columns if col not in self.data_handler.data.columns]
        elif self.office == "P":
            missing_columns = [
                col for col in columns if col not in self.data_handler.data[self.data_handler.election_id]
            ]
        return len(missing_columns) == 0

    def verify_estimand(self, estimand):
        """
        Verify which estimands can be formed given a dataset and a list of estimands we would like to create
        """
        if estimand not in self.transformation_map:
            raise ValueError(f"Estimand '{estimand}' is not supported.")
        self.transformations = self.transformation_map[estimand]

        if not self.check_input_columns(
            [col for transform in self.transformations for col in transform.__code__.co_varnames[1:]]
        ):
            return []

        return self.transformations

    def create_estimand(self, estimand=None, given_function=None):
        """
        Create an estimand. You must give either a estimand name or a pre-written function.
        """
        if estimand is None and given_function is not None:
            given_function()
        elif given_function is None and estimand is not None:
            if estimand in self.transformation_map:
                if self.transformation_map[estimand][0] in self.transformations:
                    transformation_func = self.transformations[
                        self.transformations.index(self.transformation_map[estimand][0])
                    ]
                    transformation_func()

    def generate_estimands(self):
        """
        Main function to generate estimands
        """
        if self.office == "G":
            self.pre_check_estimands()

        for key, value in self.estimands.items():
            if value is None:  # Option 1: Pass in a list of estimands we want to build from a pre-set list
                self.verify_estimand(key)
                self.create_estimand(key, None)

            else:  # Option 2: Pass in pre-built functions
                self.create_estimand(None, value)

        return self.data_handler

    # Transformation methods
    def standard(self):
        """
        Create/overwrite the standard estimands: ["dem_votes", "gop_votes", "total_votes"]
        """
        if "results_dem" in self.data_handler.data.columns and "results_turnout" in self.data_handler.data.columns:
            self.data_handler.data["dem_votes"] = self.data_handler.data["results_dem"]
            self.data_handler.data["gop_votes"] = self.data_handler.data["results_gop"]
            self.data_handler.data["total_votes"] = self.data_handler.data["results_turnout"]
        else:
            self.data_handler.data["dem_votes"] = None
            self.data_handler.data["gop_votes"] = None
            self.data_handler.data["total_votes"] = None

    def calculate_party_vote_share(self):
        self.data_handler.data["party_vote_share_dem"] = (
            self.data_handler.data["dem_votes"] / self.data_handler.data["total_votes"]
        )
        self.data_handler.data["party_vote_share_gop"] = (
            self.data_handler.data["gop_votes"] / self.data_handler.data["total_votes"]
        )

    def candidate(self):
        election_data = self.data_handler.data[self.data_handler.election_id][0]
        candidate_data = election_data["baseline_pointer"]
        # cand_set = set(candidate_data)
        for cand_name in candidate_data:
            if cand_name != "turnout":
                election_data[cand_name] = candidate_data[cand_name]
