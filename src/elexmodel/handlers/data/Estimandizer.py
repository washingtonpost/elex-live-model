import re


class Estimandizer:
    """
    Estimandizer. Generate estimands explicitly.
    """

    def __init__(self, data_handler, election_type, estimand_fns={}):
        self.data_handler = data_handler
        self.election_type = election_type
        self.estimand_fns = estimand_fns
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
        missing_columns = [col for col in columns if col not in self.data_handler.data.columns]
        return len(missing_columns) == 0

    def verify_estimand(self, estimand):
        """
        Verify which estimands can be formed given a dataset and a list of estimands we would like to create
        """
        # Check if estimand is a supported value
        if estimand not in self.transformation_map:
            raise ValueError(f"Estimand '{estimand}' is not supported.")
        self.transformations = self.transformation_map[estimand]

        # Check if estimand function contains all the neccessary local variables to run
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
            if isinstance(given_function, str):
                eval(f"{given_function}(self)")
            else:
                given_function()
        else:
            if given_function is None and estimand is not None:
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
        self.pre_check_estimands()

        for key, value in self.estimand_fns.items():
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
        if "results_dem" in self.data_handler.data.columns:
            self.data_handler.data["dem_votes"] = self.data_handler.data["results_dem"]
        if "results_gop" in self.data_handler.data.columns:
            self.data_handler.data["gop_votes"] = self.data_handler.data["results_gop"]
        if "results_turnout" in self.data_handler.data.columns:
            self.data_handler.data["total_votes"] = self.data_handler.data["results_turnout"]
        elif (
            "results_turnout" not in self.data_handler.data.columns
            and "results_dem" in self.data_handler.data.columns
            and "results_gop" in self.data_handler.data.columns
        ):
            self.data_handler.data["total_votes"] = (
                self.data_handler.data["results_dem"] + self.data_handler.data["results_gop"]
            )
        else:
            self.data_handler.data["total_votes"] = None
            if "results_dem" in self.data_handler.data.columns and "results_gop" in self.data_handler.data.columns:
                self.data_handler.data["results_dem"] = None
                self.data_handler.data["results_gop"] = None

    def calculate_party_vote_share(self):
        if "dem_votes" in self.data_handler.data.columns and "total_votes" in self.data_handler.data.columns:
            self.data_handler.data["party_vote_share_dem"] = (
                self.data_handler.data["dem_votes"] / self.data_handler.data["total_votes"]
            )
        if "gop_votes" in self.data_handler.data.columns and "total_votes" in self.data_handler.data.columns:
            self.data_handler.data["party_vote_share_gop"] = (
                self.data_handler.data["gop_votes"] / self.data_handler.data["total_votes"]
            )
        if (
            "baseline_dem_votes" in self.data_handler.data.columns
            and "baseline_total_votes" in self.data_handler.data.columns
        ):
            self.data_handler.data["baseline_party_vote_share_dem"] = (
                self.data_handler.data["baseline_dem_votes"] / self.data_handler.data["baseline_total_votes"]
            )
        if (
            "baseline_gop_votes" in self.data_handler.data.columns
            and "baseline_total_votes" in self.data_handler.data.columns
        ):
            self.data_handler.data["baseline_party_vote_share_gop"] = (
                self.data_handler.data["baseline_gop_votes"] / self.data_handler.data["baseline_total_votes"]
            )

    def candidate(self):
        # cands_old = re.findall(r'results_(\w+)_(\d+)', self.data_handler.data.columns)
        r = re.compile("results_*")
        cands = list(filter(r.match, self.data_handler.data.columns))
        # cand_set = set(candidate_data)
        for cand_name in cands:
            new_name = cand_name[8:]
            self.data_handler.data[new_name] = self.data_handler.data[cand_name]
