import numpy as np


class Estimandizer:
    """
    Estimandizer. Generate estimands explicitly.
    """

    def __init__(self, data_handler, estimands, given_function_dict={}):
        self.data_handler = data_handler
        self.estimands = estimands
        self.transformations = []
        self.given_function_dict = given_function_dict
        self.transformation_map = {
            "margin": [self.calculate_margin],
            "voter_turnout_rate": [self.calculate_voter_turnout_rate],
            "standardized_income": [self.standardize_median_household_income],
            "age_groups": [self.create_age_groups],
            "party_vote_share": [self.calculate_party_vote_share],
            "education_impact": [self.calculate_party_vote_share, self.analyze_education_impact],
            "gender_turnout_disparity": [self.investigate_gender_turnout_disparity],
            "ethnicity_voting_patterns": [self.calculate_party_vote_share, self.examine_ethnicity_voting_patterns],
            "income_impact": [
                self.calculate_party_vote_share,
                self.standardize_median_household_income,
                self.explore_income_impact,
            ],
            "candidate": [self.candidate],
        }

    def pre_check_estimands(self, election_id):
        """
        Ensure estimand isn't one of the pre-specified values that are already included
        """
        standard = ["dem_votes", "gop_votes", "total_votes"]
        if not self.check_input_columns(standard, election_id):
            self.create_estimand(None, self.standard)

    def check_input_columns(self, columns, election_id):
        """
        Check that input columns contain all neccessary values for a calculation
        """
        missing_columns = []
        if election_id == "G":
            missing_columns = [col for col in columns if col not in self.data_handler.data.columns]
        elif election_id == "P":
            missing_columns = [
                col for col in columns if col not in self.data_handler.data[self.data_handler.election_id]
            ]
        return len(missing_columns) == 0

    def verify_estimand(self, estimand, election_id):
        """
        Verify which estimands can be formed given a dataset and a list of estimands we would like to create
        """
        if estimand not in self.transformation_map:
            raise ValueError(f"Estimand '{estimand}' is not supported.")
        self.transformations = self.transformation_map[estimand]

        if not self.check_input_columns(
            [col for transform in self.transformations for col in transform.__code__.co_varnames[1:]], election_id
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

    def generate_estimands(self, election_id):
        """
        Main function to generate estimands
        """
        if election_id == "G":
            self.pre_check_estimands(election_id)
            
        # Option 1: Pass in a dict of new functions of estimands we want to build
        if self.given_function_dict != {}:
            for estimand, function in self.given_function_dict.items():
                self.verify_estimand(estimand, election_id)
                self.create_estimand(None, function)

        # Option 2: Pass in a list of estimands we want to build from a pre-set list
        for estimand in self.estimands:
            self.verify_estimand(estimand, election_id)
            self.create_estimand(estimand, None)
        return self.data_handler

    # Transformation methods
    def standard(self):
        """
        Create/overwrite the standard estimands: ["dem_votes", "gop_votes", "total_votes"]
        """
        if "results_turnout" in self.data_handler.data.columns:
            self.data_handler.data["dem_votes"] = self.data_handler.data["results_dem"]
            self.data_handler.data["gop_votes"] = self.data_handler.data["results_gop"]
            self.data_handler.data["total_votes"] = self.data_handler.data["results_turnout"]
        else:
            self.data_handler.data["dem_votes"] = None
            self.data_handler.data["gop_votes"] = None
            self.data_handler.data["total_votes"] = None

    def calculate_margin(self):
        self.data_handler.data["margin"] = self.data_handler.data["dem_votes"] - self.data_handler.data["gop_votes"]

    def calculate_voter_turnout_rate(self):
        self.data_handler.data["voter_turnout_rate"] = (
            self.data_handler.data["total_votes"] / self.data_handler.data["total_gen_voters"]
        )

    def standardize_median_household_income(self):
        mean_income = self.data_handler.data["median_household_income"].mean()
        std_income = self.data_handler.data["median_household_income"].std()
        self.data_handler.data["standardized_income"] = (
            self.data_handler.data["median_household_income"] - mean_income
        ) / std_income

    def create_age_groups(self):
        self.data_handler.data["age_group_under_30"] = np.where(self.data_handler.data["age_le_30"] == 1, 1, 0)
        self.data_handler.data["age_group_30_45"] = np.where(self.data_handler.data["age_geq_30_le_45"] == 1, 1, 0)
        self.data_handler.data["age_group_45_65"] = np.where(self.data_handler.data["age_geq_45_le_65"] == 1, 1, 0)
        self.data_handler.data["age_group_over_65"] = np.where(self.data_handler.data["age_geq_65"] == 1, 1, 0)

    def calculate_party_vote_share(self):
        self.data_handler.data["party_vote_share_dem"] = (
            self.data_handler.data["dem_votes"] / self.data_handler.data["total_votes"]
        )
        self.data_handler.data["party_vote_share_gop"] = (
            self.data_handler.data["gop_votes"] / self.data_handler.data["total_votes"]
        )

    def analyze_education_impact(self):
        self.data_handler.data["education_impact_dem"] = (
            self.data_handler.data["percent_bachelor_or_higher"] * self.data_handler.data["party_vote_share_dem"]
        )
        self.data_handler.data["education_impact_gop"] = (
            self.data_handler.data["percent_bachelor_or_higher"] * self.data_handler.data["party_vote_share_gop"]
        )

    def investigate_gender_turnout_disparity(self):
        self.data_handler.data["gender_turnout_disparity"] = (
            self.data_handler.data["gender_f"] - self.data_handler.data["gender_m"]
        )

    def examine_ethnicity_voting_patterns(self):
        ethnicities = [
            "east_and_south_asian",
            "european",
            "hispanic_and_portuguese",
            "likely_african_american",
            "other",
            "unknown",
        ]
        for ethnicity in ethnicities:
            self.data_handler.data[f"vote_share_{ethnicity}"] = (
                self.data_handler.data[f"ethnicity_{ethnicity}"] * self.data_handler.data["total_votes"]
            )

    def candidate(self):
        election_data = self.data_handler.data[self.data_handler.election_id][0]
        candidate_data = election_data["baseline_pointer"]
        # cand_set = set(candidate_data)
        for cand_name in candidate_data:
            if cand_name != "turnout":
                election_data[cand_name] = candidate_data[cand_name]

    def explore_income_impact(self):
        self.data_handler.data["income_impact_dem"] = (
            self.data_handler.data["standardized_income"] * self.data_handler.data["party_vote_share_dem"]
        )
        self.data_handler.data["income_impact_gop"] = (
            self.data_handler.data["standardized_income"] * self.data_handler.data["party_vote_share_gop"]
        )
