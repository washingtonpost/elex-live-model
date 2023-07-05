import numpy as np


class Estimandizer:
    """
    Generate estimands explicitly.
    """

    def __init__(self, data_handler, estimands):
        self.data_handler = data_handler
        self.estimands = estimands
        self.transformations = []

    def check_estimand(self, estimand):
        already_included = ["dem_votes", "gop_votes", "total_votes"]
        if estimand in already_included:
            return False
        return True

    def check_input_columns(self, columns):
        missing_columns = [col for col in columns if col not in self.data_handler.data.columns]
        return len(missing_columns) == 0

    def predict_estimands(self, estimand):
        transformation_map = {
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
        }
        if estimand not in transformation_map:
            raise ValueError(f"Estimand '{estimand}' is not supported.")
        self.transformations = transformation_map[estimand]
        if not self.check_input_columns(
            [col for transform in self.transformations for col in transform.__code__.co_varnames[1:]]
        ):
            return []
        return self.transformations

    def create_estimand(self, estimand):
        if estimand in self.transformations:
            transformation_func = self.transformations[estimand]
            new_column = transformation_func()
            self.data_handler.data[estimand] = new_column

    def generate_estimands(self):
        for estimand in self.estimands:
            if self.check_estimand(estimand):
                self.create_estimand(estimand)
        return self.data_handler

    # Transformation methods
    def calculate_margin(self):
        self.data_handler.data["margin"] = self.data_handler.data["results_dem"] - self.data_handler.data["results_gop"]

    def calculate_voter_turnout_rate(self):
        self.data_handler.data["voter_turnout_rate"] = (
            self.data_handler.data["results_turnout"] / self.data_handler.data["total_gen_voters"]
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
            self.data_handler.data["results_dem"] / self.data_handler.data["results_turnout"]
        )
        self.data_handler.data["party_vote_share_gop"] = (
            self.data_handler.data["results_gop"] / self.data_handler.data["results_turnout"]
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
                self.data_handler.data[f"ethnicity_{ethnicity}"] * self.data_handler.data["results_turnout"]
            )

    def explore_income_impact(self):
        self.data_handler.data["income_impact_dem"] = (
            self.data_handler.data["standardized_income"] * self.data_handler.data["party_vote_share_dem"]
        )
        self.data_handler.data["income_impact_gop"] = (
            self.data_handler.data["standardized_income"] * self.data_handler.data["party_vote_share_gop"]
        )

    def generate_estimand(self):
        for estimand in self.estimands:
            if self.check_estimand(estimand):
                self.create_estimand(estimand)
