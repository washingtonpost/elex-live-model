import math

import numpy as np
import pandas as pd

from elexmodel.handlers.data.BasePreprocessedDataHandler import BasePreprocessedDataHandler


class MockLiveDataHandler(BasePreprocessedDataHandler):
    """
    Handles current data, which we would pull from Dynamo on an election night
    """

    def __init__(
        self,
        election,
        office_id,
        geographic_unit_type,
        estimands,
        historical=False,
        data=None,
        s3_client=None,
        unexpected_units=0,
    ):
        self.unexpected_rows = unexpected_units

        self.shuffle_columns = [
            "postal_code",
            "county_classification",
            "county_fips",
        ]  # columns we may want to sample by
        self.shuffle_dataframe = None

        self.current_reporting_data = None

        super().__init__(
            election,
            office_id,
            geographic_unit_type,
            estimands,
            s3_client=s3_client,
            historical=historical,
            data=data,
        )

    def load_data(self, data):
        columns_to_return = ["postal_code", "geographic_unit_fips"]

        (data, more_columns) = self.estimandizer.add_estimand_results(data, self.estimands, self.historical)
        columns_to_return += more_columns

        self.shuffle_dataframe = data[self.shuffle_columns].copy()
        return data[columns_to_return].copy()

    def shuffle(self, seed=None, upweight={}, enforce=[]):
        """
        Function that allows for random shuffling of geographic units with upweights for certain
        types of counties this makes those geographic units more likely to be picked.
        Also allows a specific ordering by enforcing which geographic units come first.
        seed: int
        upweight: dict of dicts, from category to upweight by to geographic unit identifier to weight
            e.g. {"postal_code": {"AL": 3, "FL": 5}, "county_classification": {"urban": 1000, "rural": 0.3}}
            this would result in urban counties in Alabama being upweighted by 3000
        enforce: list of geographic unit fips that enforce those units to come first
            the order of enforced first elements is random
        """
        probabilities = np.ones((self.data.shape[0],))
        # if upweight is empty this forloop is skipped
        for category in upweight:  # e.g. category == "postal_code"
            weight = upweight[category]
            for value in weight:  # e.g. value == "AL"
                indices = self.data[self.shuffle_dataframe[category] == value].index
                probabilities[indices] = probabilities[indices] * weight[value]

        self.data = self.data.sample(frac=1, random_state=seed, weights=probabilities)

        # get indices of units that must come first
        mask = self.data.geographic_unit_fips.isin(enforce)
        first = self.data[mask]
        last = self.data[~mask]
        self.data = pd.concat([first, last]).reset_index(drop=True)

    def _convert_percent_to_n(self, percent, _round):
        frac = round(percent / 100.0, 2)
        if _round == "up":
            return math.ceil(frac * self.data.shape[0])
        if _round == "down":
            return math.floor(frac * self.data.shape[0])

    def get_percent_fully_reported(self, percent, _round="up"):
        n = self._convert_percent_to_n(percent, _round)
        return self.get_n_fully_reported(n)

    def _include_reporting_unexpected(self):
        """
        Changes data_reporting to include extra unexpected rows depending on unexpected_rows
        Does so by repeating randomly selected rows and changing the unit ids
        (does NOT change the ids of any of the higher level units or of results)
        """
        # creates new ids by appending numbers to existing ids
        fake_ids = [str(i) for i in range(self.unexpected_rows)]
        fake_data = self.data_reporting.sample(frac=1).reset_index(drop=True).head(self.unexpected_rows)

        fake_data["geographic_unit_fips"] = fake_data["geographic_unit_fips"] + fake_ids

        self.data_reporting = pd.concat([self.data_reporting, fake_data])

    def get_n_fully_reported(self, n):
        """
        Return n "fully reported units".
        Returns the first n units as fully reported, if data has been shuffled then random.
        """
        expected_n = n - self.unexpected_rows
        self.data_reporting = self.data[:expected_n].copy()
        self.data_nonreporting = self.data[expected_n:].copy()

        for estimand in self.estimands:
            self.data_reporting[f"raw_results_{estimand}"] = self.data[f"results_{estimand}"]
            self.data_nonreporting[f"raw_results_{estimand}"] = self.data_nonreporting[f"results_{estimand}"]
            self.data_nonreporting[f"results_{estimand}"] = 0  # set these units to not reporting
        self.data_reporting["percent_expected_vote"] = 100
        self.data_nonreporting["percent_expected_vote"] = 0
        if self.unexpected_rows > 0:
            self._include_reporting_unexpected()

        self.current_reporting_data = pd.concat([self.data_reporting, self.data_nonreporting])
        return self.current_reporting_data
