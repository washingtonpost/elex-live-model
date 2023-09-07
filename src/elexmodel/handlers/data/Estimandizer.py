from numpy import nan


class EstimandException(Exception):
    pass


class Estimandizer:
    """
    Estimandizer. Generate estimands explicitly.
    """

    def __init__(self):
        self.RESULTS_PREFIX = "results_"
        self.BASELINE_PREFIX = "baseline_"

    def check_and_create_estimands(self, data_df, estimands, historical):
        columns_to_return = []

        for estimand in estimands:
            results_col = f"{self.RESULTS_PREFIX}{estimand}"
            baseline_col = f"{self.BASELINE_PREFIX}{estimand}"

            if historical:
                data_df[results_col] = nan
            else:
                if results_col not in data_df.columns:
                    raise EstimandException("This is missing results data for estimand: ", estimand)
            columns_to_return.append(results_col)

            if baseline_col not in data_df.columns:
                raise EstimandException("Coming soon!")

        results_column_names = [x for x in data_df.columns if x.startswith(self.RESULTS_PREFIX)]
        # If this is not a historical run, then this is a live election
        # so we are expecting that there will be actual results data
        if not historical and len(results_column_names) == 0:
            raise EstimandException("This is not a test election, it is missing results data")

        return (data_df, columns_to_return)

    def add_estimand_baselines(self, data_df, estimand_baselines):
        for estimand, pointer in estimand_baselines.items():
            baseline_name = f"{self.BASELINE_PREFIX}{pointer}"
            # Adding one to prevent zero divison
            data_df[f"last_election_results_{estimand}"] = data_df[baseline_name].copy() + 1

        return data_df

    # def calculate_party_vote_share(self):
    #     """
    #     Create all possible estimands related to party vote shares
    #     """
    #     if "dem_votes" in self.data_handler.data.columns and "total_votes" in self.data_handler.data.columns:
    #         self.data_handler.data["party_vote_share_dem"] = (
    #             self.data_handler.data["dem_votes"] / self.data_handler.data["total_votes"]
    #         )
    #     if "gop_votes" in self.data_handler.data.columns and "total_votes" in self.data_handler.data.columns:
    #         self.data_handler.data["party_vote_share_gop"] = (
    #             self.data_handler.data["gop_votes"] / self.data_handler.data["total_votes"]
    #         )
    #     if (
    #         "baseline_dem_votes" in self.data_handler.data.columns
    #         and "baseline_total_votes" in self.data_handler.data.columns
    #     ):
    #         self.data_handler.data["baseline_party_vote_share_dem"] = (
    #             self.data_handler.data["baseline_dem_votes"] / self.data_handler.data["baseline_total_votes"]
    #         )
    #     if (
    #         "baseline_gop_votes" in self.data_handler.data.columns
    #         and "baseline_total_votes" in self.data_handler.data.columns
    #     ):
    #         self.data_handler.data["baseline_party_vote_share_gop"] = (
    #             self.data_handler.data["baseline_gop_votes"] / self.data_handler.data["baseline_total_votes"]
    #         )

    # def candidate(self):
    #     """
    #     Create estimands for a given candidate in a primary election
    #     """
    #     # cands_old = re.findall(r'results_(\w+)_(\d+)', self.data_handler.data.columns)
    #     r = re.compile("results_*")
    #     cands = list(filter(r.match, self.data_handler.data.columns))
    #     # cand_set = set(candidate_data)
    #     for cand_name in cands:
    #         new_name = cand_name[8:]
    #         self.data_handler.data[new_name] = self.data_handler.data[cand_name]
