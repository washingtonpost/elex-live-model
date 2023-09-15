from numpy import nan


class EstimandException(Exception):
    pass


RESULTS_PREFIX = "results_"
BASELINE_PREFIX = "baseline_"


class Estimandizer:
    """
    Estimandizer. Generate estimands explicitly.
    """

    def check_and_create_estimands(self, data_df, estimands, historical):
        columns_to_return = []
        for estimand in estimands:
            results_col = f"{RESULTS_PREFIX}{estimand}"

            if results_col not in data_df.columns:
                # will raise a KeyError if a function with the same name as `estimand` doesn't exist
                data_df = globals()[estimand](data_df, RESULTS_PREFIX)

            if historical:
                data_df[results_col] = nan
            else:
                if results_col not in data_df.columns:
                    raise EstimandException("This is missing results data for estimand: ", estimand)

            columns_to_return.append(results_col)

        results_column_names = [x for x in data_df.columns if x.startswith(RESULTS_PREFIX)]
        # If this is not a historical run, then this is a live election
        # so we are expecting that there will be actual results data
        if not historical and len(results_column_names) == 0:
            raise EstimandException("This is not a test election, it is missing results data")

        return data_df, columns_to_return

    def add_estimand_baselines(self, data_df, estimand_baselines, historical, include_results_estimand=False):
        # if we are in a historical election we are only reading preprocessed data to get
        # the historical election results of the currently reporting units.
        # so we don't care about the total voters or the baseline election.

        for estimand, pointer in estimand_baselines.items():
            if pointer is None:
                # when we are creating a new estimand
                pointer = estimand

            baseline_col = f"{BASELINE_PREFIX}{pointer}"

            if baseline_col not in data_df.columns:
                data_df = globals()[estimand](data_df, BASELINE_PREFIX)

            if not historical:
                data_df[f"last_election_results_{estimand}"] = data_df[baseline_col].copy() + 1

        if include_results_estimand:
            data_df, ___ = self.check_and_create_estimands(data_df, estimand_baselines.keys(), historical)

        return data_df


# custom estimands


def party_vote_share_dem(data_df, col_prefix):
    if f"{col_prefix}dem" in data_df.columns and f"{col_prefix}turnout" in data_df.columns:
        numer = f"{col_prefix}dem"
        denom = f"{col_prefix}turnout"
    else:
        numer = "dem"
        denom = "total"

    data_df[f"{col_prefix}party_vote_share_dem"] = data_df.apply(
        lambda x: 0 if x[numer] == 0 or x[denom] == 0 else x[numer] / x[denom], axis=1
    )

    return data_df
