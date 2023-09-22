import numpy as np

RESULTS_PREFIX = "results_"
BASELINE_PREFIX = "baseline_"


class Estimandizer:
    """
    Estimandizer. Generate estimands explicitly.
    """

    def add_estimand_results(self, data_df, estimands, historical):
        columns_to_return = []
        turnout_col = f"{RESULTS_PREFIX}turnout"

        for estimand in estimands:
            results_col = f"{RESULTS_PREFIX}{estimand}"
            additional_columns_added = []
            if results_col not in data_df.columns:
                # will raise a KeyError if a function with the same name as `estimand` doesn't exist
                try:
                    data_df, additional_columns_added = globals()[estimand](data_df, RESULTS_PREFIX)
                except KeyError as e:
                    if historical:
                        # A historical run is one where we pull in data from a past election
                        # and use it as though it were a current, live election.
                        # Live elections don't have results since they haven't happened yet.
                        # However, when we run the model from the CLI, we use the
                        # MockLiveDataHandler to generate the current set of reporting units,
                        # and that data handler expects a results_ column for every estimand specified.
                        # Hence, this is the only special case in which we'd want to add
                        # an empty results_ column.
                        data_df[results_col] = np.nan
                    else:
                        # If this is not a historical run, then this is a live election
                        # so we are expecting that there will be actual results data
                        raise e

            columns_to_return.extend([results_col] + additional_columns_added)

        # always adding turnout since we will want to generate weights
        # but if turnout is the estimand, then we only want to add it once
        if turnout_col not in columns_to_return:
            columns_to_return.append(turnout_col)

        data_df = self.add_weights(data_df, RESULTS_PREFIX)

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
                data_df, __ = globals()[estimand](data_df, BASELINE_PREFIX)

            if not historical:
                data_df[f"last_election_results_{estimand}"] = data_df[baseline_col].copy() + 1

        if include_results_estimand:
            # In the situation where we're combining Preprocessed and (Mock)Live data,
            # we are most likely measuring performance, and it's possible we may not
            # have the baseline_ columns or the results_ columns that we need.
            # Since this method is only called by the PreprocessedDataHandler, for historical runs,
            # we need to add the results from the historical election as well.
            data_df, ___ = self.add_estimand_results(data_df, estimand_baselines.keys(), historical)

        data_df = self.add_weights(data_df, BASELINE_PREFIX)
        return data_df

    def add_weights(self, data_df, col_prefix):
        data_df[f"{col_prefix}weights"] = data_df[f"{col_prefix}turnout"]
        return data_df

    def add_turnout_factor(self, data_df):
        # posinf and neginf are also set to zero because dividing by zero can lead to nan/posinf/neginf depending
        # on the type of the numeric in the numpy array. Assume that if baseline_weights is zero then turnout
        # would be incredibly low in this election too (ie. this is effectively an empty precinct) and so setting
        # the turnout factor to zero is fine
        data_df["turnout_factor"] = np.nan_to_num(
            data_df.results_weights / data_df.baseline_weights, nan=0, posinf=0, neginf=0
        )
        return data_df


# custom estimands


def party_vote_share_dem(data_df, col_prefix):
    numer = f"{col_prefix}dem"
    denom = f"{col_prefix}turnout"

    data_df[f"{col_prefix}party_vote_share_dem"] = data_df.apply(
        lambda x: 0 if x[numer] == 0 or x[denom] == 0 else x[numer] / x[denom], axis=1
    )

    return data_df, []
