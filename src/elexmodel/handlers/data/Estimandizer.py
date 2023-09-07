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
            baseline_col = f"{BASELINE_PREFIX}{estimand}"

            if baseline_col not in data_df.columns:
                # will raise a KeyError if a function with the same name as `estimand` doesn't exist
                data_df = globals()[estimand](data_df)
                data_df[results_col] = data_df[baseline_col].copy()

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

        return (data_df, columns_to_return)

    def add_estimand_baselines(self, data_df, estimand_baselines):
        for estimand, pointer in estimand_baselines.items():
            if pointer is None:
                # should only happen when we're going to create a new estimand
                pointer = estimand

            baseline_name = f"{BASELINE_PREFIX}{pointer}"

            if baseline_name not in data_df.columns:
                # will raise a KeyError if a function with the same name as `pointer` doesn't exist
                data_df = globals()[pointer](data_df)

            # Adding one to prevent zero divison
            data_df[f"last_election_results_{estimand}"] = data_df[baseline_name].copy() + 1

        return data_df


# custom estimands


def party_vote_share_dem(data_df):
    data_df[f"{BASELINE_PREFIX}party_vote_share_dem"] = (
        data_df[f"{BASELINE_PREFIX}dem"] / data_df[f"{BASELINE_PREFIX}turnout"]
    )
    return data_df
