import pandas as pd

# Model new baseline for 2022 senate model
# pull in county-level data for blending
# we are interested in county-town level data for statewide elections
# over multiple years

# president
pres_12 = pd.read_csv(
    "../../elex-live-model-data-processing/2012-11-06_USA_G/final_processed_data/P/data_county.csv",
    dtype={"geographic_unit_fips": str},
)
pres_16 = pd.read_csv(
    "../../elex-live-model-data-processing/2016-11-08_USA_G/final_processed_data/P/data_county.csv",
    dtype={"geographic_unit_fips": str},
)
pres_20 = pd.read_csv(
    "../../elex-live-model-data-processing/2020-11-03_USA_G/final_processed_data/P/data_county.csv",
    dtype={"geographic_unit_fips": str},
)

# senate
sen_16 = pd.read_csv(
    "../../elex-live-model-data-processing/2016-11-08_USA_G/final_processed_data/S_county/data_county.csv",
    dtype={"geographic_unit_fips": str},
)

# governor
gov_18 = pd.read_csv(
    "../../elex-live-model-data-processing/2018-11-06_USA_G/final_processed_data/G_county/data_county.csv",
    dtype={"geographic_unit_fips": str},
)
gov_16 = pd.read_csv(
    "../../elex-live-model-data-processing/2016-11-08_USA_G/final_processed_data/G/data_county.csv",
    dtype={"geographic_unit_fips": str},
)

list_of_dfs = [pres_12, pres_16, pres_20, sen_16, gov_18, gov_16]
all_results = pd.concat(list_of_dfs).reset_index(drop=True)

all_results = all_results.drop(["baseline_dem", "baseline_gop", "baseline_turnout"], axis=1)
all_results["results_dem_share"] = all_results["results_dem"] / (all_results["results_turnout"])
all_results["results_gop_share"] = all_results["results_gop"] / (all_results["results_turnout"])
all_results["results_turnout_share"] = all_results["results_turnout"] / (all_results["total_people"])
all_results = all_results.groupby(["geographic_unit_fips"]).mean().reset_index(drop=False)
all_results = all_results.rename(
    columns={
        "results_dem_share": "baseline_dem",
        "results_gop_share": "baseline_gop",
        "results_turnout_share": "baseline_turnout",
    }
)

# replace new baseline into 2022 preprocessed data
new_data = all_results[["geographic_unit_fips", "baseline_dem", "baseline_gop", "baseline_turnout"]]
new_data.to_csv("new_baseline_data_S_county_2022.csv", index=False)
