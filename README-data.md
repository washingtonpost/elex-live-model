# Model Data

## Overview
 
The model needs two user-generated inputs before election night: [data at the unit level](#unit-level-data) and a [config file](#config-files). Sample model inputs can be found in `[tests/fixtures/](https://github.com/washingtonpost/elex-live-model/tree/develop/tests/fixtures)`. Your column headers should match how they appear in our sample data, except when specifying your demographic fields and candidate names (more below). The model is set up to import data from S3, but you can also use locally stored files with these steps:

- create a `config` and a `data` directory at the root of the model repo
- put the config json in the `config` directory. The json’s name should be the election id (e.g. `config/2016-11-08_USA_G.json`).
- put the preprocessed data in the `data` directory. The path of the preprocessed data should be `data/{election_id}/{office_id}/data_{geographic_unit_type}.csv` (e.g. `data/2016-11-08_SA_G/P/data_county.json`). The format for `election_id``,` `office_i``d, and` `geographic_unit_type` are explained in the README.

 
### Unit-Level Data
 
The model can handle any set of geographic units, but is currently operationalized to work at the county or precinct level (if you’d like to use a different level, please reach out to the elections engineering team at elections@washpost.com for directions on what to change in the code). Unit-level data consists of **demographic data** and **baseline results** from a prior election. Sample data is in `t``ests/fixtures/data`. 
 
You can pick which demographic information — or covariates — are relevant for your case (age, income, ethnicity, education etc.) We typically get this data from the 5-year ACS using its [Census API](https://www.census.gov/data/developers/data-sets/acs-5year.html) when running the data on the county level, or the L2 national voter file when running the model on precincts. If the only data fields you need are included in the decennial census you can call from [its API](https://www.census.gov/data/developers/data-sets/decennial-census.html) instead. (It is also possible to run the mode without any covariates at all. This is equivalent to applying uniform swing on non-reporting units: the predicted change from the baseline in each *non-reporting* unit would be the median change over the *reporting* units in this case).
 
Election baseline data is not used as a covariate in the model regression, but does serve as the baseline from which new predictions deviate (for example if the model predicts there will be a 2% increase in turnout in a particular county, the 2% is an increase over the county’s prior turnout in the baseline election). Therefore we need past election results joined to the same units as the demographic data. Choose a baseline election that has a strong relationship with the current one. For example if you’re modeling a senate race, use the last senate race in the state. This is more complicated for modeling House races, for which we recommend using a recent statewide baseline election. There are many sources for election returns, included Secretary of State websites, [Harvard Dataverse/MEDSL](https://dataverse.harvard.edu/dataverse/medsl_election_returns) and the [MIT Election Data + Science Lab](https://electionlab.mit.edu/data) (don’t forget to credit your data source). 
 
For both demographic and baseline data you may need to *prorate* onto your geographic units. For example, if you have demographic data at the block-group level from the ACS, but need to put it onto precincts, you may need to prorate your data first down to blocks and then aggregate to the precinct level. We recommend the [maup](https://github.com/mggg/maup) python package for proration help. You’ll also need shapefiles to do proration and we recommend [Census Tiger/Line](https://www.census.gov/cgi-bin/geo/shapefiles/index.php) whenever possible.

The final preprocessed data — the csv of combined demographic and baseline data — can also have categorial variables added. These can be anything you like, but we recommend using pre-defined ones (such as `county_fips`, `postal_code` or `county_classification` — the last one can be used for any arbitrary unit level information). To use those in the model include them as `fixed_effects` when executing the model and in the config file.

### Config Files


Use the sample configs in `tests/fixtures/configs`.
 
| Name  | Description                                                                                | Type    | 
| ----- | ------------------------------------------------------------------------------------------ | --------- 
| `office`               | election office, e.g. `S` for Senate                                      | string  |
| `states`               | state(s) to run the model on                                              | list    |
| `geographic_unit_types`| subunit types for which you have prepared preprocessed data, e.g `county` | list  | 
| `historical_election`  | past elections (which also need prepared data), can be used to test whether a group of units that are already reporting results would have historically been a good sample to predict the non-reporting units. can be empty | list   |
| `features`             | demographic covariates, must exactly match the column headers in your data for your demographic fields                                               | list  |
| `aggregates`           | unit that the model should aggregate to, must include `unit` and `postal_code`, can also include `county_classification`  | list | 
| `fixed_effects`        | potential fixed effects (discrete features that the model can use). can be `county_classification` or `county_fips`, though `county_fips` makes less sense when `geographic_unit_type` is `county` since every row would have a different fixed effect. | list |
| `baseline_pointer`     | map of candidate (or parties) to whatever candidate (or party) from the baseline election that has teh strongest relationship over subunits | dictionary


 
## Running the model on live election night data
At the Washington Post, we run this model by using the Python package in our broader elections data pipeline. Refer to [our blog post](https://washpost.engineering/were-open-sourcing-our-live-election-night-model-a21bcb2a46c6) for more details on our elections pipeline. For a more lightweight method of deploying the model, you can invoke the Python package in a Jupyter notebook.

## Other tips

- We recommend running the nonparametric model as the default. When running the model with precincts **and** when you are interested in statewide outcomes, we recommend switching to the gaussian model.
- Unless you want to save the model data or output to s3, make sure that the `save_output` parameter is an empty array.

