# elex-live-model

The Washington Post uses this live election model to generate estimates of the number of outstanding votes on an election night based on the current results of the race. It is agnostic to the quantity that it is estimating. For general elections, we also generate estimates of the partisan split of outstanding votes, and for primaries, we split estimates by candidate.

Generally, the model works by comparing the current results to a historical baseline and regressing on the difference using demographic features. We use [quantile regression](https://en.wikipedia.org/wiki/Quantile_regression) as the underlying model and [conformal prediction](https://arxiv.org/abs/2107.07511) to produce our uncertainty estimates.

The first iteration of this model is written in R in [this repo](https://github.com/washingtonpost/2020-election-night-model).

## Installation

* We recommend that you set up a virtualenv and activate it (IE ``mkvirtualenv elex-model`` via http://virtualenvwrapper.readthedocs.io/en/latest/).
* Run ``pip install elex-model``

## Usage

We can run the model with a CLI or with Python.

We can use the model to generate current estimates or for a historical evaluation. Historical evaluation means running the "current reporting" subunits with data from a previous election, and then calculating the error that the current set of reporting subunits would have given us. This allows to test how representative the currently reporting subunits are.

**See more information on how to pass data to the model in the [data README](https://github.com/washingtonpost/elex-live-model/blob/develop/README-data.md).**

### CLI

The CLI is for local development and testing purposes only. We cannot run a live election through the CLI because it pulls vote counts from data files located either in S3 or locally. It does not retrieve current data from the Dynamo database of election results.

The CLI takes an election ID, estimands, office ID, and a geographic unit type. If you're running the model with local data files, they should be located at `elex-live-model/data/{election_id}/{office_id}/data_{geographic_unit_type}.csv`.  Otherwise, the model will attempt to find the data files in S3. 

Pass in a command like this:
```
elexmodel 2017-11-07_VA_G --estimands=dem --office_id=G --geographic_unit_type=county
```

You can also pass in multiple estimands:
```
elexmodel 2017-11-07_VA_G --estimands=dem --estimands=turnout --office_id=G --geographic_unit_type=county --percent_reporting 40
```

If you want to run a test with some nonreporting subunits, you can use the `--percent_reporting` cli parameter:
```
elexmodel 2017-11-07_VA_G --estimands=dem --office_id=G --geographic_unit_type=county --percent_reporting 40
```

#### Historical election

If you want to run a historical election, you can use the `--historical` flag. For this to succeed, the election must have historical data already prepared.
```
elexmodel 2021-11-02_VA_G --estimands=dem --office_id=G --geographic_unit_type=county --percent_reporting 60 --historical
```

### Parameters

Parameters for the CLI tool:

| Name                 | Type    | Acceptable values |
|----------------------|---------|-------------------|
| election_id          | string  | `YYYY-MM-DD_{geography}_{election_type}` geography is the state or `USA` and election type is `G` for general or `'P'` for primary |
| estimands            | list    | party name (i.e. `dem`, `gop`) or turnout in a general; `{candidate_last_name}_{polID}` in a primary |
| office_id            | string  | Presidential (`P`), Senate (`S`), House (`H`), Governor (`G`), state Senate (`Z`), state House (`Y`) |
| geographic_unit_type | string  | `county`, `precinct`, `county-district`, or `precinct-district` |
| percent_reporting    | numeric | 0-100 |
| historical           | flag    |       |
| features             | list    | features to include in the model |
| fixed_effects        | list    | `postal_code`, `county_classification` or `county_fips`, but really any prepared categorical variable |
| aggregates           | list    | list of geographies for which to calculate predictions beyond the original `postal_code`, `county_fips`, `district`, `county_classification` |
| pi_method            | string  | method for constructing prediction intervals (`nonparametric` or `gaussian`) |
| beta                 | numeric | variance inflation for `gaussian` model; | 
| robust               | flag    | flag for larger set of prediction intervals in the nonparametric case |
| save_output          | list    | `results`, `data`, `config` |
| unexpected_units     | int     | number of unexpected units to simulate; only used for testing and does not work with historical run |

Note: When running the model with multiple fixed effects, make sure they are not linearly dependent. For example, `county_fips` and `county_classification` are linearly dependent when run together. That's because every county is in one county class, so all the fixed effect columns of the counties in the county class sum up to the fixed effect column of that county class.

### Python

This is the class and function that invokes the general function to generate estimates. You can install `elex-model` as a Python package and use this code snippet in other projects.

```
from elexmodel.client import ModelClient

model_client = ModelClient()
model_response = model_client.get_estimates(
  current_results,
  election_id,
  office,
  estimand, 
  prediction_intervals,
  percent_reporting_threshold,
  geographic_unit_type,
)
```

#### Historical election

This is the class and function that invokes a historical evaluation. You can install `elex-model` as a Python package and use this code snippet in other projects.
```
from elexmodel.client import HistoricalModelClient

historical_model_client = HistoricalModelClient()
model_evaluation = historical_model_client.get_historical_evaluation(
  current_data,
  election_id,
  office,
  estimand,
  prediction_intervals,
  percent_reporting_threshold,
  geographic_unit_type
)
```

## Development

We welcome contributions to this repo. Please open a Github issue for any issues or comments you have. 

### Installation

Clone the repository and install the requirements:

```
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
```

Create a .env file in the top directory and add the below variables. Assuming your S3 bucket and path roots are named `elex-models`, set these as your variables:

```
  APP_ENV=local
  DATA_ENV=dev
  MODEL_S3_BUCKET=elex-models
  MODEL_S3_PATH_ROOT=elex-models
```

### Testing

* ``pip install -r requirements-dev.txt``
* ``tox``

We also have a `requirements-test.txt` file which is used for running unit tests only. It is installed automatically as part of installing `requirements-dev.txt`. 

### Precommit

To run precommit hooks for linting, run:
```
pre-commit run --all-files
```

### Release

To release a new version manually: 
- Decide what the next version will be per semantic versioning: `X.X.X`
- Make a new branch from develop called `release/X.X.X`
- Update the version in `setup.py`
- Update the changelog with all the chnages that will be included in the release
- Commit your updates and open a PR against main
- Once the PR is merged, tag main (or develop for a beta release) with the version's release number (`git tag X.X.X`) and push that tag to github (`git push --tags`)
- Merge main into develop

Then, we need to release this version to PyPi.This repository has a Github Action workflow that automatically builds and releases the latest version to TestPyPi and PyPi on pushes to `main`. However, to release to PyPi manually:
- Generate a distribution archive:
  - Make sure `requirements-dev.txt` is installed
  - Run `python3 -m pip install --upgrade build` to install `build`
  - Run `python3 -m build`. This should generate two files in the `dist/` directory.
  - Check to make sure the correct version is installed in the `dist/` folder that should now exist at the base of the repo folder. If you've previously run these commands locally for an earlier version, you may need to delete the older files in `dist/` order to upload them correctly in the next step. You can just delete the entire `dist/` folder and run the above command again.
- Upload the distribution archive:`
  - Run `python3 -m pip install --upgrade twine`
  - Upload to TestPyPi with `python3 -m twine upload --repository testpypi dist/*`
  - Upload to PyPi `python3 -m twine upload dist/*`


## Further Reading

We have published multiple resources to share our progress.

* October 2020: ["How The Washington Post Estimates Outstanding Votes for the 2020 Presidential Election"](https://s3.us-east-1.amazonaws.com/elex-models-prod/2020-general/write-up/election_model_writeup.pdf)
* November 1, 2020: ["What the Post’s election results will look like"](https://www.washingtonpost.com/politics/2020/11/01/post-election-model-results/)
* November 2020: [Github repository for the original election night model used in the 2020 elections](https://github.com/washingtonpost/2020-election-night-model)
* December 2020: ["An Update To The Washington Post Election Night Model"](https://s3.us-east-1.amazonaws.com/elex-models-prod/elex-models-prod/2020-general/write-up/election_model_writeup_update1.pdf)
* [2020 General election night model open sourced repository](https://github.com/washingtonpost/2020-election-night-model)
* February 21, 2021: ["How The Washington Post Estimates Outstanding Votes for the 2020 Presidential Election"](https://washpost.engineering/how-the-washington-post-estimates-outstanding-votes-for-the-2020-presidential-election-3f82f8415eda)
* November 2, 2021: ["How The Washington Post will model possible outcomes in the Virginia governor’s race"](https://www.washingtonpost.com/elections/2021/11/02/election-model-explained/)
* May 17, 2022: ["How the Washington Post’s election night model works"](https://www.washingtonpost.com/politics/2022/05/17/post-election-night-model/)
* September 14, 2022: ["We're open sourcing our live election night model"](https://washpost.engineering/were-open-sourcing-our-live-election-night-model-a21bcb2a46c6)