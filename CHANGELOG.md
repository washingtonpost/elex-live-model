# Changelog

# 2.2.5 (11/7/2024)
- fix: hot fixes for the extrapolation step + using the presidential margins to infer a ticket splitting estimate in each house / senate race [#140](https://github.com/washingtonpost/elex-live-model/pull/140)

# 2.2.4 (11/5/2024)
- fix: partial reporting bug [#138](https://github.com/washingtonpost/elex-live-model/pull/138)

# 2.2.3 (11/5/2024)
- chore: adding additional log [#135](https://github.com/washingtonpost/elex-live-model/pull/135)

# 2.2.2 (11/5/2024)
- fix: missing `est_correction` column in `VersionedResults` `DataFrame` in the event of bad data [#131](https://github.com/washingtonpost/elex-live-model/pull/131)

# 2.2.1 (11/1/2024)
- chore: downgrade botocore and s3transfer as per live team dependency [#128](https://github.com/washingtonpost/elex-live-model/pull/128)

## 2.2.0 (11/1/2024)
- chore: condensed logging of non-modeled units [#120](https://github.com/washingtonpost/elex-live-model/pull/120)
- feat: improvements to margin outlier detection [#121](https://github.com/washingtonpost/elex-live-model/pull/121)
- feat: extrapolation rule and improvements to called contest handling [#122](https://github.com/washingtonpost/elex-live-model/pull/122), [#123](https://github.com/washingtonpost/elex-live-model/pull/123), [#124](https://github.com/washingtonpost/elex-live-model/pull/124)
- feat: remove min/max during electoral votes estimation [#125](https://github.com/washingtonpost/elex-live-model/pull/125)

## 2.1.2 (10/24/2024)
- feat: `agg_model_hard_threshold` now defaults to `True`
- feat: using cross-validation to find the optimal OLS `lambda` for use in the `BootstrapElectionModel` is now optional due to the `lambda_` model parameter [#115](https://github.com/washingtonpost/elex-live-model/pull/115)

## 2.1.1 (10/10/2024)
- fix: allow multiple `alpha` values passed in to `ModelClient.get_national_summary_votes_estimates()` and change that method to return a `pandas.DataFrame` [#111](https://github.com/washingtonpost/elex-live-model/pull/111)

## 2.1.0 (09/23/2024)
- fix: model evaluation functions and margin estimand rounding [#94](https://github.com/washingtonpost/elex-live-model/pull/94)
- chore: updated requirements to their latest versions [#95](https://github.com/washingtonpost/elex-live-model/pull/95), [#103](https://github.com/washingtonpost/elex-live-model/pull/103)
- chore: remove duplicate code from the `Estimandizer` class [#96](https://github.com/washingtonpost/elex-live-model/pull/96)
- feat: verbose logging of duplicate units [#97](https://github.com/washingtonpost/elex-live-model/pull/97) and non-modeled units [#105](https://github.com/washingtonpost/elex-live-model/pull/105)
- fix: rare division by zero when creating `normalized_margin` [#98](https://github.com/washingtonpost/elex-live-model/pull/98)
- fix: aggregate model bug [#99](https://github.com/washingtonpost/elex-live-model/pull/99)
- feat: save aggregate model (national summary) predictions to s3 [#100](https://github.com/washingtonpost/elex-live-model/pull/100)
- feat: apply race calls to contest-level predictions in addition to national-level ones [#101](https://github.com/washingtonpost/elex-live-model/pull/101)
- feat: output the mean of the bootstrap as the point prediction [#102](https://github.com/washingtonpost/elex-live-model/pull/102)
- feat: save unit-level turnout predictions from the bootstrap model to s3 [#104](https://github.com/washingtonpost/elex-live-model/pull/104)
- feat: distinguish between genuinely unexpected and non-modeled (non-predictive) units [#105](https://github.com/washingtonpost/elex-live-model/pull/105)
- feat: options to override/control whether or not to allow the model to produce a race call for specified contests [#106](https://github.com/washingtonpost/elex-live-model/pull/106)

## 2.0.3 (11/06/2023)
- fix: fix print bug for aggregate model called states error [#90](https://github.com/washingtonpost/elex-live-model/pull/90)
- chore: add predicted turnout to predictions dataframe [#91](https://github.com/washingtonpost/elex-live-model/pull/91)

## 2.0.2 (11/02/2023)
- fix: allow bootstrap model parameters to be of type int as well as float [#86](https://github.com/washingtonpost/elex-live-model/pull/86)
- fix: pass alpha to national summary client function [#87](https://github.com/washingtonpost/elex-live-model/pull/87)

## 2.0.1 (10/23/2023)
- chore: updating all required packages to their latest versions and addressing some warnings that surfaced during testing [#81](https://github.com/washingtonpost/elex-live-model/pull/81)
- fix: CLI no longer throws an error if `aggregates` are missing or specified with columns that don't exist in the data [#83](https://github.com/washingtonpost/elex-live-model/pull/83)

## 2.0.0 (10/13/2023)
- fix: improved fixed effect features [#69](https://github.com/washingtonpost/elex-live-model/pull/69)
- fix: with the CLI, model-specific parameters passed in as a dictionary [#70](https://github.com/washingtonpost/elex-live-model/pull/70)
- fix: additional logic in the CLI to find the `.env` file [#71](https://github.com/washingtonpost/elex-live-model/pull/71)
- feat: all models are now subclasses of `BaseElectionModel` [#72](https://github.com/washingtonpost/elex-live-model/pull/72)
- feat: ability to create custom estimands [#75](https://github.com/washingtonpost/elex-live-model/pull/75)
- feat: bootstrap model [#76](https://github.com/washingtonpost/elex-live-model/pull/76)
- feat: conformal election model uses new faster quantile regression provided by `elex-solver` [#77](https://github.com/washingtonpost/elex-live-model/pull/77)

## 1.0.11 (08/14/2023)
- fix: upgrade to python 3.10 [#65] (https://github.com/washingtonpost/elex-live-model/pull/65)
- feat: generalize parameter checks [#64] (https://github.com/washingtonpost/elex-live-model/pull/64)
- fix: winsorization test error [#62] (https://github.com/washingtonpost/elex-live-model/pull/65)
- feat: add winsorization option [#58] (https://github.com/washingtonpost/elex-live-model/pull/58)
- fix: clean up tox errors [#57] (https://github.com/washingtonpost/elex-live-model/pull/57)
- refactor: remove model settings [#54] (https://github.com/washingtonpost/elex-live-model/pull/54)
- refactor: dynamically create default aggregates [#53] (https://github.com/washingtonpost/elex-live-model/pull/53)
- feat: add better default aggregates [#52] (https://github.com/washingtonpost/elex-live-model/pull/52)

## 1.0.10 (06/07/2023)
- fix: fixing a fixed effect bug [#35](https://github.com/washingtonpost/elex-live-model/pull/35)
- chore: updated boto3 version [#36](https://github.com/washingtonpost/elex-live-model/pull/36)
- feat: move fixed effect creation from CombinedDataHandler to Featurizer [#38](https://github.com/washingtonpost/elex-live-model/pull/38)
- refactor: removing residual column for nonreporting units [#39](https://github.com/washingtonpost/elex-live-model/pull/39)
- refactor: remove total voters column [#40](https://github.com/washingtonpost/elex-live-model/pull/40)
- feat: add regularization to model [#42](https://github.com/washingtonpost/elex-live-model/pull/42)
- feat: allow selection of fixed effects [#43](https://github.com/washingtonpost/elex-live-model/pull/43)
- fix: update checking fixed effect input [#44](https://github.com/washingtonpost/elex-live-model/pull/44)
- fix: rename regularization parameter [#46](https://github.com/washingtonpost/elex-live-model/pull/46)
- fix: stop unit tests from writing to s3 [#48](https://github.com/washingtonpost/elex-live-model/pull/48)

### 1.0.9 (03/06/2022)
- feat: allow model to return conformalization data [#32](https://github.com/washingtonpost/elex-live-model/pull/32)

### 1.0.8 (01/11/2022)
- fix: fix overwriting non-reporting lower/upper bounds in multiple prediction interval case [#23](https://github.com/washingtonpost/elex-live-model/pull/23)
- fix: fix bug when computing fixed effects [#27](https://github.com/washingtonpost/elex-live-model/pull/27)
- fix: fix overwritting columns when saving conformalization set/bounds to s3 [#28](https://github.com/washingtonpost/elex-live-model/pull/28)

### 1.0.7 (10/28/2022)
- fix: fix mape when uncontested historical baseline [#18](https://github.com/washingtonpost/elex-live-model/pull/18)
- fix: small relative weights for ecos solver [#19](https://github.com/washingtonpost/elex-live-model/pull/18)

### 1.0.6 (10/05/2022)
- fix: Gaussian model bug in lower bound of confidence intervals [#8](https://github.com/washingtonpost/elex-live-model/pull/8)
- fix: save results even with not enough subunits [#13](https://github.com/washingtonpost/elex-live-model/pull/13)
- feat: write an error message for conformity values [#11](https://github.com/washingtonpost/elex-live-model/pull/11)
- feat: automate releases [#10](https://github.com/washingtonpost/elex-live-model/pull/10)
- chore: wrap dataframe in a list to avoid deprecation [#12](https://github.com/washingtonpost/elex-live-model/pull/12)
- fix: gaussian merge [#9](https://github.com/washingtonpost/elex-live-model/pull/9)
- fix: release workflow [#14](https://github.com/washingtonpost/elex-live-model/pull/14)
- feat: add data README [#15](https://github.com/washingtonpost/elex-live-model/pull/15)

### 1.0.5 (09/20/2022)
- fix: Update README link to 2020 model [#2](https://github.com/washingtonpost/elex-live-model/pull/2)
- chore: set up repository [#4](https://github.com/washingtonpost/elex-live-model/pull/4)

### 1.0.4 (09/12/2022)
- chore: update codeowners to public news engineering group [#101](https://github.com/WPMedia/elex-live-model/pull/101)

### 1.0.3 (09/08/2022)
- feat: write conformalization data and gaussian bounds to s3 [#86](https://github.com/WPMedia/elex-live-model/pull/86)

### 1.0.2 (09/07/2022)
- fix: use combined data handler to write results [#96](https://github.com/WPMedia/elex-live-model/pull/96)

### 1.0.1 (09/06/2022)
- fix: set write_data as a class function [#93](https://github.com/WPMedia/elex-live-model/pull/93)

### 1.0.0 (09/01/2022)
- feat: implement fixed effects [#80](https://github.com/WPMedia/elex-live-model/pull/80)
- feat: add option to save preprocessed data [#76](https://github.com/WPMedia/elex-live-model/pull/76)
- feat: create model results class [#82](https://github.com/WPMedia/elex-live-model/pull/82)
- feat: implement skewed sampling [#83](https://github.com/WPMedia/elex-live-model/pull/83)
- fix: bugs in combined data and historical aggregations [#84](https://github.com/WPMedia/elex-live-model/pull/84)
- fix: handle nonreporting unexpected units [#85](https://github.com/WPMedia/elex-live-model/pull/85)
- feat: create integration test [#87](https://github.com/WPMedia/elex-live-model/pull/87)
- fix: bug in fixed effects where a column is zeroes only [#88](https://github.com/WPMedia/elex-live-model/pull/88)
- chore: remove jfrog instructions and update contribution instructions for open source [#89](https://github.com/WPMedia/elex-live-model/pull/89)

### 0.0.8 (07/28/2022)
- chore: rename observed and unobserved to reporting and nonreporting [#68](https://github.com/WPMedia/elex-live-model/pull/68)
- chore: add precommit as a workflow [#70](https://github.com/WPMedia/elex-live-model/pull/70)
- fix: change default parameters to empty lists [#71](https://github.com/WPMedia/elex-live-model/pull/71)
- chore: add unit tests and small fixes [#69](https://github.com/WPMedia/elex-live-model/pull/69)
- fix: use new artifactory secret [#72](https://github.com/WPMedia/elex-live-model/pull/72)
- chore: update README [#73](https://github.com/WPMedia/elex-live-model/pull/73)
- chore: more README updates [#74](https://github.com/WPMedia/elex-live-model/pull/74)
- feat: historic run returns state_data [#75](https://github.com/WPMedia/elex-live-model/pull/75)

### 0.0.7 (05/13/2022)
- fix: bug with observed unexpected subunits [#65](https://github.com/WPMedia/elex-live-model/pull/65)

### 0.0.6 (05/12/2022)
- fix: historical election bug [#60](https://github.com/WPMedia/elex-live-model/pull/60)
- feat: add configurable unreporting options (drop and zero) [#61](https://github.com/WPMedia/elex-live-model/pull/61)
- fix: add 1 to prediction when calculating size of prediction intervals [#63](https://github.com/WPMedia/elex-live-model/pull/63)

### 0.0.5 (04/27/2022)
- chore: update readme and requirements for releases  [#51](https://github.com/WPMedia/elex-live-model/pull/51)
- feat: add subunits reporting column [#52](https://github.com/WPMedia/elex-live-model/pull/52)
- fix: fix test warnings [#53](https://github.com/WPMedia/elex-live-model/pull/53)
- fix: rename LiveDataHandler to MockLiveDataHandler [#55](https://github.com/WPMedia/elex-live-model/pull/55)
- chore: upgrade .append to .concat [#57](https://github.com/WPMedia/elex-live-model/pull/57)
- fix: move random seed to instantiation [#56](https://github.com/WPMedia/elex-live-model/pull/56)
- feat: generate estimates for multiple estimands [#54](https://github.com/WPMedia/elex-live-model/pull/54)

### 0.0.4 (03/21/2022)
- chore: updated pandas version and boto3 version [#48](https://github.com/WPMedia/elex-live-model/pull/48)
- fix: run historical model with new estimand [#46](https://github.com/WPMedia/elex-live-model/pull/46)
- fix: cli allows multiple parameters [#45](https://github.com/WPMedia/elex-live-model/pull/45)

### 0.0.3 (02/24/2022)
- fix: another fix for n_minimum_reporting ([#43])
- chore: release beta version

### 0.0.3-beta.0 (02/23/2022)
- fix: add n_minimum_reporting [#40](https://github.com/WPMedia/elex-live-model/pull/40)

### 0.0.2 (02/22/2022)
- feat: return custom error for not enough subunits reporting [#35](https://github.com/WPMedia/elex-live-model/pull/35)
- chore: rename classes and clean up repo [#36](https://github.com/WPMedia/elex-live-model/pull/36)
- chore: rename repo to elex-live-model [#37](https://github.com/WPMedia/elex-live-model/pull/37)

### 0.0.2-beta.1 (02/14/2022)
- fix: pass in model settings as top-level param [#31] (https://github.com/WPMedia/elex-live-model/pull/31)
- fix: replace dataframe with list of lists [#32] (https://github.com/WPMedia/elex-live-model/pull/32)
- feat: make estimand flexible [#29] (https://github.com/WPMedia/elex-live-model/pull/29)
- fix: refactor requirements [#34] (https://github.com/WPMedia/elex-live-model/pull/34)
- feat: run historical election [#25] (https://github.com/WPMedia/elex-live-model/pull/25)

### 0.0.2-beta.0 (02/11/2022)
- fix: standardizes estimand naming [#21] (https://github.com/WPMedia/elex-live-model/pull/21)
- feat: add random subsetting when running from cli [#22] (https://github.com/WPMedia/elex-live-model/pull/22)
- feat: add beta parameter to increase variance of Gaussian model [#23] (https://github.com/WPMedia/elex-live-model/pull/23)
- feat: write predictions to S3 [#24] (https://github.com/WPMedia/elex-live-model/pull/24)
- fix: make S3 utils class-bawed, add logs, rename env variables [#27] (https://github.com/WPMedia/elex-live-model/pull/27)

### Initial release (02/03/2022)
