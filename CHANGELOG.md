# Changelog

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
