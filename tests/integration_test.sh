#!/bin/bash
set -e

# Move data to local root directory
cp -r tests/fixtures/data .
cp -r tests/fixtures/config .

echo "Running VA governor 2017 precinct model"
elexmodel 2017-11-07_VA_G --estimands=dem --office_id=G --geographic_unit_type=precinct --pi_method=gaussian --percent_reporting 10

echo "Running VA Governor 2017 county model"
elexmodel 2017-11-07_VA_G --estimands=dem --office_id=G --geographic_unit_type=county --percent_reporting 50

echo "Running VA Assembly 2017 precinct-district model, including district aggregates"
elexmodel 2017-11-07_VA_G --estimands=dem --office_id=Y --geographic_unit_type=precinct-district --percent_reporting 10 --aggregates=district --unexpected_units=10

echo "Running VA Assembly 2017 county-district model"
elexmodel 2017-11-07_VA_G --estimands=dem --office_id=Y --geographic_unit_type=county-district --percent_reporting 50 --aggregates=district

echo "Running VA Governor 2017 precinct model with county classification fixed effects and ethnicity features"
elexmodel 2017-11-07_VA_G --estimands=dem --office_id=G --geographic_unit_type=precinct --aggregates=county_classification --aggregates=postal_code --fixed_effects=county_classification --percent_reporting 10 --features=ethnicity_european --features=ethnicity_hispanic_and_portuguese

echo "Running VA Governor 2017 precinct model as historical with county classification aggregates"
elexmodel 2021-11-02_VA_G --estimands=dem --office_id=G --geographic_unit_type=precinct --percent_reporting 60 --historical --aggregates=county_classification

echo "Running VA Assembly 2017 precinct-district as historical model with district aggregates"
elexmodel 2021-11-02_VA_G --estimands=dem --office_id=Y --geographic_unit_type=precinct-district --percent_reporting 60 --historical --aggregates=district

echo "Running AZ 2021 Republican Senate primary precinct model"
elexmodel 2020-08-04_AZ_R --estimands mcsally_61631 --office_id=S --geographic_unit_type=precinct --percent_reporting 20 --aggregates=postal_code
