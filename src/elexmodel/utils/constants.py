from collections import defaultdict

VALID_AGGREGATES_MAPPING = {
    "postal_code": "state_data",
    "county_fips": "county_data",
    "district": "district_data",
    "county_classification": "classification_data",
    "unit": "unit_data",
}

AGGREGATE_ORDER = ["postal_code", "district", "county_classification", "county_fips"]

default_aggregates_elems = {
    "P": ["postal_code", "unit"],
    "S": ["postal_code", "unit"],
    "G": ["postal_code", "unit"],
    "P_county": ["postal_code", "unit"],
    "S_county": ["postal_code", "unit"],
    "G_county": ["postal_code", "unit"],
    "P_precinct": ["postal_code", "unit"],
    "S_precinct": ["postal_code", "unit"],
    "G_precinct": ["postal_code", "unit"],
    "H": ["postal_code", "district", "unit"],
    "Y": ["postal_code", "district", "unit"],
    "Z": ["postal_code", "district", "unit"],
    "H_county-district": ["postal_code", "district", "unit"],
    "Y_county-district": ["postal_code", "district", "unit"],
    "Z_county-district": ["postal_code", "district", "unit"],
    "H_precinct-district": ["postal_code", "district", "unit"],
    "Y_precinct-district": ["postal_code", "district", "unit"],
    "Z_precinct-district": ["postal_code", "district", "unit"],
}
DEFAULT_AGGREGATES = defaultdict(list, **default_aggregates_elems)
