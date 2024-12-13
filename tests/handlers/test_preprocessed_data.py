import os

from elexmodel.handlers.data.PreprocessedData import PreprocessedDataHandler


def test_save(va_governor_county_data, test_path):
    data_handler = PreprocessedDataHandler(
        "2017-11-07_VA_G", "G", "county", ["turnout"], {"turnout": "turnout"}, data=va_governor_county_data
    )
    local_file_path = f"{test_path}/test_dir/data_county.csv"
    if os.path.exists(local_file_path):
        os.remove(local_file_path)
    data_handler.local_file_path = local_file_path
    data_handler.save_data(va_governor_county_data)

    assert os.path.exists(local_file_path)
    os.remove(local_file_path)
    os.rmdir(f"{test_path}/test_dir")
