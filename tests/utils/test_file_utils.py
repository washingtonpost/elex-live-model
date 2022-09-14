import pandas as pd

from elexmodel.utils.file_utils import convert_df_to_csv, get_directory_path


def test_get_directory_path():
    test_directory_path = get_directory_path()
    assert str(test_directory_path).endswith("elex-live-model")  # repo name


def test_convert_df_to_csv():
    data = {"col1": [1, 2], "col2": [3, 4]}
    test_df = pd.DataFrame(data=data)
    test_csv = convert_df_to_csv(test_df)
    assert test_csv == "col1,col2\n1,3\n2,4\n"
