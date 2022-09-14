from unittest.mock import create_autospec

import boto3
import pytest

from elexmodel.handlers.s3 import S3JsonUtil
from elexmodel.utils.file_utils import S3_FILE_PATH


@pytest.fixture
def test_s3_util():
    return S3JsonUtil("elex-models-dev-fake", create_autospec(boto3.client("s3")))


def test_s3_put(test_s3_util):
    test_s3_util.put("test_filename", {"key": "value"})
    test_s3_util.client.put_object.assert_called_with(
        Bucket="elex-models-dev-fake", Body='{"key": "value"}', ContentType="application/json", Key="test_filename.json"
    )


def test_s3_get_file_path_preprocessed(test_s3_util):
    file_type = "preprocessed"
    path_info = {"election_id": "2017-11-07_VA_G", "office": "G", "geographic_unit_type": "county"}
    assert test_s3_util.get_file_path(file_type, path_info) == f"{S3_FILE_PATH}/2017-11-07_VA_G/data/G/data_county.csv"


def test_s3_get_file_path_config(test_s3_util):
    file_type = "config"
    path_info = {"election_id": "2017-11-07_VA_G", "office": "G", "geographic_unit_type": "county"}
    assert test_s3_util.get_file_path(file_type, path_info) == f"{S3_FILE_PATH}/2017-11-07_VA_G/config/2017-11-07_VA_G"
