import json
import logging

import boto3

from elexmodel.utils.file_utils import S3_FILE_PATH

LOG = logging.getLogger(__name__)


class S3Util:
    def __init__(self, bucket_name, client=None):
        self.bucket_name = bucket_name
        if not client:
            client = boto3.client("s3")
        self.client = client

    def get(self, filename, **kwargs):
        kwargs.setdefault("Bucket", self.bucket_name)
        kwargs.setdefault("Key", filename)
        LOG.debug("[%s] Retrieving %s @ %s from S3", self.bucket_name, filename, kwargs.get("VersionId", "latest"))
        result = self.client.get_object(**kwargs)
        LOG.info("[%s] Retrieved %s from S3 (LastModified: %s)", self.bucket_name, filename, result["LastModified"])
        return result["Body"]

    def put(self, filename, data, **kwargs):
        kwargs.setdefault("ContentType", "application/json")
        kwargs.setdefault("Body", data)
        kwargs.setdefault("Bucket", self.bucket_name)
        kwargs.setdefault("Key", filename)
        LOG.debug("[%s] Exporting %s to S3", self.bucket_name, filename)
        if self.client.put_object(**kwargs):
            LOG.info("[%s] Exported %s to S3", self.bucket_name, filename)
        else:
            raise Exception(f"Unable to save content in S3 ({filename})")

    def get_file_path(self, file_type, path_info):
        if file_type == "preprocessed":
            file_path = f'{S3_FILE_PATH}/{path_info["election_id"]}/data/{path_info["office"]}/data_{path_info["geographic_unit_type"]}.csv'
        elif file_type == "config":
            file_path = f'{S3_FILE_PATH}/{path_info["election_id"]}/config/{path_info["election_id"]}'
        return file_path


class S3JsonUtil(S3Util):
    def put(self, filename, data, **kwargs):
        if not isinstance(data, str):
            data = json.dumps(data)
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        super().put(filename, data, **kwargs)

    def get(self, filename, load=True, **kwargs):
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        data = super().get(filename, **kwargs)
        if load:
            return json.load(data)
        return data


class S3CsvUtil(S3Util):
    def put(self, filename, data, **kwargs):
        """
        Put a CSV to S3
        """
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        kwargs.setdefault("ContentType", "text/csv")
        super().put(filename, data, **kwargs)

    def get(self, filename, load=True, **kwargs):
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        data = super().get(filename, **kwargs)
        csv = data.read().decode("utf-8")
        return csv
