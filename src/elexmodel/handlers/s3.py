import io
import json
import queue

import boto3
import pandas as pd
from botocore.session import get_session
from dateutil import tz
from s3transfer.manager import TransferManager
from s3transfer.subscribers import BaseSubscriber

from elexmodel.logger import getModelLogger
from elexmodel.utils.file_utils import S3_FILE_PATH

LOG = getModelLogger()


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


class S3VersionUtil:
    def __init__(self, bucket_name, start_date=None, end_date=None, tz="America/New_York"):
        self.bucket_name = bucket_name
        self.s3_client = get_session().create_client("s3")
        self.manager = TransferManager(self.s3_client)

        self.start_date = start_date
        self.end_date = end_date
        self.tz = tz

    def list_versions(self, path, **kwargs):
        """
        path here is the full path - constucted via f"{base_path}/{path}"
        in previous implementations of this
        """
        response = self.s3_client.list_object_versions(Bucket=self.bucket_name, Prefix=path, **kwargs)

        versions = []
        if "Versions" in response:
            versions = response["Versions"]

        if (
            response["IsTruncated"]
            and len(versions) > 0
            and (self.start_date is None or versions[-1]["LastModified"] >= self.start_date)
        ):
            versions += self.list_versions(
                path,
                KeyMarker=response["NextKeyMarker"],
                VersionIdMarker=response["NextVersionIdMarker"],
            )
        if self.start_date is not None:
            versions = list(filter(lambda v: v["LastModified"] >= self.start_date, versions))
        if self.end_date is not None:
            versions = list(filter(lambda v: v["LastModified"] <= self.end_date, versions))
        return versions

    def wait_for_versions(self, q):
        while not q.empty():
            version, data, future = q.get()

            try:
                future.result()
                yield version, data
            except Exception as e:
                LOG.error(f"Error downloading {version['VersionId']}: {e}")

            q.task_done()

    def make_request(self, path, *, version=None, **kwargs):
        subscribers = []
        if version is not None:
            # Because we know the size of the version already, we can supply that
            # to the download manager to save a HEAD request.
            subscribers = [ProvideSizeSubscriber(version["Size"])]
            kwargs.setdefault("VersionId", version["VersionId"])

        data = io.BytesIO()
        future = self.manager.download(self.bucket_name, path, data, extra_args=kwargs, subscribers=subscribers)

        return version, data, future

    def get(self, path, sample=2):
        LOG.info("Fetching versions from %s/%s", self.bucket_name, path)
        versions = self.list_versions(path)
        if len(versions) == 0:
            LOG.info(f"No versions found for {path}")
            return None

        # Instead of asking for the results of downloads synchronously, we're
        # queuing the futures and then waiting for them to complete.
        q = queue.Queue()
        for version in versions[::sample]:
            q.put(self.make_request(path, version=version), block=False)

        csvs = []
        for version, data in self.wait_for_versions(q):
            data.seek(0)
            csv_rows = pd.read_csv(data, dtype={"geographic_unit_fips": str})
            csv_rows["last_modified"] = pd.to_datetime(version["LastModified"]).astimezone(tz=tz.gettz(self.tz))
            csvs.append(csv_rows)

        df = pd.concat(csvs)
        for col in ["dem", "gop", "total"]:
            if col == "total":
                expected_col = "results_turnout"
            else:
                expected_col = f"results_{col}"
            if col in df.columns and expected_col not in df.columns:
                df[expected_col] = df[col].copy()

        LOG.info("Fetched %s versions", len(versions))

        return df


class ProvideSizeSubscriber(BaseSubscriber):
    """
    A subscriber which provides the transfer size before it's queued.
    """

    def __init__(self, size):
        self.size = size

    def on_queued(self, future, **kwargs):
        future.meta.provide_transfer_size(self.size)
