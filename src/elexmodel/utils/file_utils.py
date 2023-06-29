import logging
import os
import pathlib
from io import StringIO

LOG = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV")
DATA_ENV = os.getenv("DATA_ENV")
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
TARGET_BUCKET = f"{os.getenv('MODEL_S3_BUCKET')}-{DATA_ENV}"
S3_FILE_PATH = f"{os.getenv('MODEL_S3_PATH_ROOT')}-{DATA_ENV}"


def get_directory_path():
    # we should expect config/data directories to be at the root
    directory_path = pathlib.Path().parent.absolute()
    if str(directory_path).endswith("notebooks"):
        directory_path = directory_path.absolute().parent
    return directory_path


def create_directory(path):
    LOG.info("Creating directory at %s", path)
    os.makedirs(path)


def convert_df_to_csv(df):
    csv_buf = StringIO()
    df.to_csv(csv_buf, header=True, index=False)
    return csv_buf.getvalue()
