import os

from dotenv import load_dotenv
from pydantic_settings import SettingsConfigDict, BaseSettings

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    PROJECT_NAME: str = "TensorDB"
    PROJECT_DESCRIPTION: str = (
        "TensorDB is a serverless database that use Zarr, Xarray and fsspec, it provides ACID transactions "
        "without using an additional database, even when using Dask on a distributed environment."
    )

    # Paths
    TENSOR_DEFINITION_PATH: str = os.getenv("TENSOR_DEFINITION_PATH", "tensor_definition")
    TRANSACTION_PATH: str = os.getenv("TRANSACTION_PATH", "transaction")
    TRANSACTION_METADATA_PATH: str = os.getenv("TRANSACTION_METADATA_PATH", "metadata")
    TRANSACTION_DATA_PATH: str = os.getenv("TRANSACTION_DATA_PATH", "data")
    TRANSACTION_STATUS_PATH: str = os.getenv("TRANSACTION_STATUS_PATH", "status")
    BRANCH_PATH: str = os.getenv("BRANCH_PATH", "branch")


settings = Settings()
