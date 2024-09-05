import os

from dotenv import load_dotenv
from pydantic_settings import SettingsConfigDict, BaseSettings

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    PROJECT_NAME: str = "TensorDB"
    PROJECT_DESCRIPTION: str = (
        "TensorDB is an storage engine that use Zarr, Xarray and fsspec to store the data, "
        "it provides ACID transactions using a filesystem, which make it extremely easy to use"
    )

    # Paths
    BRANCH_FS_FOLDER: str = os.getenv("BRANCH_FS_FOLDER", "branchfs")
    TENSOR_DEFINITION_PATH: str = os.getenv(
        "TENSOR_DEFINITION_PATH", "tensordb/tensor_definition"
    )
    TRANSACTION_PATH: str = os.getenv("TRANSACTION_PATH", "tensordb/transaction")
    TRANSACTION_METADATA_PATH: str = os.getenv(
        "TRANSACTION_METADATA_PATH", "tensordb/metadata"
    )
    TRANSACTION_DATA_PATH: str = os.getenv("TRANSACTION_DATA_PATH", "tensordb/data")
    TRANSACTION_STATUS_PATH: str = os.getenv(
        "TRANSACTION_STATUS_PATH", "tensordb/status"
    )
    BRANCH_FOLDER: str = os.getenv("BRANCH_FOLDER", "branch")
    REPOSITORY_PATH: str = os.getenv("REPOSITORY_PATH", "tensordb/repository")


settings = Settings()
