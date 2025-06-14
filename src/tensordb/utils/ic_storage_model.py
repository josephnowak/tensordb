import abc

import icechunk as ic
from pydantic import BaseModel


class BaseStorageModel(BaseModel, abc.ABC):
    model_name: str

    @abc.abstractmethod
    def get_storage(self, path) -> ic.Storage:
        pass


class LocalStorageModel(BaseModel):
    path: str

    def get_storage(self, path) -> ic.Storage:
        return ic.local_filesystem_storage(path=self.path + "/" + path)


class S3StorageModel(BaseModel):
    bucket: str
    prefix: str
    region: str
    access_key_id: str | None = None
    secret_access_key: str | None = None

    def get_storage(self, path) -> ic.Storage:
        return ic.s3_storage(
            bucket=self.bucket,
            prefix=self.prefix + "/" + path,
            region=self.region,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
        )
