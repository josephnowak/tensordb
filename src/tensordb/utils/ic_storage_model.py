import abc

import icechunk as ic
import obstore as ob
from pydantic import BaseModel


class BaseStorageModel(BaseModel, abc.ABC):
    model_name: str

    @abc.abstractmethod
    def get_storage(self, path) -> ic.Storage:
        pass

    @abc.abstractmethod
    def get_obstore(self, path) -> ob.store.ObjectStore:
        pass


class LocalStorageModel(BaseModel):
    path: str

    def get_storage(self, path) -> ic.Storage:
        return ic.local_filesystem_storage(path=self.path + "/" + path)

    def get_obstore(self, path) -> ob.store.ObjectStore:
        return ob.store.LocalStore(prefix=self.path + "/" + path, mkdir=True)


class S3StorageModel(BaseModel):
    bucket: str
    prefix: str
    region: str
    access_key_id: str | None = None
    secret_access_key: str | None = None
    s3_express: bool = True

    def get_storage(self, path) -> ic.Storage:
        return ic.s3_storage(
            bucket=self.bucket,
            prefix=self.prefix + "/" + path,
            region=self.region,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
        )

    def get_obstore(self, path) -> ob.store.ObjectStore:
        return ob.store.S3Store(
            bucket=self.bucket,
            prefix=self.prefix + "/" + path,
            region=self.region,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            s3_express=self.s3_express,
        )
