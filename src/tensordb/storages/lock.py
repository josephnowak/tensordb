import os.path

import dask.distributed


class BaseLock:
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError


class NoLock(BaseLock):
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class PrefixLock(BaseLock):
    def __init__(self, prefix: str, lock: type[BaseLock] = None):
        self.prefix = prefix
        self.lock = NoLock if lock is None else lock

    def __getitem__(self, path):
        path = os.path.join(self.prefix, path)
        return self.lock(path)


class DaskLock(PrefixLock):
    def __getitem__(self, path):
        path = os.path.join(self.prefix, path)
        return dask.distributed.Lock(path)
