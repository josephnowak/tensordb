import os.path
from abc import ABC
from typing import Type


class BaseLock(ABC):
    def __init__(self, path: str, *args, **kwargs):
        raise NotImplementedError

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


class PrefixLock:
    def __init__(self, prefix: str, lock: Type[BaseLock]):
        self.prefix = prefix
        self.lock = lock

    def get_lock(self, path):
        path = os.path.join(self.prefix, path.replace('/', '-') + '.lock')
        return self.lock(path)
