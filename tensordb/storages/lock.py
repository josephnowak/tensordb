from abc import abstractmethod


class BaseLock(object):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, *args):
        pass


class BaseMapLock(object):
    @abstractmethod
    def __getitem__(self, item) -> BaseLock:
        pass



class NoLock(BaseLock):
    """A lock that doesn't lock."""

    def __getitem__(self, item):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


no_lock = NoLock()


def get_lock(synchronizer: BaseMapLock, path):
    if synchronizer is None:
        return no_lock
    return synchronizer[path]

