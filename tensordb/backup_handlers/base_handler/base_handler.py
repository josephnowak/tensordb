from abc import abstractmethod
from typing import Dict, List, Any, Union, Callable, Set


class BaseHandler:
    @abstractmethod
    def download_files(self, **kwargs):
        pass

    @abstractmethod
    def upload_files(self, **kwargs):
        pass











