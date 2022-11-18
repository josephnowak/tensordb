from typing import Any, Generator

import pytest
from dask.distributed import LocalCluster, Client


@pytest.fixture(scope="session")
def dask_cluster() -> Generator[LocalCluster, Any, None]:
    with LocalCluster(
            processes=False,
    ) as cluster:
        yield cluster


@pytest.fixture(scope="function")
def dask_client(dask_cluster) -> Generator[LocalCluster, Any, None]:
    with Client(dask_cluster) as client:
        yield client
