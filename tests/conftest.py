from collections.abc import Generator
from typing import Any

import pytest
from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def dask_cluster() -> Generator[LocalCluster, Any]:
    with LocalCluster(
        processes=False,
    ) as cluster:
        yield cluster


@pytest.fixture(scope="session")
def dask_client(dask_cluster) -> Generator[LocalCluster, Any]:
    with Client(dask_cluster) as client:
        yield client
