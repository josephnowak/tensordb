import importlib.util
import warnings

if importlib.util.find_spec("pytest") is not None:
    from tensordb.zarr_v3.testing.store import StoreTests
else:
    warnings.warn("pytest not installed, skipping test suite", stacklevel=2)

__all__ = ["StoreTests"]
