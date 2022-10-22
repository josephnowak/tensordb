from typing import List

import numpy as np
import pytest
import xarray as xr

from tensordb.tensor_translator import defined_translation


# TODO: Add more tests for the dataset cases


class TestTensorTranslation:

    @pytest.fixture(autouse=True)
    def setup_tests(self):
        self.data_array = xr.DataArray(
            np.arange(56).reshape((7, 8)).astype(np.float64),
            dims=['a', 'b'],
            coords={'a': list(range(7)), 'b': list(range(8))}
        )
        self.dataset = xr.Dataset(
            {'first': self.data_array, 'second': self.data_array + 10}
        )

    @staticmethod
    def read_by_coords(data: xr.DataArray, coords) -> xr.DataArray:
        return data.sel(**coords)

    @staticmethod
    def read_by_coords_dataset(dataset: xr.Dataset, coords) -> List[xr.DataArray]:
        return dataset[['first', 'second']].sel(**coords)
        # return [dataset[name] for name in ['first', 'second']]

    def test_defined_translation_data_array(self):
        data = defined_translation(
            self.read_by_coords,
            dims=['a', 'b'],
            coords={'a': np.array(list(range(6))), 'b': list(range(8))},
            chunks=[2, 3],
            dtypes=np.float64,
            func_parameters={'data': self.data_array}
        )
        assert data.equals(self.data_array.sel(**data.coords))

    def test_defined_translation_dataset(self):
        data = defined_translation(
            self.read_by_coords_dataset,
            dims=['a', 'b'],
            coords={'a': list(range(6)), 'b': list(range(8))},
            chunks=[2, 3],
            dtypes=[np.float64, np.float64],
            data_names=['first', 'second'],
            func_parameters={'dataset': self.dataset},
        )
        assert data.equals(self.dataset.sel(**data.coords))


if __name__ == "__main__":
    test = TestTensorTranslation()
    # test.test_store_data()
    # test.test_append_data(remote=False)
    # test.test_update_data()
    # test.test_backup()
