import dask.threaded
import numpy as np
import obstore
import pytest
import xarray as xr

from tensordb import TensorClient
from tensordb.tensor_definition import TensorDefinition
from tensordb.utils.ic_storage_model import LocalStorageModel


def create_dummy_array(n_rows, n_cols, coords=None, dtype=None) -> xr.DataArray:
    coords = coords
    if coords is None:
        dtype = dtype
        if dtype is None:
            dtype = "<U15"
        coords = {
            "index": np.sort(np.array(list(map(str, range(n_rows))), dtype=dtype)),
            "columns": np.sort(np.array(list(map(str, range(n_cols))), dtype=dtype)),
        }

    return xr.DataArray(
        np.random.rand(n_rows, n_cols), dims=["index", "columns"], coords=coords
    )


class TestTensorClient:
    @pytest.fixture(autouse=True)
    def test_setup_tests(self, tmpdir):
        path = tmpdir.strpath.replace("\\", "/")
        self.tensor_client = TensorClient(
            ob_store=obstore.store.LocalStore(path + "/ob_store", mkdir=True),
            ic_storage=LocalStorageModel(
                path=path + "/ic_storage",
            ),
            # synchronizer='thread',
        )
        self.arr = xr.DataArray(
            data=np.array(
                [
                    [1, 2, 7, 4, np.nan],
                    [np.nan, 3, 5, np.nan, 6],
                    [3, 3, np.nan, np.nan, 6],
                    [np.nan, 3, 10, np.nan, 6],
                    [np.nan, 7, 8, 5, 6],
                ],
                dtype=float,
            ),
            dims=["index", "columns"],
            coords={"index": [0, 1, 2, 3, 4], "columns": [0, 1, 2, 3, 4]},
        )
        self.arr2 = xr.DataArray(
            data=np.array(
                [
                    [1, 2, 7, 4, 5],
                    [2, 6, 5, 5, 6],
                    [3, 3, 11, 5, 6],
                    [4, 3, 10, 5, 6],
                    [5, 7, 8, 5, 6],
                ],
                dtype=float,
            ),
            dims=["index", "columns"],
            coords={"index": [0, 1, 2, 3, 4], "columns": [0, 1, 2, 3, 4]},
        )
        self.arr3 = xr.DataArray(
            data=np.array(
                [
                    [1, 2, 7, 4, 5],
                    [2, 6, 5, 5, 6],
                    [3, 3, 11, 5, 6],
                    [4, 3, 10, 5, 6],
                    [5, 7, 8, 5, 6],
                    [5, 7, 8, 5, 6],
                ],
                dtype=float,
            ),
            dims=["index", "columns"],
            coords={"index": [0, 1, 2, 3, 4, 5], "columns": [0, 1, 2, 3, 4]},
        )

        # store the data, so all the tests of append or update avoid running the test_store multiple times
        for path, data in [
            ("data_one", self.arr),
            ("data_two", self.arr2),
            ("data_three", self.arr3),
        ]:
            definition = TensorDefinition(
                path=path, definition={}, dag={"omit_on": ["store", "append", "update"]}
            )
            self.tensor_client.create_tensor(definition=definition)
            self.tensor_client.store(new_data=data, path=path)

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_create_tensor(self, as_dict):
        path = "create_tensor1"
        definition = TensorDefinition(path=path)
        d = definition.model_dump() if as_dict else definition
        self.tensor_client.create_tensor(d)
        assert definition == self.tensor_client.get_tensor_definition(path)

        definition = TensorDefinition(path=path, definition={"a": {"c": 1}})
        d = definition.model_dump() if as_dict else definition
        self.tensor_client.create_tensor(d)
        assert definition == self.tensor_client.get_tensor_definition(path)

    def test_store(self):
        for path, data in [
            ("data_one", self.arr),
            ("data_two", self.arr2),
            ("data_three", self.arr3),
        ]:
            assert self.tensor_client.read(path=path).equals(data)

    @pytest.mark.parametrize("use_local", [True, False])
    def test_update(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        tensor_client.update(new_data=self.arr2, path="data_one")
        assert tensor_client.read(path="data_one").equals(self.arr2)

        assert (
            tensor_client.update(
                new_data=self.arr2.reindex(index=[100]), path="data_one"
            )
            is None
        )

    @pytest.mark.parametrize("use_local", [True, False])
    def test_upsert(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        tensor_client.create_tensor({"path": "data_upsert"})
        # Test creating from strach the tensor, this should call the store method
        tensor_client.upsert(new_data=self.arr2, path="data_upsert")
        assert tensor_client.read(path="data_upsert").equals(self.arr2)

        # This method should call the append and the update method
        tensor_client.upsert(new_data=self.arr3, path="data_upsert")
        assert tensor_client.read(path="data_upsert").equals(self.arr3)

    @pytest.mark.parametrize("use_local", [True, False])
    def test_drop(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        coords = {"index": [0, 3], "columns": [1, 2]}
        tensor_client.drop(path="data_one", coords=coords)
        tensor_client.read("data_one").equals(self.arr.drop_sel(coords))

    @pytest.mark.parametrize("use_local", [True, False])
    def test_append(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        arr = create_dummy_array(10, 5, dtype=int)
        arr = arr.sel(
            index=(
                ~arr.coords["index"].isin(
                    tensor_client.read(path="data_one").coords["index"]
                )
            )
        )

        for i in range(arr.sizes["index"]):
            tensor_client.append(new_data=arr.isel(index=[i]), path="data_one")

        assert tensor_client.read(path="data_one").sel(arr.coords).equals(arr)
        assert tensor_client.read(path="data_one").sizes["index"] > arr.sizes["index"]

    @pytest.mark.parametrize("use_local", [True, False])
    def test_delete_tensor(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        tensor_client.delete_tensor("data_one")
        assert not tensor_client.exist("data_one")

    @pytest.mark.parametrize("use_local", [True, False])
    def test_read_from_formula(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        definition = TensorDefinition(
            path="data_four",
            definition={
                "read": {
                    "substitute_method": "read_from_formula",
                },
                "read_from_formula": {
                    "formula": "new_data = (`data_one` * `data_two`).rolling({'index': 3}).sum()",
                    "use_exec": True,
                },
            },
        )
        tensor_client.create_tensor(definition=definition)

        data_four = tensor_client.read(path="data_four")
        data_one = tensor_client.read(path="data_one")
        data_two = tensor_client.read(path="data_two")
        assert data_four.equals((data_one * data_two).rolling({"index": 3}).sum())

        definition = TensorDefinition(
            path="data_four",
            definition={
                "read": {
                    "substitute_method": "read_from_formula",
                },
                "read_from_formula": {
                    "formula": "`data_four` * `data_two`",
                },
            },
        )
        tensor_client.create_tensor(definition=definition)
        tensor_client.store("data_four", new_data=self.arr)
        assert tensor_client.read("data_four").equals(self.arr * self.arr2)

    @pytest.mark.parametrize("use_local", [True, False])
    def test_read_data_transformation(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        definition = TensorDefinition(
            path="data_four",
            definition={
                "read": {
                    "data_transformation": [
                        {
                            "method_name": "read_from_formula",
                            "parameters": {"formula": "`data_one` * `data_two`"},
                        }
                    ],
                },
            },
        )
        tensor_client.create_tensor(definition=definition)
        assert tensor_client.read("data_four").equals(self.arr * self.arr2)

    @pytest.mark.parametrize("use_local", [True, False])
    def test_data_transformation_parameters_priority(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        definition = TensorDefinition(
            path="data_four",
            definition={
                "read": {
                    "data_transformation": [
                        {
                            "method_name": "read_from_formula",
                            "parameters": {"formula": "`data_one` * `data_two`"},
                        }
                    ],
                },
            },
        )
        tensor_client.create_tensor(definition=definition)
        assert tensor_client.read("data_four", formula="`data_three`").equals(self.arr3)

        definition = TensorDefinition(
            path="data_five",
            definition={
                "read": {
                    "data_transformation": [
                        {
                            "method_name": "read_from_formula",
                            "parameters": {"formula": "`data_three`"},
                        }
                    ],
                },
                "read_from_formula": {"formula": "`data_one` * `data_two`"},
            },
        )
        tensor_client.create_tensor(definition=definition)
        assert tensor_client.read("data_five").equals(self.arr3)

    @pytest.mark.parametrize("use_local", [True, False])
    def test_specifics_definition(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        definition = TensorDefinition(
            path="specific_definition",
            definition={
                "store": {
                    "data_transformation": [
                        {
                            "method_name": "read_from_formula",
                            "parameters": {"formula": "`data_one`"},
                        },
                        {
                            "method_name": "read_from_formula",
                            "parameters": {"formula": "new_data * `data_one`"},
                        },
                    ],
                },
            },
        )
        tensor_client.create_tensor(definition=definition)
        tensor_client.store("specific_definition")
        assert tensor_client.read("specific_definition").equals(
            self.tensor_client.read("data_one") ** 2
        )

    @pytest.mark.parametrize("use_local", [True, False])
    def test_different_client(self, use_local):
        tensor_client = self.tensor_client if use_local else self.tensor_client
        definition = TensorDefinition(
            path="different_client",
            storage={"storage_name": "json_storage"},
            definition={},
        )
        tensor_client.create_tensor(definition=definition)
        tensor_client.store(path="different_client", new_data={"a": 100})
        assert {"a": 100} == tensor_client.read(
            path="different_client", name="different_client"
        )

    @pytest.mark.parametrize("max_parallelization", [1, 2, 4])
    def test_exec_on_dag_order(self, max_parallelization):
        definitions = [
            TensorDefinition(
                path="0",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`data_one`"},
                            }
                        ]
                    }
                },
                dag={"depends": []},
            ),
            TensorDefinition(
                path="1",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`0` * 2"},
                            }
                        ]
                    }
                },
                dag={"depends": ["0"]},
            ),
            TensorDefinition(
                path="2",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`0` + 1"},
                            }
                        ]
                    }
                },
                dag={"depends": ["0"]},
            ),
            TensorDefinition(
                path="3",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`1` + `2`"},
                            }
                        ]
                    }
                },
                dag={"depends": ["2", "1"]},
            ),
        ]
        for definition in definitions:
            self.tensor_client.create_tensor(definition)

        self.tensor_client.exec_on_dag_order(
            method=self.tensor_client.store,
            parallelization_kwargs={"max_parallelization": max_parallelization},
        )
        assert self.tensor_client.read("0").equals(self.arr)
        assert self.tensor_client.read("1").equals(self.arr * 2)
        assert self.tensor_client.read("2").equals(self.arr + 1)
        assert self.tensor_client.read("3").equals(self.arr + 1 + self.arr * 2)

        self.tensor_client.exec_on_dag_order(
            method=self.tensor_client.store,
            tensors_path=["1"],
            autofill_dependencies=True,
            parallelization_kwargs={"max_parallelization": max_parallelization},
        )
        assert self.tensor_client.read("1").equals(self.arr * 2)

    @pytest.mark.parametrize(
        "max_per_group, client_type",
        [
            ({"first": 1, "second": 2}, "thread"),
            ({"first": 2, "second": 1}, "dask"),
            ({"first": 2, "second": 1}, "process"),
        ],
    )
    def test_get_dag_for_dask(self, max_per_group, client_type, dask_client):
        # TODO: Improve this tests, it only generates a DAG, so it does not need to check if the results
        #   of the computations are correct.
        definitions = [
            TensorDefinition(
                path="0",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`data_one`"},
                            }
                        ]
                    }
                },
                dag={"depends": [], "group": "first"},
            ),
            TensorDefinition(
                path="1",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`0` * 2"},
                            }
                        ]
                    }
                },
                dag={"depends": ["0"], "group": "first"},
            ),
            TensorDefinition(
                path="2",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`0` + 1"},
                            }
                        ]
                    }
                },
                dag={"depends": ["0"], "group": "second"},
            ),
            TensorDefinition(
                path="4",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`0` + 1"},
                            }
                        ]
                    }
                },
                dag={"depends": ["0"], "group": "first"},
            ),
            TensorDefinition(
                path="3",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`1` + `2`"},
                            }
                        ]
                    }
                },
                dag={"depends": ["2", "1"], "group": "second"},
            ),
            TensorDefinition(
                path="5",
                definition={
                    "store": {
                        "data_transformation": [
                            {
                                "method_name": "read_from_formula",
                                "parameters": {"formula": "`99` + `2`"},
                            }
                        ]
                    }
                },
                dag={"depends": ["3"], "omit_on": ["store"]},
            ),
        ]
        for definition in definitions:
            self.tensor_client.create_tensor(definition)

        dask_graph = self.tensor_client.get_dag_for_dask(
            method=self.tensor_client.store,
            max_parallelization_per_group=max_per_group,
            final_task_name="FinalTask",
        )
        if client_type == "dask":
            get = dask_client.get
        elif client_type == "thread":
            get = dask.threaded.get
        else:
            get = dask.multiprocessing.get

        get(dask_graph, "FinalTask")

        assert self.tensor_client.read("0").equals(self.arr)
        assert self.tensor_client.read("1").equals(self.arr * 2)
        assert self.tensor_client.read("2").equals(self.arr + 1)
        assert self.tensor_client.read("3").equals(self.arr + 1 + self.arr * 2)
        assert not self.tensor_client.exist("5", only_definition=False)

        dask_graph = self.tensor_client.get_dag_for_dask(
            method=self.tensor_client.store,
            tensors=[self.tensor_client.get_tensor_definition("1")],
            max_parallelization_per_group=max_per_group,
            final_task_name="FinalTask",
        )
        get(dask_graph, "FinalTask")
        assert self.tensor_client.read("1").equals(self.arr * 2)


if __name__ == "__main__":
    test = TestTensorClient()
    test.test_setup_tests()
    # test.test_add_definition()
    # test.test_store()
    # .test_update()
    # test.test_append()
    # test.test_backup()
    # test.test_read_from_formula()
    # test.test_ffill()
    # test.test_replace_by_last_valid()
    # test.test_last_valid_dim()
    # test.test_reindex()
    # test.test_overwrite_append_data()
    # test.test_specifics_definition()
    # test.test_different_client()
