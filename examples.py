from tensordb import TensorClient
import xarray
import fsspec

# Probably this file is never going to be used

tensor_client = TensorClient(
    local_base_map=fsspec.get_mapper('test_db'),
    backup_base_map=fsspec.get_mapper('test_db' + '/backup'),
    synchronizer='thread'
)

# Adding an empty tensor definition (there is no personalization)
tensor_client.add_tensor_definition(
    tensor_id='dummy_tensor_definition',
    new_data={
        # This key is used for modify options of the Storage constructor
        # (documented on the reserved keys section of this method)
        'handler': {
            # modify the default Storage for the zarr_storage
            'data_handler': 'zarr_storage'
        }
    }
)

# create a new empty tensor, you must always call this method to start using the tensor.
tensor_client.create_tensor(path='tensor1', tensor_definition='dummy_tensor_definition')

new_data = xarray.DataArray(
    0.0,
    coords={'index': list(range(3)), 'columns': list(range(3))},
    dims=['index', 'columns']
)

# Storing tensor1 on disk
tensor_client.store(path='tensor1', new_data=new_data)

# Reading the tensor1 (normally you will get a lazy Xarray (use dask in the backend))
tensor_client.read(path='tensor1')

# Next example

# Creating a new tensor definition using a formula that depend on the previous stored tensor
tensor_client.add_tensor_definition(
    tensor_id='tensor_formula',
    new_data={
        'store': {
            # read the docs of this method to understand the behaviour of the data_methods key
            'data_methods': ['read_from_formula'],
        },
        'read_from_formula': {
            'formula': '`tensor1` + 1 + `tensor1` * 10'
        }
    }
)

# create a new empty tensor, you must always call this method to start using the tensor.
tensor_client.create_tensor(path='tensor_formula', tensor_definition='tensor_formula')

# Storing tensor_formula on disk, check that now we do not need to send the new_data parameter, because it is generated
# from the formula that we create previously
tensor_client.store(path='tensor_formula')

# Reading the tensor_formula (normally you will get a lazy Xarray (use dask in the backend))
tensor_client.read(path='tensor_formula')


# Next Example

# Appending a new row and a new columns to the tensor_formula stored previously
new_data = xarray.DataArray(
    2.,
    coords={'index': [3], 'columns': list(range(4))},
    dims=['index', 'columns']
)

# Appending the data, you can use the compute=False parameter if you dont want to execute this immediately
tensor_client.append('tensor_formula', new_data=new_data)

# Reading the tensor_formula (normally you will get a lazy Xarray (use dask in the backend))
tensor_client.read('tensor_formula')

# Next example

# Creating a new tensor definition using an 'on the fly' formula
tensor_client.add_tensor_definition(
    tensor_id='tensor_formula_on_the_fly',
    new_data={
        'read': {
            # Read the section reserved Keywords
            'customized_method': 'read_from_formula',
        },
        'read_from_formula': {
            'formula': '`tensor1` + 1 + `tensor1` * 10'
        }
    }
)

# create a new empty tensor, you must always call this method to start using the tensor.
tensor_client.create_tensor(path='tensor_formula_on_the_fly', tensor_definition='tensor_formula_on_the_fly')

# Now we don't need to call the store method when we want to read our tensor
# the good part is that everything is still lazy
tensor_client.read(path='tensor_formula_on_the_fly')


