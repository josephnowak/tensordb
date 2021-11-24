# You can see the html documentation inside docs/html (open the index.html file), 
# in the future the doc could be in a webpage or something.

# Why was created TensorDB
TensorDB born from the necessity of completely read big time-series matrices to make historical analysis, 
previous to this project there were a lot of attempts to read the data using databases like Influx, Postgresql, 
Timescale, Cassandra, or Mongo none of those DBs gave good read times and the memory consumption was 
huge due to the normalized formats and the required transformations (except for mongo). 
For solving the aforementioned problem the use of Zarr files was thought as the best solution and
combined with Xarray provided a perfect, simple, and faster solution but then other problems arrived, 
organize and treat more than 200 files becomes really problematic, basically, every file needed a
daily append of data with a different treat and much of the characteristics that a database provides like 
triggers were lost, so to solve the problem this package was created as a way to organize and standardize the 
definition of a tensor.

# Why use TensorDB
1. Tensors' definitions are highly customizable and simple, so they provide a good way to organize and treat your datasets.
2. It uses Xarray to read the tensors, so you have the same options that Xarray provides, and it's a really well-supported library.
3. Fast reads and writes due to the use of Zarr (more formats in the future).
4. You can create new tensors using string formulas.
5. Simple syntax, easy to learn.
6. Execution of tensor methods base on a DAG, this allows to write multiple tensors on parallel without worry by the 
dependencies of one with the other

# Examples
```py
import tensordb
import fsspec
import xarray as xr


tensor_client = tensordb.TensorClient(
    base_map=fsspec.get_mapper('tensordb_path'),
)

# Creating a tensor definition
tensor_client.create_tensor(
    definition=tensordb.tensor_definition.TensorDefinition(
        path='dummy_tensor',
        definition={}
    )
)

# dummy data for the example
dummy_tensor = xr.DataArray(
    0.,
    coords={'index': list(range(3)), 'columns': list(range(3))},
    dims=['index', 'columns']
)


# Storing the dummy tensor
tensor_client.store(path='dummy_tensor', new_data=dummy_tensor)

# Reading the dummy tensor
tensor_client.read(path='dummy_tensor')


# Create a tensor directly from a formula using the following definition
tensor_client.create_tensor(
    definition=tensordb.tensor_definition.TensorDefinition(
        path='dummy_tensor_formula',
        definition={
           'store': {
                'data_transformation': [
                    {'method_name': 'read_from_formula'}
                ],
            },
            'read_from_formula': {
                'formula': '`dummy_tensor` + 1'
            }
        }
    )
)

# storing the new tensor directly from the formula
tensor_client.store(path='dummy_tensor_formula')

# reading the new dummy tensor
tensor_client.read('dummy_tensor_formula')


# Appending a new row and a new columns to a dummy tensor
new_data = xr.DataArray(
    2.,
    coords={'index': [3], 'columns': list(range(4))},
    dims=['index', 'columns']
)

tensor_client.append('dummy_tensor_formula', new_data=new_data)
tensor_client.read('dummy_tensor_formula')
```

# When is a good idea to use TensorDB
1. When you need to organize multiple tensors and personalize every one of them to have different behaviors.
2. When normalized formats are too slow.
3. When you need to make complex calculations (rollings, dot, etc.).
5. When you need to read the data in different ways, not only read by columns or rows.
7. When you don't need to delete parts of the data frequently (the deletion requires overwrite the entire tensor).
8. When you don't need to insert data in middle positions frequently (overwrite problem).

# Recommendations
1. Inherited the TensorClient class to add customized methods.
2. Use a database like Postgresql to complement TensorDB, for example, you can keep track of autoincrement indexes 
that are going to be used to generate file names.
3. Use a storage like S3 to store all the tensors

