# You can see the html documentation inside docs/html open the index.html file, in the future it will be in a webpage or google drive.

# Why was created TensorDB
TensorDB born from the necessity of completely read big time-series matrices to make historical analysis, previous to this project there were a lot of attempts to read the data using databases like Influx, Postgresql, Timescale, Cassandra, or Mongo none of those DBs gave good read times and the memory consumption was really big due to the normalized formats and the required transformations (except for mongo). For solving the aforementioned problem the use of Zarr files was thought as the best solution and combined with Xarray provided a really good, simple, and faster solution but then other problems arrived, organize and treat more than 200 files becomes really problematic, basically, every file needed a daily append of data with a different treat and much of the characteristics that a database provides like triggers were lost, so to solve the problem this package was created as a way to organize and standardize the definition of a tensor.

# Why use TensorDB
1. Tensors' definitions are highly personalizable and simple, so they provide a good way to organize and treat your datasets.
2. It uses Xarray to read the tensors, so you have the same options that Xarray provides and It's a really well-supported library.
3. Fast reads and writes due to the use of Zarr (more formats in the future).
4. Simple, smart, and efficient backup system that avoids updates of not modified data.
5. You can create new tensors using string formulas.
6. Simple syntax, easy to learn.
7. You can store or read directly from the backup without download the data.

# Examples
```py
import tensordb
import fsspec
import xarray


tensor_client = tensordb.TensorClient(
    local_base_map=fsspec.get_mapper('test_db'),
    backup_base_map=fsspec.get_mapper('test_db' + '/backup'),
)

# Adding a default tensor definition
tensor_client.add_definition(
    definition_id='dummy_tensor_definition', 
    new_data={}
)

# to create a tensor you need to specifiy the path 
# and the tensor definition that it must use
tensor_client.create_tensor(
    path='dummy_tensor', 
    definition='dummy_tensor_definition'
)

# dummy data for the example
dummy_tensor = xarray.DataArray(
    0.,
    coords={'index': list(range(3)), 'columns': list(range(3))},
    dims=['index', 'columns']
)


# Storing the dummy tensor
tensor_client.store(path='dummy_tensor', new_data=dummy_tensor)

# Reading the dummy tensor
tensor_client.read(path='dummy_tensor')



# Creating a new tensor definition using a formula,
# you have the same Xarray methods but the tensor path need to be wrapped by ``
tensor_client.add_definition(
    definition_id='dummy_tensor_formula',
    new_data={
       'store': {
            'data_transformation': ['read_from_formula'],
        },
        'read_from_formula': {
            'formula': '`dummy_tensor` + 1'
        }
    }
)

# to create a tensor you need to specifiy the path and the tensor definition that it must use
tensor_client.create_tensor(
    path='dummy_tensor_formula',
    definition='dummy_tensor_formula'
)

# storing the new dummy tensor
tensor_client.store(path='dummy_tensor_formula')

# reading the new dummy tensor
tensor_client.read('dummy_tensor_formula')


# Appending a new row and a new columns to a dummy tensor
new_data = xarray.DataArray(
    2.,
    coords={'index': [3], 'columns': list(range(4))},
    dims=['index', 'columns']
)

tensor_client.append('dummy_tensor_formula', new_data=new_data)
tensor_client.read('dummy_tensor_formula')
```

# When is a good idea to use TensorDB
1. When you need to organize multiple tensors and personalize every one of them to have different behaviors.
2. When your data can be modeled as a set of tensors of homogenous dtype (every tensor can have its own dtype).
3. When you need to make complex calculations (rollings, dot, etc.).
4. When you need to make fast reads.
5. When you need to read the data in different ways, not only read by columns or rows.
6. When you need to have a verification of the integrity of your data using a checksum.
7. When you don't need to delete parts of the data frequently (the deletion requires the overwrite of the entire tensor).
8. When you don't need to insert data in middle positions frequently (overwrite problem).

# Recomendations
1. Inherited the TensorClient class to add customized methods.
2. Use a database like Postgresql to complement TensorDB, for example, you can keep track of autoincrement indexes that are going to be used to generate file names.

