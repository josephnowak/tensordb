# Why was created TensorDB
TensorDB born from the necessity of completly read big timeseries matrices to make historical analisys, previous to this project there were a lot of attempt to read the data using databases like Influx, Postgresql, Timescale, Cassandra or Mongo none of those DBs gave good read times and the memory consumption was really big due to the normalized formats (except for mongo) and the necessity of load as much as possible data to reduce the query times. For solving the aforementioned problem the use of Parquet files was thinked as a good solution but It has a great number of limitations in term of the concatenation of new data, so to solve it Xarray and Zarr become the best solution.

# Why use TensorDB
1. Tensors' definitions are highly personalizable and simple, so they provide a good way to define your datasets.
2. It use Xarray to read the tensors, so you have the same options that Xarray provide and It's a really well-supported library.
3. Fast reads and writes due to the use of Zarr (more formats in the future).
4. Simple, smart and efficient backup system that avoid update not modified data (supported by fsspec).
5. You can create new tensors using formulas.

# When is good idea use TensorDB
1. When you need to organize multiple tensors and personalize every one of them to have different behaviours.
2. When your data can be modeled as a set of tensors of homogenous dtype (every tensor can has his own dtype).
3. When you need to make complex calculations (rollings, dot, etc.).
4. When you need to make fast reads.
5. When you need to read the data in different ways not only reading by columns or by rows.
6. When you need to make fast modifications of your data on disk.
7. When you need to have a verification of the integrity of your data using a checksum.
8. When you don't need to delete parts of the data frequently (the deletion require the overwrite of the entiere tensor).
9. When you don't need to insert data in middle position frequently (overwrite problem).


