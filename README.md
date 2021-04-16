
TODO: Add more content

# Overview
TensorDB born from the necessity of handle and store multiple time series matrices that have some relation between them (normally some mathematical formula) and require a set of transformations every day, having that problem in mind I decided to create a DB and a DBMS that support the creation of N-Dimensional arrays (Tensors) using a set of predefined methods or formulas and the same Xarray syntax.

# Why use TensorDB
1. Tensors' definitions are highly personalizable and simple.
2. Use Xarray to read the tensors, It's a really well-supported library (The syntax is almost equal to pandas).
3. Fast reads and writes due to the use of Zarr (more formats in the future).
4. Simple and efficient backup using one of the most used cloud storage system S3 (more in the future)
5. It's simple create new tensors from a formulas.
