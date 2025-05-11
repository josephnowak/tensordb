# TensorDB: Bridging SQL Tables and Tensors (A Crazy Idea 🤯)

### **Project Objective**
🚀 **Disclaimer:** This project is far from achieving its ambitious goal, and I may never have the time to fully implement it. However, I believe in the potential of this idea and want to document it here in case future advancements make it feasible for a company or individual to take it further.

---

## **1. Bridging the Gap Between SQL Tables and Tensors (Zarr/Icechunk)**

### **Problem Context**
Many real-world applications rely on databases like InfluxDB or TimescaleDB but could significantly benefit from tensor-based data storage for specific use cases. One example is the financial industry, where handling massive time series data is crucial. However, financial instruments also come with metadata that does not integrate well with tensors.

To leverage the best of both worlds, developers often use an SQL database for metadata and a separate tensor-based storage system—introducing extra complexity. Additionally, the lack of SQL compatibility with tensor libraries (e.g., Xarray) limits access to powerful SQL-based tools.

### **Proposed Solution**
🚨 *Note:* Rather than diving into deep implementation details, this is a conceptual idea I'd love to see realized in practice.

Since no SQL database currently supports tensors natively, creating a new product would be necessary—an expensive and time-consuming challenge. Instead, an optimal approach would be to adapt an existing project to minimize development effort. The goal would be similar to TimescaleDB's structure, replacing hypertables with tensors.

All proposed implementations should rely on **Dask Distributed** for scalability, which is also essential for achieving the next objective.

#### **Implementation Ideas**
1. **Extend DuckDB** 📦  
   - Develop an extension to allow DuckDB to read tensors via Icechunk internally.
   - Convert tensors into tables dynamically for seamless querying.
   - Benefit from both SQL and Xarray syntax for maximum flexibility.
   - Unknown feasibility due to lack of experience with DuckDB extensions.

2. **Use S3 Tables or PySpark** 🌐  
   - Alternative approach to DuckDB, though extending PySpark may not be viable.

3. **Develop a Custom Engine** 🚧  
   - The least desirable option due to complexity and development effort.

---

## **2. Implementing an Eventually Consistent In-Memory System**

### **Problem Context**
Most time-series databases struggle with handling frequent small insertions efficiently. They typically require batch writes to optimize performance, which can be limiting. When using Icechunk for tensor storage, write operations become extremely slow since modifying a single value necessitates rewriting an entire chunk.

### **Proposed Solution**
Leverage **Dask** to improve write efficiency:
- Data is initially stored **in-memory**, with workers holding temporary data (the scalability in terms of memory capacity would be amazing).
- The scheduler maintains pointers to access stored data.
- Periodically, data is persisted in storage—similar to Redis’s approach.

This approach introduces additional complexity since data is maintained across two locations, but it could significantly improve performance in most of the tick data scenarios.

---

## **Current Project Status**
📌 TensorDB was originally designed before Icechunk existed. The initial goal was to introduce **ACID transactions** to Zarr while providing SQL-like functionality without requiring a centralized server.

However, since **Icechunk** now exists, this project currently lacks unique benefits. As a result, I recommend avoiding the use of TensorDB unless the aforementioned objectives are achieved.

> _Everything beyond this point was written years ago._

---

## **Why TensorDB Was Created**
TensorDB was born from the need to efficiently read large time-series matrices for historical analysis. Before this project, numerous database solutions—InfluxDB, PostgreSQL, TimescaleDB, Cassandra, MongoDB—were tested, but none provided **fast read times** and **reasonable memory consumption** due to normalization overhead (except MongoDB).

Using **Zarr files** combined with **Xarray** emerged as a **simpler, faster** alternative. However, managing 200+ files became cumbersome, especially with frequent appends and loss of essential database functionalities like triggers. TensorDB was developed to **organize** and **standardize** tensor management.

---

## **Why Use TensorDB?**
1. 🔥 Highly **customizable tensor definitions** for structured data management.
2. 🚀 **Seamless integration with Xarray**, leveraging its powerful capabilities.
3. ⚡ **Fast read & write operations** using Zarr (additional formats planned).
4. 📐 Create new tensors using **string-based formulas**.
5. 🎯 **Simple syntax**, easy to learn.
6. ⚙️ **Parallel execution** using DAG, ensuring efficient tensor operations.

---

## **Example Usage**
```python
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

# Dummy data for example
dummy_tensor = xr.DataArray(
    0.,
    coords={'index': list(range(3)), 'columns': list(range(3))},
    dims=['index', 'columns']
)

# Storing the dummy tensor
tensor_client.store(path='dummy_tensor', new_data=dummy_tensor)

# Reading the dummy tensor
tensor_client.read(path='dummy_tensor')

# Creating a tensor from a formula
tensor_client.create_tensor(
    definition=tensordb.tensor_definition.TensorDefinition(
        path='dummy_tensor_formula',
        definition={
            'store': {
                'data_transformation': [{'method_name': 'read_from_formula'}],
            },
            'read_from_formula': {
                'formula': '`dummy_tensor` + 1'
            }
        }
    )
)

# Storing the tensor from formula
tensor_client.store(path='dummy_tensor_formula')

# Reading the tensor from formula
tensor_client.read('dummy_tensor_formula')

# Appending a new row & column
new_data = xr.DataArray(
    2.,
    coords={'index': [3], 'columns': list(range(4))},
    dims=['index', 'columns']
)

tensor_client.append('dummy_tensor_formula', new_data=new_data)
tensor_client.read('dummy_tensor_formula')
```

---

## **When Should You Use TensorDB?**
✅ Managing multiple **custom tensors** with unique behaviors  
✅ When **normalized formats are too slow**  
✅ Performing **complex calculations** (rolling, dot products, etc.)  
✅ Reading data in **various ways** beyond row/column queries  
✅ When **frequent deletions or middle-position insertions** aren't required  

---

## **Recommendations**
1. 🏗️ Inherit the `TensorClient` class to add custom methods.  
2. 🗄️ Complement TensorDB with **PostgreSQL** for autoincrement index tracking.  
3. ☁️ Store tensors using **S3-compatible storage** for scalability.  

---

This README should now be **more structured**, **easier to read**, and **more engaging**! 🚀 Let me know if you'd like additional refinements! 😊
