============
TensorClient
============
.. currentmodule:: tensordb


Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   TensorClient


Tensor Definition Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   TensorClient.add_definition
   TensorClient.get_definition
   TensorClient.create_tensor
   TensorClient.get_tensor_definition


Storage Methods
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   TensorClient.storage_method_caller
   TensorClient.read
   TensorClient.store
   TensorClient.append
   TensorClient.update
   TensorClient.upsert
   TensorClient.delete_tensor
   TensorClient.delete_definition
   TensorClient.exist
   TensorClient.get_cached_storage
   TensorClient.apply_data_transformation
   TensorClient.get_storage


Tensor Calculation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    TensorClient.read_from_formula
    TensorClient.append_reindex
    TensorClient.reindex
    TensorClient.fillna
    TensorClient.append_ffill
    TensorClient.ffill
    TensorClient.last_valid_dim
