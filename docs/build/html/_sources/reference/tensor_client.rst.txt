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
   TensorClient.exist
   TensorClient.get_cached_storage
   TensorClient.apply_data_transformation
   TensorClient.get_storage


Tensor Calculation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    TensorClient.read_from_formula
    TensorClient.ffill
    TensorClient.rank
