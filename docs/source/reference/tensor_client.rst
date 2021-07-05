
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

   TensorClient.add_tensor_definition
   TensorClient.create_tensor
   TensorClient.get_tensor_definition
   TensorClient.get_storage_tensor_definition
   TensorClient.exist_tensor_definition


Storage Methods
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   TensorClient.read
   TensorClient.store
   TensorClient.append
   TensorClient.update
   TensorClient.upsert
   TensorClient.backup
   TensorClient.update_from_backup
   TensorClient.set_attrs
   TensorClient.get_attrs
   TensorClient.close
   TensorClient.delete_file
   TensorClient.exist
   TensorClient.get_cached_tensor_manager


Tensor Calculation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   TensorClient.read_from_formula
