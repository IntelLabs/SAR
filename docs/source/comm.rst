.. _comm-guide:
.. currentmodule:: sar


SAR's communication routines
=============================
SAR uses only two types of collective communication calls: ``all_to_all`` and ``all_reduce``. This choice was made to improve scalability by avoiding any point-to-point communication. Currently, the only backends in `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ that support ``all_to_all`` are ``nccl``, ``ccl``, or ``mpi``.  Nvidia's ``nccl`` is already included in the PyTorch distribution and it is the natural choice when training on GPUs.

The ``ccl`` backend uses `Intel's OneCCL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html>`_ library. You can install the PyTorch bindings for OneCCL `here <https://github.com/intel/torch-ccl>`_ .  ``ccl`` is the preferred backend when training on CPUs.

You can train on CPUs and still use the ``nccl`` backend, or you can train on GPUs and use the ``ccl`` backend. However, you will incur extra overhead to move tensors back and forth between the CPU and GPU in order to provide the right tensors to the communication backend.

In an environment with a networked file system, initializing ``torch.distributed`` is quite easy: ::

  if backend_name == 'nccl':
      comm_device = torch.device('cuda')
  else:
      comm_device = torch.device('cpu')
  
  master_ip_address = sar.nfs_ip_init(rank,path_to_ip_file)
  sar.initialize_comms(rank,world_size, master_ip_address,backend_name,comm_device)

..

:func:`sar.nfs_ip_init` communicates the master's ip address to the workers through the file system. In the absence of a networked file system, you should develop your own mechanism to communicate the master's ip address.

You can specify the name of the socket that will be used for communication with `SAR_SOCKET_NAME` environment variable (if not specified, the first available socket will be selected).

      
  
Relevant methods
---------------------------------------------------------------------------

.. autosummary::
   :toctree: comm package

   initialize_comms
   rank
   world_size
   sync_params
   gather_grads
   nfs_ip_init
