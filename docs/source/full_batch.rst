Full-batch training
==========
Distributed full-batch training in SAR may require some changes to your existing GNN model. Check :ref:`preparing your GNN for full-batch training in SAR<model-prepare>` for more details. SAR supports multiple :ref:`training modes<sar-modes>` that are suitable for different graph sizes and that trade speed for memory efficiency. For more information about SAR core distributed graph objects, check out the :ref:`distributed graph objects<shards>` section.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  Preparing your GNN for full-batch training in SAR<model_prepare>
  Training modes <sar_modes>
  Distributed Graph Objects<shards>
