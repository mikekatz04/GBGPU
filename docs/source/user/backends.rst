Backends and Dispatch
~~~~~~~~~~~~~~~~~~~~~

Package overview
----------------

.. automodule:: gbgpu

Backend selection
-----------------

Every GBGPU compute object picks its backend once, at construction: from ``force_backend``,
or, for the sig-het engine built by ``for_band_engine`` and for the lookup scorer, from the
chunked delegate (see :doc:`likelihood_path` for which objects need an explicit value). The
package-level helpers below prefix ``gbgpu_`` to a bare flavor.

.. autofunction:: gbgpu.get_backend

.. autofunction:: gbgpu.has_backend

``gbgpu.get_first_backend`` does not work as written (a string is iterated character by
character, a list fails at the prefixing); use
``gpubackendtools.get_first_backend(["gbgpu_cuda13x", "gbgpu_cpu"])`` instead.

Build helpers
-------------

.. autofunction:: gbgpu.get_include

.. autofunction:: gbgpu.get_cmake_module_path

Backend classes: ``gbgpu.cutils``
---------------------------------

.. automodule:: gbgpu.cutils

.. autoclass:: gbgpu.cutils.GBGPUBackendMethods
    :show-inheritance:

.. autoclass:: gbgpu.cutils.GBGPUBackend
    :show-inheritance:

.. autoclass:: gbgpu.cutils.GBGPUCpuBackend
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.cutils.GBGPUCuda11xBackend
    :show-inheritance:

.. autoclass:: gbgpu.cutils.GBGPUCuda12xBackend
    :show-inheritance:

.. autoclass:: gbgpu.cutils.GBGPUCuda13xBackend
    :show-inheritance:

Dispatch base: ``gbgpu.parallelbase``
-------------------------------------

.. automodule:: gbgpu.parallelbase

.. autoclass:: gbgpu.parallelbase.GBGPUParallelModule
    :show-inheritance:

Compatibility shims
-------------------

* ``gbgpu.utils.parallelbase`` re-exports :class:`gbgpu.parallelbase.GBGPUParallelModule`
  (``gbgpu.gbgpu`` imports it from there).
* ``gbgpu.wdm_het`` re-exports ``lisatools.wdm_het`` (the chunked-heterodyne geometry and
  Tukey-window helpers, moved to LAT so that BBHx can share them). New code imports from
  ``lisatools.wdm_het``.
