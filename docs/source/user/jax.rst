JAX Implementations
~~~~~~~~~~~~~~~~~~~

.. automodule:: gbgpu.jax

The modules below import ``jax`` and ``lisatools.jax`` at module level, so they need JAX
installed. They are validation and autograd tools: GBGPU registers no ``gbgpu_jax``
backend, and the global fit runs the C++ / CUDA kernels. ``gbgpu.jax.wdm`` re-exports the
three direct-WDM kernels, the chunked-heterodyne fill / ll / swap kernels (not their grad /
hessian variants), ``fast_wdm_inner_heterodyne_jax``, ``gb_chunk_fd_to_wdm_jax`` and
``ALPHA_AUTO``.

Not documented here: ``gbgpu.jax.wrappers`` and ``gbgpu.jax.wdm.computation_group``
(C++-API adapter classes that no registered backend exposes), and
``gbgpu.jax.tdi_on_the_fly.sobbh_run_wave_tdi`` (it imports ``gbgpu.jax.sources.sobbh``,
which does not exist; the JAX SOBBH source is ``bbhx.jax.sources.sobbh``).

UCB source: ``gbgpu.jax.sources.ucb``
-------------------------------------

.. automodule:: gbgpu.jax.sources.ucb

.. autoclass:: gbgpu.jax.sources.ucb.JaxUCBSource
    :members:
    :show-inheritance:

TDI on the fly: ``gbgpu.jax.tdi_on_the_fly``
--------------------------------------------

.. automodule:: gbgpu.jax.tdi_on_the_fly

.. autofunction:: gbgpu.jax.tdi_on_the_fly.run_wave_tdi

.. autofunction:: gbgpu.jax.tdi_on_the_fly.gb_run_wave_tdi

Direct WDM kernels: ``gbgpu.jax.wdm.kernels``
---------------------------------------------

.. automodule:: gbgpu.jax.wdm.kernels

.. autofunction:: gbgpu.jax.wdm.kernels.gb_wdm_get_ll_jax

.. autofunction:: gbgpu.jax.wdm.kernels.gb_wdm_fill_global_jax

.. autofunction:: gbgpu.jax.wdm.kernels.gb_wdm_swap_ll_jax

Chunked-heterodyne kernels: ``gbgpu.jax.wdm.heterodyne_kernels``
----------------------------------------------------------------

.. automodule:: gbgpu.jax.wdm.heterodyne_kernels
    :members:

Chunked-heterodyne primitives: ``gbgpu.jax.wdm.fast_inner_heterodyne``
----------------------------------------------------------------------

.. automodule:: gbgpu.jax.wdm.fast_inner_heterodyne
    :members:
    :undoc-members:

Signal-heterodyne kernels: ``gbgpu.jax.wdm.signal_het_kernels``
---------------------------------------------------------------

.. automodule:: gbgpu.jax.wdm.signal_het_kernels

.. autofunction:: gbgpu.jax.wdm.signal_het_kernels.gb_signal_het_get_ll_sparse_jax

.. autofunction:: gbgpu.jax.wdm.signal_het_kernels.gb_signal_het_fd_centered_jax

.. autofunction:: gbgpu.jax.wdm.signal_het_kernels.gb_signal_het_get_ll_in_kernel_jax
