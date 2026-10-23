WDM and FD Likelihood Computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The computation objects of the GB likelihood on WDM and FD grids. :doc:`likelihood_path`
describes how the global fit chains them. Source parameters are ``(num_bin, 9)`` physical
rows ``(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta)`` with the sky angles in the
orbits' frame.

Chunked heterodyne: ``gbgpu.gbcomps``
-------------------------------------

.. automodule:: gbgpu.gbcomps

.. autoclass:: gbgpu.gbcomps.GBWDMComputations
    :show-inheritance:

.. autoclass:: gbgpu.gbcomps.GBFDComputations
    :members:
    :show-inheritance:

Signal heterodyne: ``gbgpu.gbsignalhetcomputations``
----------------------------------------------------

.. automodule:: gbgpu.gbsignalhetcomputations

.. autoclass:: gbgpu.gbsignalhetcomputations.GBSignalHetComputations
    :members:
    :show-inheritance:
    :exclude-members: hessian_wdm

Direct-to-WDM lookup: ``gbgpu.gblookupcomputations``
----------------------------------------------------

.. automodule:: gbgpu.gblookupcomputations

.. autoclass:: gbgpu.gblookupcomputations.GBLookupComputations
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.gblookupcomputations.GBLookupWDMComputations
    :members:
    :show-inheritance:

Band likelihood engines: ``gbgpu.gb_likelihood``
------------------------------------------------

.. automodule:: gbgpu.gb_likelihood

.. autofunction:: gbgpu.gb_likelihood.make_band_likelihood_engine

.. autoclass:: gbgpu.gb_likelihood.WDMBandLikelihoodEngine
    :members:
    :show-inheritance:
    :exclude-members: hessian

.. autoclass:: gbgpu.gb_likelihood.FDBandLikelihoodEngine
    :members:
    :show-inheritance:

.. autoclass:: gbgpu.gb_likelihood.BandLikelihoodEngine
    :members:

.. autoclass:: gbgpu.gb_likelihood.TwoQuadraturePhaseMaxMixin

.. autoclass:: gbgpu.gb_likelihood.SwapLLResult
