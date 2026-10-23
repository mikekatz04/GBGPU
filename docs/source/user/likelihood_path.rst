GB likelihoods in the global fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GBGPU holds two galactic-binary (GB) stacks:

* :class:`gbgpu.gbgpu.GBGPU` (:doc:`main`, :doc:`derivedwaves`): the FastGB
  frequency-domain waveform (:meth:`~gbgpu.gbgpu.GBGPUBase.run_wave`) and the legacy FD
  likelihood, F-statistic, swap and global-template kernels. Its response is an equal-arm
  approximation (arm vectors normalized by the constant ``orbits.armlength``, one transfer
  frequency).
* The WDM / FD computation objects (:doc:`computations`), built on the ``lisatools``
  orbits, TDI configuration and domain settings. Their response follows the orbits object,
  so with the global fit's ``lisatools.detector.L1Orbits`` the arms are unequal and vary in
  time.

The GB likelihood of the 9-month and 12-month global fits is the second stack. This page
describes how the pieces fit together on those runs; the API pages give the full arguments.

The production chain
--------------------

1. ``lisatools.globalfit.stock.erebor.variants.gb_no_fg`` builds a
   :class:`~gbgpu.gbcomps.GBWDMComputations` (chunked heterodyne, the exact WDM likelihood)
   on the run's WDM grid and orbits, as ``gb_info.gb_wdm_comp``. ``GB_LIKELIHOOD=lookup``
   would select :class:`~gbgpu.gblookupcomputations.GBLookupWDMComputations` instead; the
   launchers keep the default ``chunked``.
2. ``lisatools.globalfit.stock.erebor.gb.build_sighet_engine`` wraps that comp, for the GB
   and the VGB branch, with
   :meth:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations.for_band_engine` of
   :class:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations`, passing a GB WDM lookup
   table and ``anchor_engine="lookup"``. The chunked comp stays the delegate.
3. Inside the sig-het object a :class:`~gbgpu.gblookupcomputations.GBLookupComputations` on
   that table builds the v5 carrier references (``SIGHET_REF_BUILD=lookup``), scores the
   anchor offsets (``SIGHET_ANCHOR_ENGINE=lookup``) and computes the proposal information
   matrices (``SIGHET_INFOMAT_ENGINE=lookup``, refreshed by LAT's ``GB_CHOL_CACHE=1``).
4. ``lisatools.globalfit.moves.gbbands.make_routed_band_engine`` calls
   :func:`~gbgpu.gb_likelihood.make_band_likelihood_engine`, which returns a
   :class:`~gbgpu.gb_likelihood.WDMBandLikelihoodEngine` around the sig-het object. The GB
   moves drive it through LAT's ``SubBandBuffer``.

Routing. Between
:meth:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations.setup_in_model` and
:meth:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations.clear_in_model` (an in-model
block), every GB and VGB in-model repeat proposal is scored by the sig-het v5 kernel against
per-source references, plus the anchor offset. Everything else (RJ births and removals,
template fills, swaps, scores outside an in-model block) goes to the chunked delegate. The
F-statistic grid fit of the RJ birth proposals scores its candidates through the sig-het
F-stat (:meth:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations.setup_fstat_references`
/ :meth:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations.get_fstat_ll_wdm`; LAT
``FSTAT_USE_SIGHET``, default on).

The :class:`~gbgpu.gbgpu.GBGPU` object that ``lisatools.globalfit.recipe`` builds per GB / VGB
branch is a container on this path (the moves read its ``gpus``, ``orbits`` and
``backend``). :meth:`~gbgpu.gbgpu.GBGPUBase.run_wave` runs only in the head rank's
warm-start match referee (CPU backend, ``GB_WARM_SELF=1``) and on the monitor rank, which
also uses :class:`~gbgpu.gbcomps.GBFDComputations` on CPU.

Production sig-het settings
---------------------------

The :meth:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations.for_band_engine`
arguments come from the LAT ``GBSettings`` / ``VGBSettings`` ``sighet_*`` fields
(``lisatools.globalfit.stock.erebor.gb.sighet_engine_kwargs``). On the 9-month and 12-month
launchers:

* ``nt_layer``: ``SIGHET_NT_LAYER`` = 180 (9 months) / 240 (12 months);
* ``n_cp_build``: ``SIGHET_N_CP`` = 256;
* ``tukey_alpha``: ``SIGHET_TUKEY_ALPHA`` = 0.01;
* field defaults: ``v5=1``, ``v4_knots=128``, ``v4_band=16``, ``v3_n_nodes=64``,
  ``n_sparse_fd=1024``, ``max_r=0``;
* ``lookup_table``: the table resolved by ``lisatools.wdm_lookup_store.resolve_gb_lookup_table``.

The environment knobs read by the sig-het module, with their defaults and launcher values,
are listed in the :mod:`gbgpu.gbsignalhetcomputations` module documentation
(:doc:`computations`).

Parameters and frames
---------------------

Every computation object takes ``(num_bin, 9)`` rows of physical parameters
``(amp, f0 [Hz], fdot [Hz/s], fddot [Hz/s^2], phi0, iota, psi, lam, beta)``, angles in rad.
The sky angles are read in the frame of the orbits' spacecraft positions: the global fit
builds its ``L1Orbits`` with ``frame="icrs"``, so ``lam`` is the right ascension and
``beta`` the declination (LAT samples ``alpha`` / ``sin_delta`` and converts).
``convert_to_ra_dec=True`` is the deprecated ecliptic -> ICRS conversion of the sky columns;
the :class:`~gbgpu.gblookupcomputations.GBLookupComputations` methods refuse it.

The FastGB templates of :class:`~gbgpu.gbgpu.GBGPU` carry ``exp(+i phi0)``, the opposite of
the LDC / JaxGB convention of the LAT analysis side, which the chunked-heterodyne kernels
follow. ``GBGPU(flip_ref_phase=True)`` negates ``phi0``: :meth:`~gbgpu.gbgpu.GBGPUBase.run_wave`
negates a copy, while :meth:`~gbgpu.gbgpu.GBGPUBase.get_ll`,
:meth:`~gbgpu.gbgpu.GBGPUBase.get_fstat_ll` (where that column is ``beta``),
:meth:`~gbgpu.gbgpu.GBGPUBase.swap_likelihood_difference`,
:meth:`~gbgpu.gbgpu.GBGPUBase.generate_global_template` and
:meth:`~gbgpu.gbgpu.GBGPUBase.information_matrix` negate column 4 of the caller's
``params`` array in place. No global-fit caller sets it.

Choosing the backend
--------------------

Importing ``gbgpu`` registers the backends ``gbgpu_cpu``, ``gbgpu_cuda11x``,
``gbgpu_cuda12x`` and ``gbgpu_cuda13x`` (:doc:`backends`). Pass ``force_backend``
explicitly. A flavor (``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, or ``"cuda"``
/ ``"gpu"`` for the first available CUDA flavor) is prefixed to ``gbgpu_<flavor>`` by
:class:`~gbgpu.parallelbase.GBGPUParallelModule` (for
:class:`~gbgpu.gbcomps.GBWDMComputations`, by its LAT base through
``_BACKEND_PREFIX = "gbgpu"``). The global fit passes its flavor
(``cuda13x`` on the compute ranks); the warm-start referee and the monitor pass ``"cpu"``.

* ``force_backend=None`` does not work for :class:`~gbgpu.gbgpu.GBGPU`,
  :class:`~gbgpu.thirdbody.GBGPUThirdBody` and the
  :class:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations` constructor: their
  ``supported_backends`` return bare flavors, which resolve to the first registered backend
  of that flavor (a ``gbt_*`` backend of gpubackendtools), and construction then fails
  because that backend has no GB wraps (``sharedmem`` / ``GBComputationGroupWrap``).
* The :class:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations` constructor is the
  CPU-only offline / validation path and works only with ``"cpu"`` (``"gbgpu_cpu"`` passes
  its guard, then fails in the ``lisatools`` frontends). The production
  :meth:`~gbgpu.gbsignalhetcomputations.GBSignalHetComputations.for_band_engine` takes no
  backend argument: it inherits the chunked delegate's, CPU or CUDA.
* :class:`~gbgpu.gblookupcomputations.GBLookupComputations` has no ``force_backend``; it
  uses the delegate's flavor.
* :class:`~gbgpu.gbgpu.GBGPU` and :class:`~gbgpu.thirdbody.GBGPUThirdBody` also require an
  ``orbits`` object; ``orbits=None`` raises ``ValueError``.

.. code-block:: python

    from lisatools.detector import EqualArmlengthOrbits
    from gbgpu.gbgpu import GBGPU

    gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"), force_backend="cpu")

Legacy shared-memory kernels
----------------------------

The legacy FD kernels behind :class:`~gbgpu.gbgpu.GBGPU` (``SharedMemoryGBGPU.cu``) are
compiled only with the CMake option ``GBGPU_WITH_SHAREDMEM=ON``. ``ON`` is the CMake default;
``LISAanalysistools/install.sh`` builds the development stack with ``OFF``, which also drops
the configure-time cuFFTDx / mathdx download:

.. code-block:: bash

    pip install -e . --config-settings=cmake.define.GBGPU_WITH_SHAREDMEM=OFF

With it off, :meth:`~gbgpu.gbgpu.GBGPUBase.get_ll`, :meth:`~gbgpu.gbgpu.GBGPUBase.get_fstat_ll`,
:meth:`~gbgpu.gbgpu.GBGPUBase.swap_likelihood_difference`,
:meth:`~gbgpu.gbgpu.GBGPUBase.generate_global_template` and the
``use_c_implementation=True`` paths of :meth:`~gbgpu.gbgpu.GBGPUBase.run_wave` /
:meth:`~gbgpu.gbgpu.GBGPUBase.information_matrix` raise ``RuntimeError``. The Python paths
of :meth:`~gbgpu.gbgpu.GBGPUBase.run_wave`, :meth:`~gbgpu.gbgpu.GBGPUBase.inject_signal` and
:meth:`~gbgpu.gbgpu.GBGPUBase.information_matrix`,
:meth:`~gbgpu.gbgpu.GBGPUBase.fill_global_template` and all WDM / FD computation objects
work. The WDM global fits do not call the legacy kernels; FD-basis code in LAT does (the FD
branch of ``recipe``, ``gathergalaxy``, the FD-only ``gbspecialstretch.para_log_like``,
``lisatools.sources.gb.waveform.GBXYZTDIWaveform``).
