"""FastGB galactic-binary (GB) waveforms, injections and legacy frequency-domain likelihoods.

:class:`GBGPU` builds the LISA TDI response of batches of circular, slowly chirping galactic
binaries with the FastGB method (Cornish & Littenberg 2007, arXiv:0704.1808, following the LDC
``ldc`` implementation): the slowly varying part of each link response is sampled on ``N``
points over the observation time ``T`` and FFT'd, so each source occupies ``N`` frequency bins
starting at ``start_inds = rint(f0 * T) - N / 2`` on the ``df = 1 / T`` grid.
:class:`GBGPUBase` holds the machinery; a derived class adds astrophysics through four hooks
(``prepare_additional_args``, ``special_get_N``, ``shift_frequency``, ``add_to_phasing``).
:class:`GBGPU` is the identity case (plain circular binaries) and
:class:`gbgpu.thirdbody.GBGPUThirdBody` the hierarchical-triple one.

Main entry points: :meth:`GBGPUBase.run_wave` (Python / ``xp`` waveforms, or the shared-memory
kernel with ``use_c_implementation=True``), :meth:`GBGPUBase.get_ll`,
:meth:`GBGPUBase.get_fstat_ll`, :meth:`GBGPUBase.swap_likelihood_difference`,
:meth:`GBGPUBase.generate_global_template`, :meth:`GBGPUBase.fill_global_template`,
:meth:`GBGPUBase.inject_signal` and :meth:`GBGPUBase.information_matrix`. The likelihood
methods, :meth:`GBGPUBase.generate_global_template` and the C paths of :meth:`GBGPUBase.run_wave`
/ :meth:`GBGPUBase.information_matrix` run the legacy FD shared-memory kernels
(``SharedMemoryGBGPU.cu``, reached through ``backend.sharedmem``), which need a GBGPU build with
``GBGPU_WITH_SHAREDMEM=ON``; with it off they raise ``RuntimeError``.
:meth:`GBGPUBase.fill_global_template` uses the gbgpu_utils ``fill_global`` kernel instead.

GB parameters (the physical basis, in this column order everywhere in this module except the
5-column intrinsic layout of :meth:`GBGPUBase.get_fstat_ll`): ``amp``
(dimensionless strain amplitude), ``f0`` [Hz], ``fdot`` [Hz/s], ``fddot`` [Hz/s^2], ``phi0``,
``iota``, ``psi`` [rad], and the sky longitude / latitude ``lam``, ``beta`` [rad]. ``f0``,
``fdot``, ``fddot`` and ``phi0`` are defined at the start ``t0`` of the observation window. The
sky angles are read in the frame of the spacecraft positions of the orbits object
(:attr:`lisatools.detector.Orbits.frame`): with the ICRS L1 orbits of the global fit, ``lam`` is
the right ascension and ``beta`` the declination. ``beta`` is converted internally to the polar
angle ``theta = pi / 2 - beta``.

The response forms the Michelson X, Y, Z combinations from the spacecraft positions with the arm
unit vectors normalized by the constant ``orbits.armlength`` and one transfer frequency
``C_SI / (2 pi armlength)`` (an equal-arm approximation of the response); ``tdi2=True``
multiplies by the second-generation factor at ``f0``. The production (unequal-arm) GB likelihood
is the WDM path in :mod:`gbgpu.gbcomps` / :mod:`gbgpu.gbsignalhetcomputations`, not this module.

Pipeline: in the 9-month / 12-month global fits (WDM domain) the :class:`GBGPU` built by
``lisatools.globalfit.recipe`` (``build_gb_moves`` / ``build_vgb_moves``) is a container only;
the GB moves read its ``gpus``, ``orbits`` and ``backend`` (``d_d`` is set only by the FD-only
F-stat MCMC / refit setups). Every GB likelihood goes
through :class:`gbgpu.gbcomps.GBWDMComputations` wrapped by
:meth:`gbgpu.gbsignalhetcomputations.GBSignalHetComputations.for_band_engine`.
:meth:`GBGPUBase.run_wave` itself runs in the head rank's warm-start match referee
(``lisatools.globalfit.warmstart.match_referee``, CPU backend, ICRS L1 orbits, ``t0`` = data
start; ``GB_WARM_SELF=1``) and on the monitor rank.

Example:
    .. code-block:: python

        import numpy as np
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.utils.constants import YRSID_SI
        from gbgpu.gbgpu import GBGPU

        gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"), force_backend="cpu")
        params = np.tile([2e-23, 2e-3, 7.5e-18, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], (10, 1))
        gb.run_wave(*params.T, T=1.0 * YRSID_SI, dt=10.0)
        gb.XYZf.shape  # (10, 3, gb.N)
"""

from multiprocessing.sharedctypes import Value
import time
import warnings
import abc

import numpy as np

# import constants
from lisatools.utils.constants import YRSID_SI
from .utils.citation import *
from .utils.parallelbase import GBGPUParallelModule

# try:
#     from lisatools.sensitivity import A1TDISens
#     tdi_available = True

# except (ModuleNotFoundError, ImportError) as e:
#     tdi_available = False
#     warnings.warn("tdi module not found. No sensitivity information will be included.")

from .utils.utility import *

from lisatools.detector import EqualArmlengthOrbits, Orbits, L1Orbits
from typing import Optional, Sequence, TypeVar, Union, Dict, Any

from gpubackendtools.parallelbase import ParallelModuleBase

# Integer codes of ``tdi_channel_setup`` / ``window`` passed to the shared-memory kernels.
tdi_channel_setup_map = {"XYZ": 1, "AET": 2, "AE": 3}
window_map = {"tukey": 1, "planck": 2} # None is mapped to 0

class GBGPUBase(GBGPUParallelModule, abc.ABC):
    """FastGB waveform, injection and legacy FD-likelihood machinery for galactic binaries.

    Generates the frequency-domain TDI response (X, Y, Z and A, E, T) of batches of GBs and,
    through the legacy shared-memory kernels, batched FD log-likelihoods, F-statistics,
    swap-likelihood differences and global templates; information matrices come from
    finite differences of :meth:`run_wave` (or, on the C path, a shared-memory kernel).
    Everything runs on the backend picked at construction (``numpy`` on ``gbgpu_cpu``, ``cupy`` on
    ``gbgpu_cuda*``). Parameter conventions and the response approximation are described in
    the module docstring (:mod:`gbgpu.gbgpu`).

    A derived class supplies four hooks that :meth:`run_wave` calls by name:
    ``prepare_additional_args(*args)`` (extra source parameters, after the 9 base ones),
    ``special_get_N(amp, f0, T, *args, oversample=...)`` (points per waveform),
    ``shift_frequency(fi, xi, *add_args)`` and
    ``add_to_phasing(arg_phasing, f0, fdot, fddot, xi, *add_args)``. :class:`GBGPU`
    implements them as identities (circular binaries);
    :class:`gbgpu.thirdbody.GBGPUThirdBody` adds a third body. :class:`GBGPUBase` itself
    defines none of them, so instantiate a subclass.

    Args:
        orbits: :class:`~lisatools.detector.Orbits` instance (e.g.
            :class:`~lisatools.detector.EqualArmlengthOrbits` or the global fit's
            :class:`~lisatools.detector.L1Orbits`) giving the spacecraft positions and
            ``armlength``. Configured on assignment (``orbits._ensure_configured()``). Despite
            the ``Optional`` annotation, ``None`` raises ``ValueError``.
        force_backend (str | Backend | tuple, optional): Backend to run on. A bare flavor
            (``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, or the ``"cuda"`` /
            ``"gpu"`` aliases for the first available CUDA flavor) is prefixed with
            ``gbgpu_``; full names (``"gbgpu_cuda13x"``), a backend object or a
            ``(package, flavor)`` tuple are also accepted (see
            :class:`gbgpu.parallelbase.GBGPUParallelModule`). The backend must carry the
            GB ``sharedmem`` wrap. Default ``None`` takes the first loadable entry of
            :meth:`supported_backends`; those are bare flavors, which resolve to the first
            registered backend of that flavor (``gbt_*`` when gpubackendtools registered
            first), not necessarily a ``gbgpu_*`` one, so pass a flavor explicitly.
        t0 (float, optional): Absolute time [s] of the start of the observation window:
            spacecraft positions are evaluated at ``t0 + linspace(0, T, N, endpoint=False)``
            and the source parameters are defined there. Default ``None`` uses
            ``orbits.sc_t0`` when the orbits have it (:class:`~lisatools.detector.L1Orbits`),
            else ``0.0`` (``Orbits.t0`` is not read).
        flip_ref_phase (bool, optional): If ``True``, negate ``phi0`` before generating
            (the JaxGB phase convention). Without it the templates carry ``exp(+i phi0)``,
            the opposite of the LDC / JaxGB ``exp(-i phi0)`` of the LAT analysis side (see
            ``lisatools.sources.gb.waveform``). Default ``False``; the global-fit callers do
            not set it (the LAT GB transform container owns the phase sign).

    Raises:
        ValueError: If ``orbits`` is not an :class:`~lisatools.detector.Orbits` instance.
        AttributeError: If the resolved backend has no ``sharedmem`` attribute (a
            non-GBGPU backend).

    Attributes:
        gpus (list[int] or None): CUDA device ids for multi-GPU dispatch in
            :meth:`get_ll`, :meth:`get_fstat_ll`, :meth:`generate_global_template`,
            :meth:`swap_likelihood_difference` and the C path of
            :meth:`information_matrix`. ``None`` (default) means one device / CPU. Set it
            after construction (``lisatools.globalfit.recipe`` sets the rank's GPU list).
        d_d (float or xp.ndarray or None): ``<d|d>`` term added in :meth:`get_ll`; must be
            set by the caller first. Default ``None``.
        t0_abs (float): Resolved absolute start time [s] (see ``t0``).
        flip_ref_phase (bool): See the argument.
        force_backend: The ``force_backend`` argument as given.
        shared_mem_backend: ``backend.sharedmem``, the ``GBGPUComputationWrap`` instance
            holding the shared-memory kernels.
        XYZf (xp.ndarray): Set by :meth:`run_wave`: complex FD X, Y, Z templates, shape
            ``(num_bin, 3, N)`` (Python path). See :meth:`run_wave` for the C path.
        AETf (xp.ndarray): Set by :meth:`run_wave`: complex FD A, E, T templates,
            ``A = (Z - X) / sqrt(2)``, ``E = (X - 2 Y + Z) / sqrt(6)``,
            ``T = (X + Y + Z) / sqrt(3)``, shape ``(num_bin, 3, N)`` (Python path).
        start_inds (xp.ndarray): Set by :meth:`run_wave`: per-binary index of the first
            template bin on the ``df = 1 / T`` grid, ``rint(f0 * T) - N / 2``, shape
            ``(num_bin,)``. ``kmin`` is the same array.
        N (int): Points per template of the last :meth:`run_wave` call.
        num_bin (int): Number of binaries of the last call.
        df (float): Frequency resolution ``1 / T`` [Hz] of the last call.
        d_h, h_h (xp.ndarray): Complex per-binary ``<d|h>`` and ``<h|h>`` of the last
            :meth:`get_ll` (``d_h`` is ``|<d|h>|`` with ``phase_maximize=True``).
        phase_angle (xp.ndarray or None): Phase-maximizing rotation of the last
            :meth:`get_ll` / :meth:`swap_likelihood_difference` with ``phase_maximize``.
        A_max, psi_max, iota_max, phi0_max (xp.ndarray): F-statistic maximized extrinsic
            parameters of the last :meth:`get_fstat_ll`.

    Example:
        .. code-block:: python

            from lisatools.detector import EqualArmlengthOrbits
            from gbgpu.gbgpu import GBGPU

            gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                       force_backend="cpu")
            gb.backend.name  # "gbgpu_cpu"
    """

    def __init__(self, orbits: Optional[Orbits | L1Orbits] = None, force_backend = None, t0 = None, flip_ref_phase=False):
        self.force_backend = force_backend
        GBGPUParallelModule.__init__(self, force_backend=self.force_backend)
        
        self.d_d = None
        
        if isinstance(orbits, (Orbits, L1Orbits)):
            self.orbits = orbits
        else:
            raise ValueError("Expected an Orbits object")
        
        self.orbits = orbits
        # `gpus` controls multi-GPU dispatch in get_ll / fill_global etc.
        # `None` -> CPU mode (or single-GPU on a single-device system).
        # A user enabling multi-GPU sets this to a list of device IDs
        # explicitly after construction.
        self.gpus = None

        # handling of correct orbits for Mojito-like data (from mojito branch):
        # absolute start time for spacecraft-position evaluation. ``t0`` arg
        # overrides; otherwise read the orbit's ``sc_t0`` (0.0 if absent).
        if t0 is None:
            t0_abs = getattr(self.orbits, 'sc_t0', 0.0)
        else:
            t0_abs = t0
        self.t0_abs = t0_abs

        # mojito: option to flip the reference phase in waveform generation.
        self.flip_ref_phase = flip_ref_phase

        # mojito alias: the shared-memory backend module. dev accesses it as
        # ``self.backend.sharedmem`` (used in the multi-GPU dispatch loops);
        # the mojito-introduced WaveComp / InfoMat paths use this alias. Both
        # resolve to the same module object.
        self.shared_mem_backend = getattr(self.backend, "sharedmem")

    @classmethod
    def supported_backends(cls):
        """Backend candidates for ``force_backend=None``, most preferred first.

        Returns:
            list[str]: The BARE flavors ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``
            (:meth:`~gpubackendtools.parallelbase.ParallelModuleBase.GPU_RECOMMENDED`),
            without the ``gbgpu_`` prefix, so the default pick can land on another
            package's backend (see the class ``force_backend`` argument).
        """
        return cls.GPU_RECOMMENDED()

    # ---- CPU/GPU dispatch helpers ----
    # `self.gpus is None` -> CPU mode (or single-GPU on a single-device
    # system). The helpers below let the rest of the class call
    # cuda.runtime.setDevice / deviceSynchronize / cuda.Device(...)
    # unconditionally without crashing on numpy.

    def _xp_set_device(self, device) -> None:
        """Set the current CUDA device (``cupy.cuda.runtime.setDevice``); no-op on CPU."""
        if hasattr(self.xp, "cuda"):
            self.xp.cuda.runtime.setDevice(device)

    def _xp_sync(self) -> None:
        """Synchronize the current CUDA device (``deviceSynchronize``); no-op on CPU."""
        if hasattr(self.xp, "cuda"):
            self.xp.cuda.runtime.deviceSynchronize()

    def _xp_device(self, device):
        """Return a ``cupy.cuda.Device(device)`` context manager on GPU, a null context on CPU."""
        if hasattr(self.xp, "cuda"):
            return self.xp.cuda.Device(device)
        import contextlib
        return contextlib.nullcontext()

    def _intra_split_index(self, index_arr, data_splits, gpu, num_per_gpu):
        """Intra-shard row index for each entry of ``index_arr`` on shard ``gpu``.

        The per-GPU buffers hold their rows in ascending global order, so the
        intra-shard index is the *rank* of the row among all rows that
        ``data_splits`` assigns to ``gpu``. Exact for uneven contiguous splits
        (``np.array_split`` with ``n % ngpus != 0``), where the legacy
        ``% num_per_gpu`` block trick misindexes every shard after the first.
        The modulo fallback only remains for callers passing no
        ``data_splits`` (single-buffer path, where rank == global index).

        Args:
            index_arr (xp.ndarray): Global row indices (into the full template / data set)
                of the binaries routed to ``gpu``.
            data_splits (xp.ndarray or None): Device id of every global row.
            gpu (int): The shard's device id.
            num_per_gpu (int): Block size for the modulo fallback.

        Returns:
            xp.ndarray: ``int32`` row indices local to the shard, same shape as ``index_arr``.
        """
        xp = self.xp
        if data_splits is None:
            return (index_arr % num_per_gpu).astype(np.int32)
        shard_rows = xp.where(xp.asarray(data_splits) == gpu)[0]
        return xp.searchsorted(shard_rows, xp.asarray(index_arr)).astype(np.int32)

    @property
    def get_ll_func(self):
        """Backend ``get_ll`` kernel (``GBGPUComputationWrap.get_ll``, gbgpu_utils); unused here."""
        return getattr(self.backend, 'get_ll')

    @property
    def fill_global_func(self):
        """Backend ``fill_global`` kernel (gbgpu_utils), used by :meth:`fill_global_template`."""
        return getattr(self.backend, 'fill_global')

    @property
    def orbits(self) -> Orbits:
        """The :class:`~lisatools.detector.Orbits` object giving positions and ``armlength``.

        The setter stores the object and calls ``orbits._ensure_configured()``; ``None``
        stores a default :class:`~lisatools.detector.EqualArmlengthOrbits` (unreachable
        from the constructor, which rejects ``None``) and a non-``Orbits`` value raises
        ``ValueError``.
        """
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        if orbits is None:
            self._orbits = EqualArmlengthOrbits()
        elif not isinstance(orbits, Orbits):
            raise ValueError(
                "Input orbits must be of type Orbits (from LISA Analysis Tools)"
            )
        else:
            self._orbits = orbits
        self.orbits._ensure_configured()

    @property
    def citation(self):
        """str: BibTeX entries to cite: the GBGPU Zenodo record, Cornish & Littenberg 2007
        (FastGB) and Robson et al. 2018 (third body)."""
        return zenodo + cornish_fastb + robson_triple

    def run_wave(
        self,
        amp,
        f0,
        fdot,
        fddot,
        phi0,
        iota,
        psi,
        lam,
        beta,
        *args,
        N: Optional[int] = None,
        T: float = 4 * YRSID_SI,
        dt: float = 10.0,
        oversample: int = 1,
        tdi2: bool = False,
        tdi_channel_setup: str = "AE",
        use_c_implementation: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0
    ):
        """Generate the FD TDI templates of a batch of binaries (FastGB).

        Based on the FastGB implementation of the LISA Data Challenges' ``ldc`` package.
        Python / ``xp`` path (default): the slow part of the six link responses is built on
        the ``N`` times ``linspace(0, T, N, endpoint=False)`` (positions from
        ``orbits.get_pos`` at ``t0_abs`` plus those times), optionally windowed, FFT'd and
        combined into X, Y, Z;
        A, E, T follow from :func:`gbgpu.utils.utility.AET`. Each template covers the ``N``
        bins starting at ``start_inds = rint(f0 * T) - N / 2`` on the ``df = 1 / T`` grid and
        is scaled by ``0.5 * T / N`` (times the TDI-2 factor when ``tdi2``). Results are
        stored on the object (nothing is returned). Derived classes extend the model through
        the hooks listed in the class docstring.

        Args:
            amp (float or np.ndarray): Dimensionless strain amplitude, scalar or shape
                ``(num_bin,)``. All parameters are made at least 1D and copied to the
                backend (``xp.asarray``).
            f0 (float or np.ndarray): GW frequency at ``t0_abs`` [Hz].
            fdot (float or np.ndarray): First frequency derivative [Hz/s].
            fddot (float or np.ndarray): Second frequency derivative [Hz/s^2].
            phi0 (float or np.ndarray): Initial GW phase [rad] (negated first when
                ``flip_ref_phase``).
            iota (float or np.ndarray): Inclination [rad].
            psi (float or np.ndarray): Polarization angle [rad].
            lam (float or np.ndarray): Sky longitude [rad] in the orbits frame (ICRS right
                ascension with the global fit's L1 orbits).
            beta (float or np.ndarray): Sky latitude [rad] in the orbits frame (ICRS
                declination with L1 orbits); converted to ``theta = pi / 2 - beta``.
            *args: Extra per-binary parameters of a derived class, passed to its
                ``prepare_additional_args`` / ``special_get_N`` (e.g. ``A2, varpi, e2, P2, T2``
                for :class:`gbgpu.thirdbody.GBGPUThirdBody`). Empty for :class:`GBGPU`.
            N: Points per template, shared by the whole batch. Default ``None`` takes the
                maximum of ``special_get_N(amp, f0, T, *args, oversample=oversample)``
                (:func:`gbgpu.utils.utility.get_N` for :class:`GBGPU`).
            T: Observation time [s]. Truncated first to ``int(T / dt) * dt``. Default
                ``4 * YRSID_SI``.
            dt: Data sampling cadence [s]; on the Python path it only enters the
                truncation of ``T``. Default ``10.0``.
            oversample: Multiplier on the automatic ``N`` (only used when ``N`` is
                ``None``). Default ``1``.
            tdi2: Multiply by the second-generation TDI factor
                ``2i sin(2 omega L) exp(-2i omega L)``, ``omega L = 2 pi f0 L / c``, evaluated
                at ``f0``. Default ``False`` (first generation).
            tdi_channel_setup: ``"AE"``, ``"AET"`` or ``"XYZ"``. Used only by the C path
                (the Python path always fills both ``XYZf`` and ``AETf`` with 3 channels).
                Default ``"AE"``.
            use_c_implementation: Run ``backend.sharedmem.SharedMemoryWaveComp_wrap``
                instead of the Python path. Default ``False`` (every global-fit caller uses
                the Python path).
            window: Taper on the slow part, ``"tukey"``, ``"planck"`` or ``None``
                (rectangular). Default ``None``.
            window_alpha: Taper parameter (Tukey alpha / Planck epsilon); must be ``0.0``
                when ``window`` is ``None``. Default ``0.0``.

        Raises:
            ValueError: If ``[t0_abs, t0_abs + T]`` exceeds the orbit time range
                (``orbits.sc_t_base`` when present, else ``orbits.t_base``).
            KeyError: For an unknown ``window``.
            AssertionError: If ``window`` is ``None`` and ``window_alpha != 0``.
            TypeError: If ``*args`` is non-empty on :class:`GBGPU`, whose
                ``prepare_additional_args`` returns (does not raise) a ``ValueError``.

        Note:
            Attributes set: ``XYZf`` and ``AETf``, complex ``xp`` arrays of shape
            ``(num_bin, 3, N)``; ``start_inds`` (= ``kmin``), ``(num_bin,)`` ints; ``N``,
            ``num_bin`` and ``df = 1 / T``. The C path instead sets ``start_inds`` from the
            kernel and ONE of ``XYZf`` (``tdi_channel_setup="XYZ"``) or ``AETf``
            (``"AE"`` / ``"AET"``), shape ``(num_bin, nchannels, N)`` with
            ``nchannels = 2`` for ``"AE"``; it stops in ``breakpoint()`` for an empty batch.

        Example:
            .. code-block:: python

                import numpy as np
                from lisatools.detector import EqualArmlengthOrbits
                from lisatools.utils.constants import YRSID_SI
                from gbgpu.gbgpu import GBGPU

                gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                           force_backend="cpu")
                params = np.tile([2e-23, 2e-3, 7.5e-18, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                 (10, 1))  # (num_bin, 9)
                gb.run_wave(*params.T, T=1.0 * YRSID_SI, dt=10.0, oversample=2)
                A0 = gb.A[0]          # (N,) TDI A of the first binary
                f0_bins = gb.freqs[0]  # (N,) its frequencies [Hz]
        """

        # get number of observation points and adjust T accordingly
        N_obs = int(T / dt)
        T = N_obs * dt
        
        # === MODIFIED: Check orbital data availability ===
        # For L1Orbits, use sc_t_base directly instead of t_base
        t_source = getattr(self.orbits, 'sc_t_base', self.orbits.t_base)
        orbit_t_min = t_source.min()
        orbit_t_max = t_source.max()
        
        if self.t0_abs < orbit_t_min or self.t0_abs + T > orbit_t_max: 
            raise ValueError(
                f"Observation time window [{self.t0_abs:.2f}, {self.t0_abs + T:.2f}] "
                f"exceeds orbital data range [{orbit_t_min:.2f}, {orbit_t_max:.2f}]"
            )

        # if given scalar parameters, make sure at least 1D
        amp = np.atleast_1d(amp)
        f0 = np.atleast_1d(f0)
        fdot = np.atleast_1d(fdot)
        fddot = np.atleast_1d(fddot)
        phi0 = np.atleast_1d(phi0)
        iota = np.atleast_1d(iota)
        psi = np.atleast_1d(psi)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        
        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            phi0 = -phi0

        # if circular base
        if len(args) == 0:
            add_args = ()

        else:
            # add_args = self.prepare_additional_args(*args)
            add_args = getattr(self, "prepare_additional_args")(*args)

        # set N if it is not given based on timescales in the waveform
        if N is None:
            # N_temp = self.special_get_N(amp, f0, T, *args, oversample=oversample)
            N_temp = getattr(self, "special_get_N")(amp, f0, T, *args, oversample=oversample)
            
            if hasattr(N_temp.max(), "item"):
                N = int(N_temp.max().item())
            else:
                N = int(N_temp.max())

        self.N = N
        
        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # get spacecraft positions
        tm_rel = self.xp.linspace(0, T, num=N, endpoint=False)
        tm_abs = tm_rel + self.t0_abs
        Ps = self._spacecraft(tm_abs)
        Ps_arr = self.xp.array(Ps).flatten()
        
        # number of binaries is determined from length of amp array
        self.num_bin = num_bin = len(amp)

        # polar angle from ecliptic latitude
        theta = np.pi / 2 - beta

        # bin spacing
        self.df = df = 1 / T

        # instantiate GPU/CPU arrays

        # copy to GPU if needed
        amp = self.xp.asarray(amp.copy())
        f0 = self.xp.asarray(f0.copy())  # in mHz
        fdot = self.xp.asarray(fdot.copy())
        fddot = self.xp.asarray(fddot.copy())
        phi0 = self.xp.asarray(phi0.copy())
        iota = self.xp.asarray(iota.copy())
        psi = self.xp.asarray(psi.copy())
        lam = self.xp.asarray(lam.copy())
        theta = self.xp.asarray(theta.copy())
        
        cosiota = self.xp.cos(iota.copy())
        
        # figure out start inds
        # q_check = (f0 * T).astype(np.int32)
        # self.start_inds = (q_check - N / 2).astype(xp.int32)
        # self.start_inds = (q_check - N / 2).astype(xp.int32)

        cosiota = self.xp.cos(iota)

        # transfer frequency
        fstar = C_SI / (self.orbits.armlength * 2 * np.pi)
        
        if use_c_implementation:
            if tdi_channel_setup == "AE":
                nchannels = 2
            else:
                nchannels = 3

            tm_abs = tm_rel + self.t0_abs
            response_out = self.xp.ones((self.num_bin * nchannels * int(N),), dtype=complex)
            _start_inds = self.xp.zeros((self.num_bin,), dtype=np.int32)
            if self.num_bin == 0:
                breakpoint()
                        
            tuple_in = (
                response_out,
                _start_inds,
                amp, f0, fdot, fddot, phi0, iota, psi, lam, theta,
                T, dt, N, num_bin, 
                tdi_channel_setup_map[tdi_channel_setup],
                Ps_arr, self.orbits.armlength, tdi2,
                window_type, window_alpha
            )
            # self.backend.sharedmem.SharedMemoryWaveComp_wrap(*tuple_in)
            self.shared_mem_backend.SharedMemoryWaveComp_wrap(*tuple_in)
            self.start_inds = _start_inds
            response_out = response_out.reshape(self.num_bin, nchannels, N)

            if tdi_channel_setup == "XYZ":
                self.XYZf = response_out
            else:
                self.AETf = response_out
            # setup waveforms for efficient GPU likelihood or global template building
            return

        cosps, sinps = self.xp.cos(2.0 * psi), self.xp.sin(2.0 * psi)

        Aplus = amp * (1.0 + cosiota * cosiota)
        Across = -2.0 * amp * cosiota

        DP = Aplus * cosps - 1.0j * Across * sinps
        DC = -Aplus * sinps - 1.0j * Across * cosps

        # sky location basis vectors
        sinth, costh = self.xp.sin(theta), self.xp.cos(theta)
        sinph, cosph = self.xp.sin(lam), self.xp.cos(lam)
        u = self.xp.array([costh * cosph, costh * sinph, -sinth]).T[:, None, :]
        v = self.xp.array([sinph, -cosph, self.xp.zeros_like(cosph)]).T[:, None, :]
        k = self.xp.array([-sinth * cosph, -sinth * sinph, -costh]).T[:, None, :]

        # polarization tensors
        eplus = self.xp.matmul(v.transpose(0, 2, 1), v) - self.xp.matmul(
            u.transpose(0, 2, 1), u
        )
        ecross = self.xp.matmul(u.transpose(0, 2, 1), v) + self.xp.matmul(
            v.transpose(0, 2, 1), u
        )

        # # time points evaluated
        # # tm = self.xp.linspace(0, T, num=N, endpoint=False)
        # tm = self.xp.linspace(t0_abs, t0_abs + T, num=N, endpoint=False)
        # # get the spacecraft positions from orbits
        # Ps = self._spacecraft(tm)
    
        # time domain information
        # === MODIFIED: Pass RELATIVE time to waveform construction ===
        Gs, q = self._construct_slow_part(
            T,
            self.orbits.armlength,
            Ps,
            tm_rel,
            f0,
            fdot,
            fddot,
            fstar,
            phi0,
            k,
            DP,
            DC,
            eplus,
            ecross,
            *add_args,
        )

        if window_type != 0:
            w_arr = self._get_window_array(window_type, window_alpha, N)
            for ij in Gs.keys():
                Gs[ij] =  Gs[ij] * w_arr[None,:]
        
        # transform to TDI observables
        XYZf, f_min = self._computeXYZ(T, Gs, f0, fdot, fddot, fstar, amp, q, tm_rel)
        # NOTE this means that the correct times are passed down to construct_slow_part and _computeXYZ
        # Only need to check if these arguments are used correctly
        self.start_inds = self.kmin = self.xp.round(f_min / df).astype(int)
        fctr = 0.5 * T / N

        # adjust for TDI2 if needed
        if tdi2:
            omegaL = 2 * np.pi * f0 * (self.orbits.armlength / C_SI)
            tdi2_factor = 2.0j * self.xp.sin(2 * omegaL) * self.xp.exp(-2j * omegaL)
            fctr *= tdi2_factor

        if isinstance(fctr, (float, int)):
            fctr = self.xp.array([fctr])

        XYZf *= fctr[:, None, None]

        # we do not care about T right now
        self.AETf = self.xp.asarray(AET(XYZf[:, 0], XYZf[:, 1], XYZf[:, 2])).transpose(
            1, 0, 2
        )
        # setup waveforms for efficient GPU likelihood or global template building
        self.XYZf = XYZf

    @property
    def X_out(self):
        """xp.ndarray: X of the last run, ``XYZf[:, 0].T.flatten()``, shape ``(N * num_bin,)``."""
        return self.XYZf[:, 0].T.flatten()

    @property
    def Y_out(self):
        """xp.ndarray: Y of the last run, ``XYZf[:, 1].T.flatten()``, shape ``(N * num_bin,)``."""
        return self.XYZf[:, 1].T.flatten()

    @property
    def Z_out(self):
        """xp.ndarray: Z of the last run, ``XYZf[:, 2].T.flatten()``, shape ``(N * num_bin,)``."""
        return self.XYZf[:, 2].T.flatten()

    @property
    def A_out(self):
        """xp.ndarray: A of the last run, ``AETf[:, 0].T.flatten()``, shape ``(N * num_bin,)``.

        This is the 1D layout :meth:`fill_global_template` accepts.
        """
        return self.AETf[:, 0].T.flatten()

    @property
    def E_out(self):
        """xp.ndarray: E of the last run, ``AETf[:, 1].T.flatten()``, shape ``(N * num_bin,)``."""
        return self.AETf[:, 1].T.flatten()

    @property
    def T_out(self):
        """xp.ndarray: T of the last run, ``AETf[:, 2].T.flatten()``, shape ``(N * num_bin,)``.

        Raises:
            ValueError: If ``AETf`` holds only A and E (C path with ``"AE"``).
        """
        if self.AETf.shape[1] == 2: # TODO: can this be removed?
            raise ValueError("Requesting T channel when AE was generated.")

        return self.AETf[:, 2].T.flatten()

    def _computeXYZ(self, T, Gs, f0, fdot, fddot, fstar, ampl, q, tm):
        """Combine the link transfer functions into FD Michelson X, Y, Z (LDC FastGB).

        Args:
            T (float): Observation time [s].
            Gs (dict): Slow-part link responses from :meth:`_construct_slow_part`, keys
                ``"12"``, ``"23"``, ``"31"``, ``"21"``, ``"32"``, ``"13"``, each
                ``(num_bin, N)`` complex.
            f0, fdot, fddot (xp.ndarray): ``(num_bin,)`` frequency parameters.
            fstar (float): Transfer frequency ``C_SI / (2 pi armlength)`` [Hz].
            ampl (xp.ndarray): ``(num_bin,)`` amplitudes (factored out of the transfer
                function and restored after the FFT).
            q (np.ndarray): ``(num_bin,)`` carrier bin ``rint(f0 * T)``.
            tm (xp.ndarray): ``(N,)`` times relative to ``t0_abs`` [s].

        Returns:
            tuple: ``(XYZf, f_min)``: the fftshifted length-``N`` FFT of the slow X, Y, Z,
            ``(num_bin, 3, N)`` complex (before the ``0.5 T / N`` scaling of
            :meth:`run_wave`), and the frequency of the first bin ``(q - N / 2) / T`` [Hz].
        """

        # get true frequency as a function of time
        f = (
            f0[:, None]
            + fdot[:, None] * tm[None, :]
            + 1 / 2 * fddot[:, None] * tm[None, :] ** 2
        )

        # compute transfer function
        omL = f / fstar
        SomL = self.xp.sin(omL)
        fctr = self.xp.exp(-1.0j * omL)
        fctr2 = 4.0 * omL * SomL * fctr / ampl[:, None]

        # Notes from LDC below

        ### I have factored out 1 - exp(1j*omL) and transformed to
        ### fractional frequency: those are in fctr2
        ### I have rremoved Ampl to reduce dynamical range, will restore it later

        Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
        Ysl = Gs["32"] - Gs["12"] + (Gs["23"] - Gs["21"]) * fctr
        Zsl = Gs["13"] - Gs["23"] + (Gs["31"] - Gs["32"]) * fctr

        # time domain slow part
        XYZsl = fctr2[:, None, :] * self.xp.array([Xsl, Ysl, Zsl]).transpose(1, 0, 2)

        # frequency domain slow part
        XYZf_slow = ampl[:, None, None] * self.xp.fft.fft(XYZsl, axis=-1)

        # for testing
        # Xtry = 4.0*(self.G21 - self.G31 + (self.G12 - self.G13)*fctr)/self.ampl

        M = XYZf_slow.shape[2]  # len(XYZf_slow)
        XYZf = self.xp.fft.fftshift(XYZf_slow, axes=-1)

        # closest bin frequency
        f0 = (q - M / 2) / T  # freq = (q + self.xp.arange(M) - M/2)/T
        return XYZf, f0

    def _spacecraft(self, t):
        """Spacecraft positions at absolute times ``t`` [s].

        Args:
            t (xp.ndarray): ``(N,)`` absolute times [s].

        Returns:
            list: ``[P1, P2, P3]``, each ``(N, 3)`` positions [m] from ``orbits.get_pos``.
        """

        # output arrays
        P1 = self.orbits.get_pos(t, self.xp.full_like(t, 1).astype(self.xp.int32))
        P2 = self.orbits.get_pos(t, self.xp.full_like(t, 2).astype(self.xp.int32))
        P3 = self.orbits.get_pos(t, self.xp.full_like(t, 3).astype(self.xp.int32))
        return [P1, P2, P3]

    def _construct_slow_part(
        self,
        T,
        arm_length,
        Ps,
        tm,
        f0,
        fdot,
        fddot,
        fstar,
        phi0,
        k,
        DP,
        DC,
        eplus,
        ecross,
        *add_args,
    ):
        """Build the slow (heterodyned) part of the six link responses in the time domain.

        Computes the delayed times ``xi = tm - k.P_i / c`` at each spacecraft, the
        instantaneous frequencies (passed through ``shift_frequency``), the polarization
        projections onto each arm and the phase ``om k.P_i / c - argS`` (passed through
        ``add_to_phasing``), then the LDC FastGB transfer functions ``G_ij``.

        Args:
            T (float): Observation time [s].
            arm_length (float): Constant arm length [m] normalizing the arm unit vectors.
            Ps (list): ``[P1, P2, P3]`` spacecraft positions, each ``(N, 3)`` [m].
            tm (xp.ndarray): ``(N,)`` times relative to ``t0_abs`` [s].
            f0, fdot, fddot, phi0 (xp.ndarray): ``(num_bin,)`` source parameters.
            fstar (float): Transfer frequency [Hz].
            k (xp.ndarray): ``(num_bin, 1, 3)`` propagation direction.
            DP, DC (xp.ndarray): ``(num_bin,)`` complex plus / cross polarization amplitudes.
            eplus, ecross (xp.ndarray): ``(num_bin, 3, 3)`` polarization tensors.
            *add_args: Output of ``prepare_additional_args`` (derived classes).

        Returns:
            tuple: ``(Gs, q)``: dict of ``(num_bin, N)`` complex link responses keyed
            ``"12"``, ``"23"``, ``"31"``, ``"21"``, ``"32"``, ``"13"``, and the carrier bins
            ``q = rint(f0 * T)``.
        """

        # these are the orbits (equal-arm lengths assumed)
        P1, P2, P3 = Ps
        r = dict()

        # unit vectors of constellation arms
        r["12"] = (P2 - P1) / arm_length  ## [3xNt]
        r["13"] = (P3 - P1) / arm_length
        r["23"] = (P3 - P2) / arm_length
        r["31"] = -r["13"]

        # wave propagation axis dotted with constellation unit vectors
        kdotr = dict()
        for ij in ["12", "13", "23"]:
            kdotr[ij] = self.xp.dot(k.squeeze(), r[ij].T)  ### should be size Nt
            kdotr[ij[-1] + ij[0]] = -kdotr[ij]

        # wave propagation axis dotted with spacecraft positions
        kdotP = self.xp.array(
            [self.xp.dot(k, P1.T), self.xp.dot(k, P2.T), self.xp.dot(k, P3.T)]
        )[:, :, 0].transpose(1, 0, 2)

        kdotP /= C_SI

        Nt = len(tm)

        # delayed time at the spacecraft
        xi = tm - kdotP

        # instantaneous frequency of wave at the spacecraft at xi
        fi = (
            f0[:, None, None]
            + fdot[:, None, None] * xi
            + 1 / 2.0 * fddot[:, None, None] * xi**2
        )

        # for regular GBGPU shift is zero
        # fi[:] = self.shift_frequency(fi, xi, *add_args)
        fi[:] = getattr(self, "shift_frequency")(fi, xi, *add_args)

        # transfer frequency ratio
        fonfs = fi / fstar  # Ratio of true frequency to transfer frequency

        # LDC notes with '###'
        ### compute transfer f-n
        q = np.rint(f0 * T)  # index of nearest Fourier bin
        df = 2.0 * np.pi * (q / T)
        om = 2.0 * np.pi * f0

        ### The expressions below are arg2_i with om*kR_i factored out
        A = dict()
        for ij in ["12", "23", "31"]:
            aij = (
                self.xp.dot(eplus, r[ij].T) * r[ij].T * DP[:, None, None]
                + self.xp.dot(ecross, r[ij].T) * r[ij].T * DC[:, None, None]
            )
            A[ij] = aij.sum(axis=1)

        # below is information from the LDC about matching the original LDC.
        # The current code matches the time-domain-generated tempaltes in the LDC.

        # These are wfm->TR + 1j*TI in c-code

        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*fdot*(xi[0]**2)
        # arg2_2 = 2.0*np.pi*f0*xi[1] + phi0 - df*tm + np.pi*fdot*(xi[1]**2)
        # arg2_3 = 2.0*np.pi*f0*xi[2] + phi0 - df*tm + np.pi*fdot*(xi[2]**2)

        ### These (y_sr) reproduce exactly the FastGB results
        # self.y12 = 0.25*np.sin(arg12)/arg12 * np.exp(1.j*(arg12 + arg2_1)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y23 = 0.25*np.sin(arg23)/arg23 * np.exp(1.j*(arg23 + arg2_2)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y31 = 0.25*np.sin(arg31)/arg31 * np.exp(1.j*(arg31 + arg2_3)) * ( Dp31*self.DP + Dc31*self.DC )
        # self.y21 = 0.25*np.sin(arg21)/arg21 * np.exp(1.j*(arg21 + arg2_2)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y32 = 0.25*np.sin(arg32)/arg32 * np.exp(1.j*(arg32 + arg2_3)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y13 = 0.25*np.sin(arg13)/arg13 * np.exp(1.j*(arg13 + arg2_1)) * ( Dp31*self.DP + Dc31*self.DC )

        ### Those are corrected values which match the time domain results.
        ## om*kdotP_i singed out for comparison with another code.

        argS = (
            phi0[:, None, None]
            + (om[:, None, None] - df[:, None, None]) * tm[None, None, :]
            + np.pi * fdot[:, None, None] * (xi**2)
            + 1 / 3 * np.pi * fddot[:, None, None] * (xi**3)
        )

        # called kdotP in LDC code
        arg_phasing = om[:, None, None] * kdotP - argS
        # arg_phasing[:] = self.add_to_phasing(arg_phasing, f0, fdot, fddot, xi, *add_args)
        arg_phasing[:] = getattr(self, "add_to_phasing")(arg_phasing, f0, fdot, fddot, xi, *add_args)
        
        # get Gs transfer functions
        Gs = dict()
        for ij, ij_sym, s in [
            ("12", "12", 0),
            ("23", "23", 1),
            ("31", "31", 2),
            ("21", "12", 1),
            ("32", "23", 2),
            ("13", "31", 0),
        ]:

            # TODO: evolution of the amplitude
            arg_ij = 0.5 * fonfs[:, s, :] * (1 + kdotr[ij])
            Gs[ij] = (
                0.25
                * self.xp.sin(arg_ij)
                / arg_ij
                * self.xp.exp(-1.0j * (arg_ij + arg_phasing[:, s]))
                * A[ij_sym]
            )
        ### Lines blow are extractions from another python code and from C-code in LDC
        # y = -0.5j*self.omL*A*sinc(args)*np.exp(-1.0j*(args + self.om*kq))
        # args = 0.5*self.omL*(1.0 - kn)
        # arg12 = 0.5*fonfs[0,:] * (1 + kdotr12)
        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*self.fdot*(xi[0]**2)  -> om*k.Ri
        # arg1 = 0.5*wfm->fonfs[i]*(1. + wfm->kdotr[i][j])
        # arg2 =  PI*2*f0*wfm->xi[i] + phi0 - df*t
        # sinc = 0.25*sin(arg1)/arg1
        # tran1r = aevol*(wfm->dplus[i][j]*wfm->DPr + wfm->dcross[i][j]*wfm->DCr)
        # tran1i = aevol*(wfm->dplus[i][j]*wfm->DPi + wfm->dcross[i][j]*wfm->DCi)
        # tran2r = cos(arg1 + arg2)
        # tran2i = sin(arg1 + arg2)
        # wfm->TR[i][j] = sinc*(tran1r*tran2r - tran1i*tran2i)
        # wfm->TI[i][j] = sinc*(tran1r*tran2i + tran1i*tran2r)
        return Gs, q

    def _get_window_array(self, window_type: int, window_alpha: float, N: int) -> Any:
        """Taper applied to the slow part by :meth:`run_wave` (Python path).

        Args:
            window_type: ``0`` rectangular, ``1`` Tukey (``window_alpha`` = alpha), ``2``
                Planck (``window_alpha`` = epsilon); codes from ``window_map``.
            window_alpha: Taper parameter; ``<= 0`` gives the rectangular window.
            N: Number of samples.

        Returns:
            xp.ndarray: ``(N,)`` float64 window.

        Raises:
            ValueError: For an unknown ``window_type``.
        """
        if window_type == 0 or window_alpha <= 0.0 or N <= 1:
            return self.xp.ones(N, dtype=self.xp.float64)
        
        i = self.xp.arange(N)
        if window_type == 1:  # Tukey
            alpha = window_alpha
            w = self.xp.zeros(N)
            r = (2.0 * i) / (alpha * (N - 1))
            l1 = int(self.xp.rint((alpha * (N - 1)) / 2.0))
            l2 = int(self.xp.rint((N - 1) * (1.0 - alpha / 2.0)))
            
            mask1 = i < l1
            w = self.xp.where(mask1, 0.5 * (1.0 + self.xp.cos(np.pi * (r - 1.0))), w)
            mask2 = (i >= l1) & (i < l2)
            w = self.xp.where(mask2, 1.0, w)
            mask3 = i >= l2
            w = self.xp.where(mask3, 0.5 * (1.0 + self.xp.cos(np.pi * (r - 2.0 / alpha + 1.0))), w)
            return w

        elif window_type == 2:  # Planck
            epsilon = window_alpha
            w = self.xp.zeros(N)
            n1 = epsilon * (N - 1)
            n2 = (1.0 - epsilon) * (N - 1)

            mask1 = i < n1
            z1 = self.xp.where(mask1, epsilon * (N - 1) / (i + 1e-15) + epsilon * (N - 1) / ((i - epsilon * (N - 1)) + 1e-15), 0.0)
            w = self.xp.where(mask1, 1.0 / (1.0 + self.xp.exp(self.xp.clip(z1, -700.0, 700.0))), w)

            mask2 = (i >= n1) & (i <= n2)
            w = self.xp.where(mask2, 1.0, w)

            mask3 = i > n2
            z2 = self.xp.where(mask3, epsilon * (N - 1) / ((N - 1) - i + 1e-15) + epsilon * (N - 1) / ((N - 1) - i - epsilon * (N - 1) + 1e-15), 0.0)
            w = self.xp.where(mask3, 1.0 / (1.0 + self.xp.exp(self.xp.clip(z2, -700.0, 700.0))), w)
            return w
        
        else:
            raise ValueError("Window type not supported. Supported types are 'None': 0 (retangular/no window), 'tukey': 1 (tukey window) 'planck': 2 (planck window).")

    
    @property
    def A(self):
        """xp.ndarray: TDI A templates of the last :meth:`run_wave`, shape ``(num_bin, N)``."""
        return self.AETf[:, 0]

    @property
    def E(self):
        """xp.ndarray: TDI E templates of the last :meth:`run_wave`, shape ``(num_bin, N)``."""
        return self.AETf[:, 1]

    @property
    def T(self):
        """xp.ndarray: TDI T templates of the last :meth:`run_wave`, shape ``(num_bin, N)``.

        Raises:
            ValueError: If ``AETf`` holds only A and E (C path with ``"AE"``).
        """
        if self.AETf.shape[1] == 2:
            raise ValueError("Requesting T channel when AE was generated.")
        return self.AETf[:, 2]

    @property
    def X(self):
        """xp.ndarray: TDI X templates of the last :meth:`run_wave`, shape ``(num_bin, N)``."""
        return self.X_out.reshape(self.N, self.num_bin).T

    @property
    def Y(self):
        """xp.ndarray: TDI Y templates of the last :meth:`run_wave`, shape ``(num_bin, N)``."""
        return self.Y_out.reshape(self.N, self.num_bin).T

    @property
    def Z(self):
        """xp.ndarray: TDI Z templates of the last :meth:`run_wave`, shape ``(num_bin, N)``."""
        return self.Z_out.reshape(self.N, self.num_bin).T

    @property
    def freqs(self):
        """xp.ndarray: Frequencies [Hz] of each template bin, shape ``(num_bin, N)``.

        Bin ``k`` of binary ``i`` sits at ``(start_inds[i] + k) * df``.
        """
        freqs_out = (
            self.xp.arange(self.N)[None, :] + self.start_inds[:, None]
        ) * self.df
        return freqs_out

    def get_ll(
        self,
        params,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=False,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        oversample=1,
        return_cupy=False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0
        # **kwargs
    ):
        """Batched FD log-likelihood of GB templates against residual data (legacy kernel).

        Generates each binary's template inside ``backend.sharedmem.SharedMemoryLikeComp_wrap``
        and accumulates ``<d|h> = 4 df sum_k conj(d_k) h_k psd_k`` and ``<h|h>`` (per channel
        for ``"AE"`` / ``"AET"``, over the full channel matrix for ``"XYZ"``), ``df = 1 / T``.
        Binaries are grouped by template length ``N`` and routed to devices by
        ``data_splits[data_index]``. Not on the global-fit path (WDM runs use
        :class:`gbgpu.gbcomps.GBWDMComputations`); the LAT callers are ``gathergalaxy`` and
        the FD-only ``gbspecialstretch.para_log_like``.

        Args:
            params (xp.ndarray): ``(num_bin, 9)`` physical parameters, columns
                ``(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta)`` (see the module
                docstring).
            data_minus_template (xp.ndarray or list): Complex residual data. One array, or a
                list with one array per device of ``gpus``. Each is ``(nchannels,
                data_length)`` (one data row), ``(num_data, nchannels, data_length)``, or the
                flat 1D form of the latter (then ``data_length`` is required). ``nchannels`` is
                2 for ``"AE"``, else 3.
            psd (xp.ndarray or list): Noise weights multiplied into the inner product, i.e.
                the INVERSE PSD (``"AE"`` / ``"AET"``: ``(nchannels, data_length)`` or
                ``(num_psd, nchannels, data_length)``) or the inverse cross-spectral matrix
                (``"XYZ"``: ``(3, 3, data_length)`` or ``(num_psd, 3, 3, data_length)``), or
                flat 1D. Same list-per-device convention; cast to complex128.
            start_freq_ind (int or xp.ndarray, optional): Index on the ``df`` grid of the
                first bin of the data rows: an int for all rows, or an int32 array of length
                ``num_data``. Default ``0``.
            data_index (xp.ndarray, optional): ``(num_bin,)`` int32 data row of each binary.
                Default ``None`` (all row 0).
            noise_index (xp.ndarray, optional): ``(num_bin,)`` int32 psd row of each binary.
                Default ``None`` (all row 0).
            adjust_inplace (bool, optional): Ignored.
            use_c_implementation (bool, optional): Ignored (always the kernel).
            N (int or xp.ndarray, optional): Template length: one int, or ``(num_bin,)``.
                Default ``None`` uses :func:`gbgpu.utils.utility.get_N` on ``(amp, f0)``.
            T (float, optional): Observation time [s] (not truncated to ``dt``). Default
                ``4 * YRSID_SI``.
            dt (float, optional): Sampling cadence [s] passed to the kernel. Default ``10.0``.
            data_length (int, optional): Bins per data row; required for flat 1D inputs,
                otherwise read from the array shapes.
            data_splits (xp.ndarray, optional): Device id of every data row. Default ``None``
                puts all rows on one device (requires ``gpus`` to be ``None`` or one id).
            phase_maximize (bool, optional): Replace ``<d|h>`` by ``|<d|h>|`` (analytic
                maximization over ``phi0``); the rotation is stored on ``phase_angle`` and
                the complex value on ``non_marg_d_h`` (host). Default ``False``.
            tdi_channel_setup (str, optional): ``"AE"``, ``"AET"`` or ``"XYZ"``. Default
                ``"AE"``.
            num_per_gpu (int, optional): Must be ``None``.
            oversample (int, optional): Multiplier on the automatic ``N``. Default ``1``.
            return_cupy (bool, optional): Return the backend array instead of a host one.
                Default ``False``.
            tdi2: Second-generation TDI factor (as in :meth:`run_wave`). Default ``False``.
            window: ``"tukey"``, ``"planck"`` or ``None``. Default ``None``.
            window_alpha: Taper parameter; ``0.0`` when ``window`` is ``None``.

        Returns:
            np.ndarray: ``(num_bin,)`` float values of ``-0.5 * Re(d_d + <h|h> - 2 <d|h>)``
            (an ``xp`` array with ``return_cupy=True``). The complex ``d_h`` / ``h_h`` are kept
            on the object.

        Raises:
            ValueError: If ``self.d_d`` is ``None``, or an automatic ``N`` is zero.
            NotImplementedError: If ``num_per_gpu`` is given.
            KeyError: For an unknown ``window``.
            AssertionError: For index arrays that are not int32 or inconsistent shapes.

        Warning:
            With ``flip_ref_phase=True`` the ``phi0`` column of the CALLER's ``params`` is
            negated in place on every call. A ``ValueError`` raised by the kernel drops into
            ``breakpoint()``.
        """
        if self.d_d is None:
            raise ValueError(
                "self.d_d attribute must be set before computing log-Likelihood. This attribute is the data with data inner product (<d|d>)."
            )

        if num_per_gpu is not None:
            raise NotImplementedError("Need to check this.")

        if self.gpus is not None:
            # set first index gpu device to control main operations
            return_to_main_device = self.gpus[0]
            self._xp_set_device(return_to_main_device)

        self.num_bin = num_bin = params.shape[0]
        
        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params[:, 4] = -params[:, 4]

        if N is None:
            # TODO: G
            N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert params.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]

        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # fill index values if not given
        if data_index is None:
            data_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        # check that index values are ready for computation
        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32

        nchannels = 3 if tdi_channel_setup != "AE" else 2

        if isinstance(data_minus_template, self.xp.ndarray):
            data_minus_template_in = [data_minus_template.astype(self.xp.complex128)]
        else:
            data_minus_template_in = [tmp.astype(self.xp.complex128) for tmp in data_minus_template]

        if isinstance(psd, self.xp.ndarray):
            psd_in = [psd.astype(self.xp.complex128)]
        else:
            psd_in = [p.astype(self.xp.complex128) for p in psd]

        num_data = []
        for t_i, t in enumerate(data_minus_template_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_data.append(int(t.shape[0] / (data_length * nchannels)))
                
            elif t.ndim == 2:
                num_data.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # TODO: print("check this does not create memory issues")
                data_minus_template_in[t_i] = t.flatten()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_data.append(ntemplate)
                assert _nchannels == nchannels
                # TODO: print("check this does not create memory issues")
                
                data_minus_template_in[t_i] = t.flatten()

        if tdi_channel_setup == "AE" or tdi_channel_setup == "AET":
            # assumes nchannels will 1D really
            psd_sub_shape = (nchannels,)
            
        else:
            assert nchannels == 3
            psd_sub_shape = (nchannels, nchannels)

        num_psd = []
        for t_i, t in enumerate(psd_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                
            elif t.ndim == 2:
                assert tdi_channel_setup in ["AET", "AE"]
                num_psd.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()

            elif t.ndim == 3:
                if tdi_channel_setup in ["AE", "AET"]:
                    ntemplate, _nchannels, data_length = t.shape
                    num_psd.append(ntemplate)
                    assert t.shape[1] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()


                else:  # XYZ
                    _nchannels, _nchannels, data_length = t.shape
                    num_psd.append(1)
                    assert t.shape[1] == t.shape[0] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()
                assert _nchannels == nchannels
                
            elif t.ndim == 4:
                assert tdi_channel_setup == "XYZ"
                ntemplate, _nchannels, _nchannels, data_length = t.shape
                num_psd.append(ntemplate)
                assert t.shape[1] == t.shape[2] == nchannels
                data_length = t.shape[3]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()
            
                assert _nchannels == nchannels
            # print("check this does not create memory issues")
        
        # initialize Likelihood terms <d|h> and <h|h>
        d_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        h_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        
        # if self.gpus is not None:
        do_synchronize = False
        main_device = self.xp.cuda.runtime.getDevice() if hasattr(self.xp, "cuda") else -1
        # CPU mode (self.gpus is None) -> treat as a single virtual
        # device (gpu_id = -1) so the data_splits / num_per_gpu defaults +
        # the dispatch loop below run exactly once and skip every
        # cupy-specific setDevice / synchronize call.
        _gpus_iter = self.gpus if self.gpus is not None else [-1]
        if data_splits is None:
            assert len(_gpus_iter) == 1
            data_splits = self.xp.full(num_data[0], _gpus_iter[0])

        if num_per_gpu is None:
            # No batching cap: valid for ANY gpu count (the dispatch loop
            # routes rows by data_splits); only the data_splits default
            # above is inherently single-device.
            num_per_gpu = int(2**31 - 1)
            # make really high so just keeps (int32-safe: numpy 2 rejects
            # int32 arrays modulo a Python int beyond the int32 range)

        inputs_in = []
        for nnn, N_here in enumerate(unique_N):
            N_here = N_here.item()
            # get spacecraft positions (mojito: real-orbit Ps_arr threaded
            # into the kernels for mojito-orbit / TDI2 aware computation).
            tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
            tm_abs = tm_rel + self.t0_abs
            Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

            for gpu_i, gpu in enumerate(_gpus_iter):
                if self.gpus is not None:
                    self._xp_set_device(main_device)
                keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[data_index] == gpu)
                num_split_here = keep_bool.sum().item()
                inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                if num_split_here == 0:
                    continue
                if self.gpus is not None:
                    self._xp_set_device(gpu)
                
                params_here = self.xp.asarray(params)[keep_bool]
                
                # theta_add = np.pi / 2 - beta_add
                params_here[:, 8] = np.pi / 2 - params_here[:, 8]
                
                data_minus_template_here = data_minus_template_in[gpu_i]
                psd_here = psd_in[gpu_i]

                params_tuple = tuple([pars_tmp.copy()for pars_tmp in params_here.T])
                
                if isinstance(start_freq_ind, int):
                    start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
                
                else:
                    assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                    # TODO: fix this num_data
                    
                    start_freq_ind_tmp = start_freq_ind
                assert len(start_freq_ind_tmp) == num_data[gpu_i]

                d_h_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                h_h_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                
                noise_index_in = self.xp.asarray(noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                data_index_in = self.xp.asarray(data_index[keep_bool] % num_per_gpu).astype(np.int32)
        
                tuple_in = (
                    (
                        d_h_temp,
                        h_h_temp,
                        data_minus_template_here,
                        psd_here,
                        data_index_in,
                        noise_index_in,
                    )
                    + params_tuple
                    + (
                        T, dt, N_here, 
                        num_split_here, start_freq_ind_tmp, data_length, 
                        tdi_channel_setup_map[tdi_channel_setup], 
                        gpu, do_synchronize, 
                        num_data[0], num_psd[0],
                        Ps_arr, self.orbits.armlength, tdi2,
                        window_type, window_alpha
                    )
                )    
                
                # self.xp.cuda.runtime.deviceSynchronize()

                # for testing
                try:
                    self._xp_sync()
                    self.backend.sharedmem.SharedMemoryLikeComp_wrap(*tuple_in)
                    inputs_in.append([gpu, inds_here, tuple_in])
                    self._xp_sync()
                except ValueError:
                    breakpoint()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(gpu):
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(main_device):
                self._xp_sync()

                d_h[inds_gpu] = inputs_tmp[0][:]
                h_h[inds_gpu] = inputs_tmp[1][:]
                
        self._xp_set_device(main_device)
        self._xp_sync()
        
        if phase_maximize:
            self.non_marg_d_h = d_h.copy()
            # Maximising rotation (matches swap_likelihood_difference's
            # convention): d_h * exp(-1j * phase_angle) is real-positive.
            # Callers subtract this from the sampling-basis phi0 on accept.
            self.phase_angle = self.xp.arctan2(d_h.imag, d_h.real)
            try:
                self.non_marg_d_h = self.non_marg_d_h.get()
            except AttributeError:
                pass

            d_h = self.xp.abs(d_h)
        else:
            self.phase_angle = None

        # store these likelihood terms for later if needed
        self.h_h = h_h
        self.d_h = d_h

        # compute Likelihood
        like_out = -1.0 / 2.0 * (self.d_d + h_h - 2 * d_h).real

        if return_cupy:
            return like_out

        # back to CPU if on GPU
        try:
            return like_out.get()

        except AttributeError:
            return like_out


    def get_fstat_ll(
        self,
        params,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=False,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        oversample=1,
        return_cupy=False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0
        # **kwargs
    ):
        """Batched FD F-statistic of GB templates against residual data (legacy kernel).

        ``backend.sharedmem.SharedMemoryFstatLikeComp_wrap`` builds, per binary, the four
        F-statistic basis filters ``A_i`` sharing its ``(f0, fdot, fddot, lam, beta)`` and
        returns ``N_i = <d|A_i>`` and ``M_ij = <A_i|A_j>`` (inner product as in
        :meth:`get_ll`). The statistic is ``N^T M^-1 N``; the amplitudes ``a = M^-1 N`` are
        also turned into the maximizing ``A_max``, ``psi_max`` (mod pi), ``iota_max``
        (mod pi) and ``phi0_max`` (mod 2 pi), stored on the object. FD-only: the LAT caller
        (``gbspecialstretch.para_log_like``) refuses non-FD runs.

        Args:
            params (xp.ndarray): ``(num_bin, 5)`` intrinsic parameters
                ``(f0 [Hz], fdot [Hz/s], fddot [Hz/s^2], lam, beta)`` (sky in the orbits
                frame). Not the 9-column layout of :meth:`get_ll`.
            data_minus_template (xp.ndarray or list): As in :meth:`get_ll`.
            psd (xp.ndarray or list): As in :meth:`get_ll`.
            start_freq_ind, data_index, noise_index: As in :meth:`get_ll`.
            T, dt, data_length, data_splits, tdi_channel_setup: As in :meth:`get_ll`.
            num_per_gpu, return_cupy, tdi2, window, window_alpha: As in :meth:`get_ll`.
            adjust_inplace, use_c_implementation, phase_maximize: Ignored.
            N (int or xp.ndarray, optional): Template length. Default ``None`` uses
                :func:`gbgpu.utils.utility.get_N` with ``amp = 1e-25`` and ``f0``.
            oversample (int, optional): Multiplier on the automatic ``N``. Default ``1``.

        Returns:
            np.ndarray: ``(num_bin,)`` float ``N^T M^-1 N`` per binary (an ``xp`` array with
            ``return_cupy=True``).

        Raises:
            NotImplementedError: If ``num_per_gpu`` is given.
            ValueError: If an automatic ``N`` is zero.
            KeyError: For an unknown ``window``.
            AssertionError: For index arrays that are not int32 or inconsistent shapes.

        Warning:
            With ``flip_ref_phase=True`` this method negates column 4 of the caller's
            ``params`` in place, which in this 5-column layout is ``beta``, not ``phi0``.
            A ``ValueError`` raised by the kernel drops into ``breakpoint()``.
        """
        if num_per_gpu is not None:
            raise NotImplementedError("Need to check this.")

        if self.gpus is not None:
            # set first index gpu device to control main operations
            return_to_main_device = self.gpus[0]
            self._xp_set_device(return_to_main_device)

        self.num_bin = num_bin = params.shape[0]

        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params[:, 4] = -params[:, 4]
        
        if N is None:
            # TODO: G
            #params are different for fstat
            N = get_N(self.xp.full_like(params[:, 0], 1e-25), self.xp.asarray(params[:, 0]), T, oversample=oversample)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert params.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]

        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # check that index values are ready for computation
        if data_index is None:
            data_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32
        
        nchannels = 3 if tdi_channel_setup != "AE" else 2

        if isinstance(data_minus_template, self.xp.ndarray):
            data_minus_template_in = [data_minus_template.astype(self.xp.complex128)]
        else:
            data_minus_template_in = [tmp.astype(self.xp.complex128) for tmp in data_minus_template]

        if isinstance(psd, self.xp.ndarray):
            psd_in = [psd.astype(self.xp.complex128)]
        else:
            psd_in = [p.astype(self.xp.complex128) for p in psd]

        num_data = []
        for t_i, t in enumerate(data_minus_template_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_data.append(int(t.shape[0] / (data_length * nchannels)))
                
            elif t.ndim == 2:
                num_data.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # TODO: print("check this does not create memory issues")
                data_minus_template_in[t_i] = t.flatten()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_data.append(ntemplate)
                assert _nchannels == nchannels
                # TODO: print("check this does not create memory issues")
                
                data_minus_template_in[t_i] = t.flatten()

        if tdi_channel_setup == "AE" or tdi_channel_setup == "AET":
            # assumes nchannels will 1D really
            psd_sub_shape = (nchannels,)
            
        else:
            assert nchannels == 3
            psd_sub_shape = (nchannels, nchannels)

        num_psd = []
        for t_i, t in enumerate(psd_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                
            elif t.ndim == 2:
                assert tdi_channel_setup in ["AET", "AE"]
                num_psd.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()

            elif t.ndim == 3:
                if tdi_channel_setup in ["AE", "AET"]:
                    ntemplate, _nchannels, data_length = t.shape
                    num_psd.append(ntemplate)
                    assert t.shape[1] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()


                else:  # XYZ
                    _nchannels, _nchannels, data_length = t.shape
                    num_psd.append(1)
                    assert t.shape[1] == t.shape[0] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()
                assert _nchannels == nchannels
                
            elif t.ndim == 4:
                assert tdi_channel_setup == "XYZ"
                ntemplate, _nchannels, _nchannels, data_length = t.shape
                num_psd.append(ntemplate)
                assert t.shape[1] == t.shape[2] == nchannels
                data_length = t.shape[3]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()
            
                assert _nchannels == nchannels
            # print("check this does not create memory issues")
        
        # initialize Likelihood terms <d|h> and <h|h>
        M_mat = self.xp.zeros((self.num_bin, 4, 4))
        N_arr = self.xp.zeros((self.num_bin, 4))
        
        # if self.gpus is not None:
        do_synchronize = False
        main_device = self.xp.cuda.runtime.getDevice() if hasattr(self.xp, "cuda") else -1
        # CPU mode (self.gpus is None) -> treat as a single virtual
        # device (gpu_id = -1) so the data_splits / num_per_gpu defaults +
        # the dispatch loop below run exactly once and skip every
        # cupy-specific setDevice / synchronize call.
        _gpus_iter = self.gpus if self.gpus is not None else [-1]
        if data_splits is None:
            assert len(_gpus_iter) == 1
            data_splits = self.xp.full(num_data[0], _gpus_iter[0])

        if num_per_gpu is None:
            # No batching cap: valid for ANY gpu count (the dispatch loop
            # routes rows by data_splits); only the data_splits default
            # above is inherently single-device.
            num_per_gpu = int(2**31 - 1)
            # make really high so just keeps (int32-safe: numpy 2 rejects
            # int32 arrays modulo a Python int beyond the int32 range)

        inputs_in = []
        for nnn, N_here in enumerate(unique_N):
            N_here = N_here.item()
            # get spacecraft positions (mojito: real-orbit Ps_arr threaded
            # into the kernels for mojito-orbit / TDI2 aware computation).
            tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
            tm_abs = tm_rel + self.t0_abs
            Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

            for gpu_i, gpu in enumerate(_gpus_iter):
                if self.gpus is not None:
                    self._xp_set_device(main_device)
                keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[data_index] == gpu)
                num_split_here = keep_bool.sum().item()
                inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                if num_split_here == 0:
                    continue
                if self.gpus is not None:
                    self._xp_set_device(gpu)
                
                params_here = self.xp.asarray(params)[keep_bool]
                data_index_here = (data_index[keep_bool] % num_per_gpu).astype(np.int32)
                noise_index_here = (noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                
                # theta_add = np.pi / 2 - beta_add
                params_here[:, -1] = np.pi / 2 - params_here[:, -1]
                
                data_minus_template_here = data_minus_template_in[gpu_i]
                psd_here = psd_in[gpu_i]

                params_tuple = tuple([pars_tmp.copy()for pars_tmp in params_here.T])
                
                if isinstance(start_freq_ind, int):
                    start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
                
                else:
                    assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                    # TODO: fix this num_data
                    start_freq_ind_tmp = start_freq_ind
                assert len(start_freq_ind_tmp) == num_data[gpu_i]
                    
                M_mat_temp = self.xp.zeros(num_split_here * 4 * 4, dtype=self.xp.complex128)
                N_arr_temp = self.xp.zeros(num_split_here * 4, dtype=self.xp.complex128)
                
                noise_index_in = self.xp.asarray(noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                data_index_in = self.xp.asarray(data_index[keep_bool] % num_per_gpu).astype(np.int32)
                    
                tuple_in = (
                    (
                        M_mat_temp,
                        N_arr_temp,
                        data_minus_template_here,
                        psd_here,
                        data_index_in,
                        noise_index_in,
                    ) + params_tuple
                    + (
                        T, dt, N_here, 
                        num_split_here, start_freq_ind_tmp, data_length, 
                        tdi_channel_setup_map[tdi_channel_setup], 
                        gpu, do_synchronize, 
                        num_data[0], num_psd[0],
                        Ps_arr, self.orbits.armlength, tdi2,
                        window_type, window_alpha
                    )
                )    
                
                # self.xp.cuda.runtime.deviceSynchronize()

                # for testing
                try:
                    self._xp_sync()
                    self.backend.sharedmem.SharedMemoryFstatLikeComp_wrap(*tuple_in)
                    inputs_in.append([gpu, inds_here, tuple_in])
                    self._xp_sync()
                except ValueError:
                    breakpoint()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(gpu):
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(main_device):
                self._xp_sync()
                # TODO: change inside to double instead of cmplx?
                M_mat[inds_gpu] = inputs_tmp[0][:].reshape(-1, 4, 4).real
                N_arr[inds_gpu] = inputs_tmp[1][:].reshape(-1, 4).real
                
        self._xp_set_device(main_device)
        self._xp_sync()
        
        M_mat_inv = self.xp.linalg.inv(M_mat)

        a_i = self.xp.einsum("...ij,...j->...i", M_mat_inv, N_arr)
        fstat_logl = self.xp.einsum("...i,...i->...", a_i, N_arr)
        
        a1 = a_i[:, 0]
        a2 = a_i[:, 1]
        a3 = a_i[:, 2]
        a4 = a_i[:, 3]
        A_plus = (
            self.xp.sqrt((a1 + a4) ** 2 + (a2 - a3) ** 2)
            + self.xp.sqrt((a1 - a4) ** 2 + (a2 + a3) ** 2)
        )

        A_cross = (
            self.xp.sqrt((a1 + a4) ** 2 + (a2 - a3) ** 2)
            - self.xp.sqrt((a1 - a4) ** 2 + (a2 + a3) ** 2)
        )

        self.A_max = (A_plus + self.xp.sqrt(A_plus ** 2 - A_cross ** 2)) / 2.
        self.psi_max = (0.5 * self.xp.arctan2(A_plus * a4 - A_cross * a1, -(A_cross * a2 + A_plus * a3))) % (np.pi)
        self.iota_max = self.xp.arccos(-A_cross / (A_plus + self.xp.sqrt(A_plus ** 2 - A_cross ** 2))) % (np.pi)
        c = self.xp.sign(self.xp.sin(2. * self.psi_max))
        self.phi0_max = self.xp.arctan2(c * (A_plus * a4 - A_cross * a1), -c * (A_cross * a2 + A_plus * a3)) % (2 * np.pi)
        
        if return_cupy:
            return fstat_logl

        # back to CPU if on GPU
        try:
            return fstat_logl.get()

        except AttributeError:
            return fstat_logl

    # TODO check: keep/remove/document -- broken, never computes (NameError on gpu_i); no callers
    def get_chi_sqared(
        self,
        params,
        psd,
        phase_maximize=False,
        start_freq_ind=0,
        noise_index=None,
        use_c_implementation=True,
        N: int=None,
        T=4 * YRSID_SI,
        dt=10.0,
        oversample=1,
        return_cupy=False,
        **kwargs,
    ):
        """Pairwise chi-squared between binaries (unfinished; does not compute anything).

        Intended to give ``-1/2 <h1 - h2|h1 - h2>`` for every pair of binaries with the
        ``SharedMemoryChiSquaredComp_wrap`` kernel. As written the kernel call is commented
        out, both ``start_freq_ind`` branches index ``num_data[gpu_i]`` with an undefined
        ``gpu_i`` (``NameError``), and nothing is returned. The original body is kept as
        comments below the method.

        Args:
            params (xp.ndarray): ``(num_bin, 9)`` parameters (only ``amp`` and ``f0`` are read,
                for ``get_N``).
            psd (xp.ndarray): ``(num_psd, nchannels, data_length)`` noise weights; a non-array
                raises ``NotImplementedError``.
            phase_maximize, use_c_implementation, return_cupy: Unused by the body as written.
            oversample (int, optional): Multiplier on the automatic ``N``.
            **kwargs: Unused.
            start_freq_ind (int or xp.ndarray, optional): First-bin index of ``psd``.
            noise_index (xp.ndarray, optional): ``(num_bin,)`` int32 psd rows.
            N (int, optional): Template length (``get_N`` maximum when ``None``).
            T (float, optional): Observation time [s], truncated to ``int(T / dt) * dt``.
            dt (float, optional): Sampling cadence [s].

        Returns:
            None: Nothing (the method raises ``NameError`` before finishing).
        """

        if self.gpus is not None:
            # set first index gpu device to control main operations
            self._xp_set_device(self.gpus[0])

        # get number of observation points and adjust T accordingly
        N_obs = int(T / dt)
        T = N_obs * dt

        self.num_bin = num_bin = params.shape[0]

        if N is None:
            # TODO: G
            N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample).max().item()

        # else N will be int

        df = self.df = 1. / T

        # get shape of information
        if not isinstance(psd, self.xp.ndarray):
            raise NotImplementedError

        num_data, nchannels, data_length = psd.shape
        
        if nchannels < 2:
            raise ValueError("Calculates for A and E channels.")
        elif nchannels > 2:
            warnings.warn("Only calculating A and E channels here currently.")

        df = self.df  = 1. / T
        psd = [dat.copy().flatten() for dat in psd.transpose(1, 0, 2)]

        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)

        # check that index values are ready for computation
        assert noise_index.dtype == self.xp.int32

        comparison_length = len(psd[0])

        assert noise_index.max() * data_length <= comparison_length

        if isinstance(start_freq_ind, int):
            start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
        
        else:
            assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
            # TODO: fix this num_data
            start_freq_ind_tmp = start_freq_ind
        assert len(start_freq_ind_tmp) == num_data[gpu_i]
              
        num_here = params.shape[0]

        num_comps_all = int(num_here * (num_here + 1) / 2) - num_here

        # initialize Likelihood terms <d|h> and <h|h>
        h1_h1 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)
        h2_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
        h1_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
    
    #! ======================================================================
    # TODO: check if we actually need chi_squared because it has some errors, additionally, it is not updated for Mojito orbits.
    # def get_chi_sqared(
    #     self,
    #     params,
    #     psd,
    #     phase_maximize=False,
    #     start_freq_ind=0,
    #     noise_index=None,
    #     use_c_implementation=True,
    #     N: Optional[int] = None,
    #     T=4 * YRSID_SI,
    #     dt=10.0,
    #     oversample=1,
    #     return_cupy=False,
    #     **kwargs,
    # ):
    #     """Get batched log likelihood

    #     Generate the individual log likelihood for a batched set of Galactic binaries.
    #     This is also GPU/CPU agnostic.

        gpu = self.xp.cuda.runtime.getDevice() if hasattr(self.xp, "cuda") else -1
        do_synchronize = True

    #     Raises:
    #         TypeError: If data arrays are NumPy/CuPy while template arrays are CuPy/NumPy.

    #     Returns:
    #         1D double np.ndarray: Log likelihood values associated with each binary.

    #     """

    #     if self.gpus is not None:
    #         # set first index gpu device to control main operations
    #         self.xp.cuda.runtime.setDevice(self.gpus[0])

    #     # get number of observation points and adjust T accordingly
    #     N_obs = int(T / dt)
    #     T = N_obs * dt

    #     self.num_bin = num_bin = params.shape[0]

    #     if N is None:
    #         # TODO: G
    #         N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample).max().item()

    #     # else N will be int

    #     df = self.df = 1. / T

    #     # get shape of information
    #     if not isinstance(psd, self.xp.ndarray):
    #         raise NotImplementedError

    #     num_data, nchannels, data_length = psd.shape
        
    #     if nchannels < 2:
    #         raise ValueError("Calculates for A and E channels.")
    #     elif nchannels > 2:
    #         warnings.warn("Only calculating A and E channels here currently.")

    #     psd = [dat.copy().flatten() for dat in psd.transpose(1, 0, 2)]

    #     if noise_index is None:
    #         noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)

    #     # check that index values are ready for computation
    #     assert noise_index.dtype == self.xp.int32

    #     comparison_length = len(psd[0])

    #     assert noise_index.max() * data_length <= comparison_length

    #     if isinstance(start_freq_ind, int):
    #         start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
        
    #     else:
    #         assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
    #         # TODO: fix this num_data
    #         start_freq_ind_tmp = start_freq_ind
    #     assert len(start_freq_ind_tmp) == num_data[gpu_i]
              
    #     num_here = params.shape[0]

    #     num_comps_all = int(num_here * (num_here + 1) / 2) - num_here

    #     # initialize Likelihood terms <d|h> and <h|h>
    #     h1_h1 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)
    #     h2_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
    #     h1_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
    
    #     amp, f0, fdot, fddot, phi0, iota, psi, lam, beta = [self.xp.atleast_1d(self.xp.asarray(pars_tmp.copy()))for pars_tmp in params.T]
    #     self.num_bin = num_bin = len(amp)

    #     theta = np.pi / 2 - beta

    #     gpu = self.xp.cuda.runtime.getDevice()
    #     do_synchronize = True

    #     # raise NotImplementedError
    # TODO check if T dependence is not spacecraft related
    #     self.backend.sharedmem.SharedMemoryChiSquaredComp_wrap(
    #         h1_h1,
    #         h2_h2,
    #         h1_h2,
    #         psd[0],
    #         psd[1],
    #         noise_index,
    #         amp, f0, fdot, fddot, phi0, iota, psi, lam, theta, T, dt, N, num_bin, start_freq_ind_tmp, data_length, gpu, do_synchronize, num_data, num_psd
    #     )


    #     if phase_maximize:
    #         self.non_marg_h1_h2 = h1_h2.copy()
    #         try:
    #             self.non_marg_h1_h2 = self.non_marg_h1_h2.get()
    #         except AttributeError:
    #             pass

    #         h1_h2 = self.xp.abs(h1_h2)

    #     # store these likelihood terms for later if needed
    #     self.h1_h1 = h1_h1
    #     self.h2_h2 = h2_h2
    #     self.normalized_corr = h1_h2 / np.sqrt(h1_h1 * h2_h2)

    #     # compute Likelihood
    #     chi_squared_out = -1.0 / 2.0 * (h1_h1 + h2_h2 - 2 * h1_h2).real

    #     if return_cupy:
    #         return chi_squared_out

    #     # back to CPU if on GPU
    #     try:
    #         return chi_squared_out.get()

    #     except AttributeError:
    #         return chi_squared_out
    #! ======================================================================

    
    def fill_global_template(
        self, group_index, templates, A, E, start_inds, N=None, start_freq_ind=0
    ):
        """Add already generated A / E waveforms into global template buffers.

        Takes waveforms in hand (``A``, ``E``, ``start_inds``, e.g. from :meth:`run_wave`)
        and their grouping index and adds each one into its template row with the backend
        ``fill_global`` kernel (gbgpu_utils). To build the waveforms as well, use
        :meth:`generate_global_template`.

        Args:
            group_index (xp.ndarray): ``(num_bin,)`` template row each binary is added to
                (cast to int32).
            templates (xp.ndarray): Complex buffer ``(num_templates, nchannels,
                data_length)`` on the backend, updated in place; only channels 0 and 1 (A, E)
                are written (``nchannels > 2`` warns).
            A (xp.ndarray): TDI A of each binary, ``(num_bin, N)``, or the 1D layout of
                :attr:`A_out` (``(num_bin, N).T.flatten()``).
            E (xp.ndarray): TDI E, same layout as ``A``.
            start_inds (xp.ndarray): ``(num_bin,)`` index of each waveform's first bin on the
                full ``df`` grid (:attr:`start_inds` after :meth:`run_wave`).
            N (int, optional): Waveform length; required when ``A`` / ``E`` are 1D, read
                from the shape when 2D. Default ``None``.
            start_freq_ind (int, optional): Index on the full grid of the first bin of
                ``templates``. Default ``0``.

        Raises:
            ValueError: If ``nchannels < 2``, ``A`` / ``E`` have more than 2 dimensions, or
                1D waveforms come without ``N``.
            TypeError: If ``templates`` is not an array of this backend (numpy vs cupy).
        """

        # get shape of information
        total_groups, nchannels, data_length = templates.shape
        group_index = self.xp.asarray(group_index, dtype=self.xp.int32)
        self.num_bin = num_bin = len(group_index)

        if nchannels < 2:
            raise ValueError("Calculates for A and E channels.")
        elif nchannels > 2:
            warnings.warn("Only calculating A and E channels here currently.")

        # check if arrays are of same type
        if isinstance(templates, self.xp.ndarray) is False:
            raise TypeError(
                "Make sure the data arrays are the same type as template arrays (cupy vs numpy)."
            )

        # prepare temporary buffers for C/CUDA
        # These are required to ensure the python memory order
        # is read properly in C/CUDA
        template_A = self.xp.zeros_like(
            templates[:, 0], dtype=self.xp.complex128
        ).flatten()
        template_E = self.xp.zeros_like(
            templates[:, 1], dtype=self.xp.complex128
        ).flatten()

        # shift start inds (see above)
        start_inds = (start_inds - start_freq_ind).astype(self.xp.int32)

        # check A, E, N inputs
        if A.ndim > 2 or E.ndim > 2:
            raise ValueError("A_in, E_in have maximum allowable dimension of 2.")
        elif A.ndim == 2:
            N = A.shape[1]
            assert E.ndim == 2
            # assumes the shape is the same as self.A
            A = A.T.flatten()
            E = E.T.flatten()

        elif A.ndim == 1:
            if N is None:
                raise ValueError(
                    "If providing a 1D flattened array for A and E, the N kwarg also needs to be provided."
                )

        # fill the templates in C/CUDA
        self.fill_global_func(
            template_A,
            template_E,
            A,
            E,
            start_inds,
            N,
            num_bin,
            group_index,
            data_length,
        )

        # read out to buffer arrays
        templates[:, 0] += template_A.reshape(total_groups, data_length)
        templates[:, 1] += template_E.reshape(total_groups, data_length)

    def generate_global_template(
        self,
        params,
        group_index,
        templates,
        start_freq_ind=0,
        use_c_implementation=False,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        batch_size=None,
        oversample=1,
        data_length=None,
        tdi_channel_setup="AE",
        data_splits=None,
        num_per_gpu=None,
        factors=None,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        **kwargs,
    ):
        """Generate FD templates from parameters and add them into global template buffers.

        Runs ``backend.sharedmem.SharedMemoryGenerateGlobal_wrap``, which builds each
        binary's template in the kernel and atomically adds ``factors[i] * h_i`` into row
        ``group_index[i]`` of ``templates`` (``(row * nchannels + channel) * data_length +
        bin`` layout, bins offset by ``start_freq_ind``). Binaries are grouped by template
        length ``N``. With ``gpus`` set, binaries are routed to device buffers by
        ``data_splits[group_index]`` and re-indexed to the shard's local rows; with
        ``gpus=None`` a CPU backend runs one buffer. Used by the FD branches of
        ``lisatools.globalfit.recipe`` and by ``lisatools.sources.gb.waveform.GBXYZTDIWaveform``;
        not by the WDM global fit.

        Args:
            params (xp.ndarray): ``(num_bin, 9)`` physical parameters (module docstring).
            group_index (xp.ndarray): ``(num_bin,)`` int32 template row of each binary.
            templates (xp.ndarray or list): Complex128 buffer(s) updated in place: one
                array, or one per device of ``gpus``. Each is ``(nchannels, data_length)``
                (one template), ``(num_templates, nchannels, data_length)``, or flat 1D (then
                ``data_length`` is required). Must be C-contiguous: the kernel writes into
                ``t.ravel()``, which is a copy (and the writes are lost) otherwise. A list
                argument has its entries replaced by the raveled views.
            start_freq_ind (int or xp.ndarray, optional): Index on the ``df = 1 / T`` grid of
                the first bin of the template rows: an int, or int32 ``(num_templates,)``.
                Default ``0``.
            use_c_implementation (bool, optional): Ignored.
            N (int or xp.ndarray, optional): Template length, one int or ``(num_bin,)``.
                Default ``None`` uses :func:`gbgpu.utils.utility.get_N` (stops in
                ``breakpoint()`` if a value is zero).
            T (float, optional): Observation time [s]; must be a ``float`` on the GPU branch.
                Default ``4 * YRSID_SI``.
            dt (float, optional): Sampling cadence [s]; must be a ``float`` on the GPU
                branch. Default ``10.0``.
            batch_size (int, optional): Ignored.
            oversample (int, optional): Multiplier on the automatic ``N``. Default ``1``.
            data_length (int, optional): Bins per template row; required for flat buffers.
            tdi_channel_setup (str, optional): ``"AE"`` (2 channels), ``"AET"`` or
                ``"XYZ"`` (3 channels). Default ``"AE"``.
            data_splits (xp.ndarray, optional): Device id of every template row. Default
                ``None``: one device (asserts a single entry in ``gpus``).
            num_per_gpu (int, optional): Block size of the modulo fallback of
                :meth:`_intra_split_index`; default ``None`` (asserts a single GPU on the GPU
                branch).
            factors (xp.ndarray, optional): ``(num_bin,)`` float64 multipliers. Default
                ``None`` (ones).
            tdi2: Second-generation TDI factor. Default ``False``.
            window: ``"tukey"``, ``"planck"`` or ``None``. Default ``None``.
            window_alpha: Taper parameter; ``0.0`` when ``window`` is ``None``.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: On a CuPy backend with ``gpus=None``.
            KeyError: For an unknown ``window``.
            AssertionError: For a non-int32 ``group_index``, non-float64 ``factors`` or
                inconsistent shapes.

        Warning:
            With ``flip_ref_phase=True`` the ``phi0`` column of the caller's ``params`` is
            negated in place.

        Example:
            .. code-block:: python

                import numpy as np
                from lisatools.detector import EqualArmlengthOrbits
                from lisatools.utils.constants import YRSID_SI
                from gbgpu.gbgpu import GBGPU

                gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                           force_backend="cpu")
                Tobs, dt = 1.0 * YRSID_SI, 10.0
                data_length = int(round(Tobs / dt)) // 2 + 1
                params = np.array([[2e-23, 2e-3, 7.5e-18, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
                template = np.zeros(3 * data_length, dtype=np.complex128)
                gb.generate_global_template(
                    params, np.zeros(1, dtype=np.int32), template, T=Tobs, dt=dt,
                    tdi_channel_setup="XYZ", tdi2=True, data_length=data_length,
                )
                xyz = template.reshape(3, data_length)
        """
        if self.gpus is not None:
            # set first index gpu device to control main operations
            self._xp_set_device(self.gpus[0])

        self.num_bin = num_bin = params.shape[0]

        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params[:, 4] = -params[:, 4]

        if N is None:
            # TODO: G
            N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample)
            if self.xp.any(N == 0):
                breakpoint()
        else:
            if isinstance(N, self.xp.ndarray):
                assert params.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]
        
        if factors is None:
            factors = self.xp.ones(self.num_bin, dtype=self.xp.float64)
        
        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # check that index values are ready for computation
        assert len(factors) == self.num_bin
        assert factors.dtype == self.xp.float64
        
        assert group_index.dtype == self.xp.int32
        
        nchannels = 3 if tdi_channel_setup != "AE" else 2

        if isinstance(templates, self.xp.ndarray):
            templates_in = [templates]
        else:
            templates_in = templates

        num_templates = []
        for t_i, t in enumerate(templates_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_templates.append(int(t.shape[0] / (data_length * nchannels)))
                
            elif t.ndim == 2:
                num_templates.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                templates_in[t_i] = t.ravel()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_templates.append(ntemplate)
                assert _nchannels == nchannels
                # print("check this does not create memory issues")
                
                templates_in[t_i] = t.ravel()

        if self.gpus is not None:
            do_synchronize = False
            main_device = self.xp.cuda.runtime.getDevice() if hasattr(self.xp, "cuda") else -1
            if data_splits is None:
                assert len(self.gpus) == 1
                data_splits = self.xp.full(num_templates[0], self.gpus[0])
                
            if num_per_gpu is None:
                assert len(self.gpus) == 1
                num_per_gpu = int(1e9)
                # make really high so just keeps

            inputs_in = []
            for nnn, N_here in enumerate(unique_N):
                N_here = N_here.item()
                
                # get spacecraft positions
                tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
                tm_abs = tm_rel + self.t0_abs
                Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()  
            
                for gpu_i, gpu in enumerate(self.gpus):
                    self._xp_set_device(main_device)
                    keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[group_index] == gpu)
                    num_split_here = keep_bool.sum().item()
                    if num_split_here == 0:
                        continue 
                    self._xp_set_device(gpu)
                    
                    params_here = self.xp.asarray(params)[keep_bool]
                    group_index_here = self._intra_split_index(
                        group_index[keep_bool], data_splits, gpu, num_per_gpu
                    )
                    factors_here = factors[keep_bool]

                    # theta_add = np.pi / 2 - beta_add
                    params_here[:, 8] = np.pi / 2 - params_here[:, 8]
            
                    templates_here = templates_in[gpu_i]
                    params_N_tuple = tuple([pars_tmp.copy()for pars_tmp in params_here.T])

                    if isinstance(start_freq_ind, int):
                        start_freq_ind_tmp = self.xp.full(num_templates[gpu_i], start_freq_ind, dtype=np.int32)
                    
                    else:
                        assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                        # TODO: fix this num_templates
                        start_freq_ind_tmp = start_freq_ind
                    assert len(start_freq_ind_tmp) == num_templates[gpu_i]
                    
                    assert isinstance(T, float)
                    assert isinstance(dt, float)
                    
                    tuple_in = (
                        (
                            templates_here,
                            group_index_here,
                            factors_here,
                        )
                        + params_N_tuple
                        + (
                            T, dt, N_here, 
                            num_split_here, start_freq_ind_tmp, data_length, 
                            tdi_channel_setup_map[tdi_channel_setup], 
                            gpu, do_synchronize,
                            Ps_arr, self.orbits.armlength, tdi2,
                            window_type, window_alpha
                        )
                    )

                    # for testing
                    try:
                        self._xp_sync()
                        self.backend.sharedmem.SharedMemoryGenerateGlobal_wrap(*tuple_in)
                        inputs_in.append([gpu, tuple_in])
                        self._xp_sync()
                    except ValueError:
                        breakpoint()
  
            for gpu, inputs_tmp in inputs_in:
                with self.xp.cuda.Device(gpu):
                    self._xp_sync()

            self.xp.cuda.runtime.setDevice(main_device)
            self.xp.cuda.runtime.deviceSynchronize()
            
                # else:
                #     amp, f0, fdot, fddot, phi0, iota, psi, lam, beta = [self.xp.atleast_1d(self.xp.asarray(pars_tmp.copy()))for pars_tmp in params_N.T]
                #     self.num_bin = num_bin = len(amp)

                #     theta = np.pi / 2 - beta

                #     # get shape of information
                #     if isinstance(templates, self.xp.ndarray):
                #         total_groups, nchannels, data_length = templates.shape
                #         if nchannels < 2:
                #             raise ValueError("Calculates for A and E channels.")
                #         elif nchannels > 2:
                #             warnings.warn("Only calculating A and E channels here currently.")

                #     # check if arrays are of same type
                #     else:
                #         raise_it = True
                #         if isinstance(templates, list):
                #             if isinstance(templates[0], self.xp.ndarray):
                #                 if templates[0].ndim == 1:
                #                     if data_length is not None:
                #                         raise_it = False

                #             elif isinstance(templates[0], list):
                #                 if templates[0][0].ndim == 1:
                #                     if data_length is not None:
                #                         raise_it = False

                #         if raise_it:
                #             raise TypeError(
                #                 "Make sure the data arrays are the same type as template arrays (cupy vs numpy). If providing them as lists of flattened (must be flat) arrays for memory efficiency, you must provide the data_length kwarg."
                #             )

                #         total_length = templates[0].shape[0]
                        
                #         total_groups = total_length / data_length / 2  # 2 channels A and E

                #         try:
                #             assert float(int(total_groups)) == float(total_groups) and total_groups > 0
                #         except AssertionError:
                #             breakpoint()

                #         total_groups = int(total_groups)

                #     group_index_in = self.xp.asarray(group_index_N, dtype=self.xp.int32)
                #     factors_in = self.xp.asarray(factors_N)
                    
                #     # prepare temporary buffers for C/CUDA
                #     # These are required to ensure the python memory order
                #     # is read properly in C/CUDA
                #     if isinstance(templates, list):
                #         template_A = templates[0]
                #         template_E = templates[1]
                #     else:
                #         template_A = self.xp.zeros_like(
                #             templates[:, 0], dtype=self.xp.complex128
                #         ).flatten()
                #         template_E = self.xp.zeros_like(
                #             templates[:, 1], dtype=self.xp.complex128
                #         ).flatten()
                    
                #     gpu = self.xp.cuda.runtime.getDevice()
                #     do_synchronize = True
                #     SharedMemoryGenerateGlobal_wrap(
                #         template_A,
                #         template_E,
                #         group_index_in,
                #         factors_in,
                #         amp, f0, fdot, fddot, phi0, iota, psi, lam, theta, T, dt, N_here, num_bin, start_freq_ind, data_length, gpu, do_synchronize
                #     )

                #     if isinstance(templates, self.xp.ndarray):
                #         templates[:, 0] += template_A.reshape(total_groups, data_length)
                #         templates[:, 1] += template_E.reshape(total_groups, data_length)

        elif not self.backend.uses_cupy:
            # CPU path: mirrors the GPU branch above but without any
            # cuda.runtime calls. data_splits picks which template buffer
            # entry each source contributes to; on CPU we just use a single
            # buffer (gpu id 0).
            do_synchronize = False
            gpu = 0
            if data_splits is None:
                data_splits = self.xp.full(num_templates[0], gpu)
            if num_per_gpu is None:
                num_per_gpu = int(10000000)

            for nnn, N_here in enumerate(unique_N):
                N_here = N_here.item()
                keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[group_index] == gpu)
                num_split_here = keep_bool.sum().item()
                if num_split_here == 0:
                    continue

                # spacecraft positions on the same per-N grid the GPU branch uses
                tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
                tm_abs = tm_rel + self.t0_abs
                Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

                params_here = self.xp.asarray(params)[keep_bool]
                group_index_here = self._intra_split_index(
                    group_index[keep_bool], data_splits, gpu, num_per_gpu
                )
                factors_here = factors[keep_bool]
                params_here[:, 8] = np.pi / 2 - params_here[:, 8]

                templates_here = templates_in[0]
                params_N_tuple = tuple([pars_tmp.copy() for pars_tmp in params_here.T])

                if isinstance(start_freq_ind, int):
                    start_freq_ind_tmp = self.xp.full(num_templates[0], start_freq_ind, dtype=np.int32)
                else:
                    assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                    start_freq_ind_tmp = start_freq_ind
                assert len(start_freq_ind_tmp) == num_templates[0]

                tuple_in = (
                    (templates_here, group_index_here, factors_here)
                    + params_N_tuple
                    + (T, dt, N_here, num_split_here, start_freq_ind_tmp, data_length,
                       tdi_channel_setup_map[tdi_channel_setup], gpu, do_synchronize,
                       Ps_arr, self.orbits.armlength, tdi2,
                       window_type, window_alpha)
                )
                self.backend.sharedmem.SharedMemoryGenerateGlobal_wrap(*tuple_in)

        else:
            raise NotImplementedError
            kwargs["T"] = T
            kwargs["dt"] = dt
            kwargs["N"] = N_here
            num_here = params_N.shape[0]
            # batch if needed
            if batch_size is None:
                batch_size = num_here

            inds_batch = np.arange(batch_size, num_here, batch_size)
            batches = np.split(np.arange(num_here), inds_batch)

            for batch in batches: 
                group_index_in = group_index[batch]
                params_in = params_N[batch]
                # produce TDI templates
                self.run_wave(*params_in.T, **kwargs)
                self.fill_global_template(
                    group_index_in,
                    templates,
                    self.A_out,
                    self.E_out,
                    self.start_inds,
                    self.N,
                    start_freq_ind=start_freq_ind,
                )
        return

    def swap_likelihood_difference(
        self,
        params_remove,
        params_add,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=False,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        oversample=1,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        **kwargs,
    ):
        """Log-likelihood change of swapping one source for another, batched (legacy kernel).

        ``backend.sharedmem.SharedMemorySwapLikeComp_wrap`` builds both templates per row
        and returns ``<d|h_rem>``, ``<d|h_add>``, ``<h_rem|h_rem>``, ``<h_add|h_add>`` and
        ``<h_add|h_rem>`` (inner product as in :meth:`get_ll`). With
        ``data_minus_template`` the current residual ``r`` (data minus a model that
        contains ``h_rem``), the result is the change from replacing ``h_rem`` by
        ``h_add`` in the model:

        ``-1/2 Re(<h_add|h_add> + <h_rem|h_rem> - 2 <h_add|h_rem> - 2 <r|h_add> + 2 <r|h_rem>)``.

        GPU only: requires ``gpus`` set to ONE device (the defaults of ``data_splits`` and
        ``num_per_gpu`` assert a single GPU, and ``num_per_gpu`` must be ``None``); with
        ``gpus=None`` it raises ``NotImplementedError`` (the CPU fallback after the raise
        is dead code). The LAT caller is ``gathergalaxy`` (not on the global-fit path).

        Args:
            params_remove (xp.ndarray): ``(num_bin, 9)`` parameters of the sources taken out.
            params_add (xp.ndarray): ``(num_bin, 9)`` parameters of the sources put in.
            data_minus_template, psd: As in :meth:`get_ll`.
            start_freq_ind, data_index, noise_index, T, dt, data_length: As in
                :meth:`get_ll`.
            data_splits, tdi_channel_setup, tdi2, window, window_alpha: As in :meth:`get_ll`.
            adjust_inplace, use_c_implementation (bool, optional): Ignored.
            N (int or xp.ndarray, optional): Template length. Default ``None`` uses
                :func:`gbgpu.utils.utility.get_N` on ``params_add``.
            oversample (int, optional): Multiplier on the automatic ``N``. Default ``1``.
            phase_maximize (bool, optional): Rotate the ``h_add`` terms by the phase
                ``arg(<r|h_add> + <h_add|h_rem>)`` that maximizes the result (stored on
                ``phase_angle``). Default ``False``.
            num_per_gpu (int, optional): Must be ``None``.
            **kwargs: Ignored on the GPU branch.

        Returns:
            np.ndarray: ``(num_bin,)`` float log-likelihood differences (host). The five
            complex inner products stay on the object (``d_h_remove``, ``d_h_add``,
            ``remove_remove``, ``add_add``, ``add_remove``).

        Raises:
            NotImplementedError: If ``gpus`` is ``None`` or ``num_per_gpu`` is given.
            ValueError: If an automatic ``N`` is zero.
            KeyError: For an unknown ``window``.

        Warning:
            With ``flip_ref_phase=True`` the ``phi0`` column of BOTH caller arrays is negated
            in place. A ``ValueError`` raised by the kernel drops into ``breakpoint()``.
        """
        if num_per_gpu is not None:
            raise NotImplementedError("Need to check this.")

        if self.gpus is not None:
            # set first index gpu device to control main operations
            return_to_main_device = self.gpus[0]
            self._xp_set_device(return_to_main_device)

        self.num_bin = num_bin = params_add.shape[0]
        
        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params_add[:, 4] = -params_add[:, 4]
            params_remove[:, 4] = -params_remove[:, 4]
        
        if N is None:
            # TODO: G
            N = get_N(self.xp.asarray(params_add[:, 0]), self.xp.asarray(params_add[:, 1]), T, oversample=oversample)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert params_add.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params_add.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]

        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"

        # check that index values are ready for computation
        if data_index is None:
            data_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32
        
        nchannels = 3 if tdi_channel_setup != "AE" else 2

        if isinstance(data_minus_template, self.xp.ndarray):
            data_minus_template_in = [data_minus_template.astype(self.xp.complex128)]
        else:
            data_minus_template_in = [tmp.astype(self.xp.complex128) for tmp in data_minus_template]

        if isinstance(psd, self.xp.ndarray):
            psd_in = [psd.astype(self.xp.complex128)]
        else:
            psd_in = [p.astype(self.xp.complex128) for p in psd]

        num_data = []
        for t_i, t in enumerate(data_minus_template_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_data.append(int(t.shape[0] / (data_length * nchannels)))
                
            elif t.ndim == 2:
                num_data.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # TODO: print("check this does not create memory issues")
                data_minus_template_in[t_i] = t.flatten()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_data.append(ntemplate)
                assert _nchannels == nchannels
                # TODO: print("check this does not create memory issues")
                
                data_minus_template_in[t_i] = t.flatten()

        if tdi_channel_setup == "AE" or tdi_channel_setup == "AET":
            # assumes nchannels will 1D really
            psd_sub_shape = (nchannels,)
            
        else:
            assert nchannels == 3
            psd_sub_shape = (nchannels, nchannels)

        num_psd = []
        for t_i, t in enumerate(psd_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                
            elif t.ndim == 2:
                assert tdi_channel_setup in ["AET", "AE"]
                num_psd.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()

            elif t.ndim == 3:
                if tdi_channel_setup in ["AE", "AET"]:
                    ntemplate, _nchannels, data_length = t.shape
                    num_psd.append(ntemplate)
                    assert t.shape[1] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()


                else:  # XYZ
                    _nchannels, _nchannels, data_length = t.shape
                    num_psd.append(1)
                    assert t.shape[1] == t.shape[0] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()
                assert _nchannels == nchannels
                
            elif t.ndim == 4:
                assert tdi_channel_setup == "XYZ"
                ntemplate, _nchannels, _nchannels, data_length = t.shape
                num_psd.append(ntemplate)
                assert t.shape[1] == t.shape[2] == nchannels
                data_length = t.shape[3]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()
            
                assert _nchannels == nchannels
            # print("check this does not create memory issues")
        
        # initialize Likelihood terms <d|h> and <h|h>
        d_h_remove = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        d_h_add = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        add_remove = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        remove_remove = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        add_add = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)

        if self.gpus is not None:
            do_synchronize = False
            main_device = self.xp.cuda.runtime.getDevice() if hasattr(self.xp, "cuda") else -1
            if data_splits is None:
                assert len(self.gpus) == 1
                data_splits = self.xp.full(num_data[0], self.gpus[0])
                
            if num_per_gpu is None:
                assert len(self.gpus) == 1
                num_per_gpu = int(1e9)
                # make really high so just keeps

            inputs_in = []
            for nnn, N_here in enumerate(unique_N):
                N_here = N_here.item()
                
                # get spacecraft positions
                tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
                tm_abs = tm_rel + self.t0_abs
                Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()  
                
                for gpu_i, gpu in enumerate(self.gpus):
                    self._xp_set_device(main_device)
                    keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[data_index] == gpu)
                    num_split_here = keep_bool.sum().item()
                    inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                    if num_split_here == 0:
                        continue 
                    self._xp_set_device(gpu)
                    
                    params_remove_here = self.xp.asarray(params_remove)[keep_bool]
                    params_add_here = self.xp.asarray(params_add)[keep_bool]
                    data_index_here = (data_index[keep_bool] % num_per_gpu).astype(np.int32)
                    noise_index_here = (noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                    
                    # theta_add = np.pi / 2 - beta_add
                    params_remove_here[:, 8] = np.pi / 2 - params_remove_here[:, 8]
                    params_add_here[:, 8] = np.pi / 2 - params_add_here[:, 8]
            
                    data_minus_template_here = data_minus_template_in[gpu_i]
                    psd_here = psd_in[gpu_i]

                    params_remove_tuple = tuple([pars_tmp.copy()for pars_tmp in params_remove_here.T])
                    params_add_tuple = tuple([pars_tmp.copy()for pars_tmp in params_add_here.T])

                    if isinstance(start_freq_ind, int):
                        start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
                    
                    else:
                        assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                        # TODO: fix this num_data
                        start_freq_ind_tmp = start_freq_ind
                    assert len(start_freq_ind_tmp) == num_data[gpu_i]

                    d_h_remove_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                    d_h_add_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                    add_remove_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                    remove_remove_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                    add_add_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)

                    noise_index_in = self.xp.asarray(noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                    data_index_in = self.xp.asarray(data_index[keep_bool] % num_per_gpu).astype(np.int32)
                        
                    tuple_in = (
                        (
                            d_h_remove_temp,
                            d_h_add_temp,
                            remove_remove_temp,
                            add_add_temp,
                            add_remove_temp,
                            data_minus_template_here,
                            psd_here,
                            data_index_in,
                            noise_index_in,
                        ) + params_add_tuple
                        + params_remove_tuple
                        + (
                            T, dt, N_here, 
                            num_split_here, start_freq_ind_tmp, data_length, 
                            tdi_channel_setup_map[tdi_channel_setup], 
                            gpu, do_synchronize, 
                            num_data[0], num_psd[0],
                            Ps_arr, self.orbits.armlength, tdi2,
                            window_type, window_alpha
                        )
                    )    
                  
                    # self.xp.cuda.runtime.deviceSynchronize()
                    
                    # for testing
                    try:
                        self._xp_sync()
                        self.backend.sharedmem.SharedMemorySwapLikeComp_wrap(*tuple_in)
                        inputs_in.append([gpu, inds_here, tuple_in])
                        self._xp_sync()
                    except ValueError:
                        breakpoint()
  
            for gpu, inds_gpu, inputs_tmp in inputs_in:
                with self.xp.cuda.Device(gpu):
                    self._xp_sync()

            for gpu, inds_gpu, inputs_tmp in inputs_in:
                with self.xp.cuda.Device(main_device):
                    self._xp_sync()

                    d_h_remove[inds_gpu] = inputs_tmp[0][:]
                    d_h_add[inds_gpu] = inputs_tmp[1][:]
                    remove_remove[inds_gpu] = inputs_tmp[2][:]
                    add_add[inds_gpu] = inputs_tmp[3][:]
                    add_remove[inds_gpu] = inputs_tmp[4][:]

            self.xp.cuda.runtime.setDevice(main_device)
            self.xp.cuda.runtime.deviceSynchronize()

        else:  
            raise NotImplementedError
            # add kwargs
            kwargs["T"] = T
            kwargs["dt"] = dt
            kwargs["N"] = N
            df = 1. / T
            
            # produce TDI templates
            self.run_wave(*params_remove.T, **kwargs)
            A_remove, E_remove, start_inds_remove = self.A_out.copy(), self.E_out.copy(), self.start_inds.copy()

            # produce TDI templates
            self.run_wave(*params_add.T, **kwargs)
            A_add, E_add, start_inds_add = self.A_out.copy(), self.E_out.copy(), self.start_inds.copy()

            assert A_remove.shape == A_add.shape
            assert E_remove.shape == E_add.shape
            assert start_inds_remove.shape == start_inds_add.shape

            

            # shift start inds based on the starting index of the data stream
            start_inds_temp_add = (start_inds_add - start_freq_ind).astype(self.xp.int32)
            start_inds_temp_remove = (start_inds_remove - start_freq_ind).astype(self.xp.int32)

            # get ll through C/CUDA
            self.swap_ll_diff_func(
                d_h_remove,
                d_h_add,
                add_remove,
                remove_remove,
                add_add,
                A_remove,
                E_remove,
                start_inds_temp_remove,
                A_add, 
                E_add,
                start_inds_temp_add,
                data_minus_template[0],
                data_minus_template[1],
                psd[0],
                psd[1],
                df,
                self.N,
                self.num_bin,
                data_index,
                noise_index,
                data_length,
            )

            self.A_remove = A_remove.reshape(-1, num_bin).T
            self.E_remove = E_remove.reshape(-1, num_bin).T
            self.start_inds_remove = start_inds_remove

            self.A_add = A_add.reshape(-1, num_bin).T
            self.E_add = E_add.reshape(-1, num_bin).T
            self.start_inds_add = start_inds_add

        if phase_maximize:
            self.phase_angle = self.xp.arctan2(d_h_add.imag + add_remove.imag, d_h_add.real + add_remove.real)  
            self.non_marg_d_h_add = d_h_add.copy()
            self.non_marg_add_remove = add_remove.copy()
            try:
                self.non_marg_d_h_add = self.non_marg_d_h_add.get()
                self.non_marg_add_remove = self.non_marg_add_remove.get()
            except AttributeError:
                pass

            d_h_add *= self.xp.exp(-1j * self.phase_angle)
            add_remove *= self.xp.exp(-1j * self.phase_angle)
            
        self.d_h_remove = d_h_remove
        self.d_h_add = d_h_add
        self.add_remove = add_remove
        self.remove_remove = remove_remove
        self.add_add = add_add
        
        # compute Likelihood
        ll_diff = -1 / 2 * (-2 * d_h_add + 2 * d_h_remove - 2 * add_remove + add_add + remove_remove).real
        # back to CPU if on GPU
        try:
            return ll_diff.get()

        except AttributeError:
            return ll_diff

    def inject_signal(self, *args, fmax=None, T=4.0 * YRSID_SI, dt=10.0, **kwargs):
        """Place one binary's A and E templates on a zero-filled FD grid ``[0, fmax]``.

        Runs :meth:`run_wave` and writes the first binary's A / E (Python path) into
        host arrays on the frequencies ``arange(0, fmax + df, df)``, ``df = 1 / T`` (``T``
        truncated to ``int(T / dt) * dt``), at bins ``start_inds[0]`` to
        ``start_inds[0] + N``.

        Args:
            *args: The source parameters of :meth:`run_wave` (``amp, f0, fdot, fddot, phi0,
                iota, psi, lam, beta`` plus any derived-class extras), scalars or length-1
                arrays.
            fmax (float, optional): Highest frequency of the grid [Hz]. Default ``None``
                means the Nyquist frequency ``1 / (2 dt)``.
            T (float, optional): Observation time [s]. Default ``4 * YRSID_SI``.
            dt (float, optional): Sampling cadence [s]. Default ``10.0``.
            **kwargs: Forwarded to :meth:`run_wave` (``N``, ``oversample``, ``tdi2``, ...).

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(A, E)``, complex128 host arrays of length
            ``len(arange(0, fmax + df, df))``.

        Raises:
            ValueError: If ``T`` exceeds ``orbits.t_base.max()``, or from :meth:`run_wave`.

        Warning:
            The host copy tests ``backend.name == "gpu"``, which no GBGPU backend name
            matches (they are ``gbgpu_cuda*``), so on a CUDA backend the CuPy template is
            assigned into a NumPy array and fails. Use a CPU backend.
        """

        # get binspacing
        if fmax is None:
            fmax = 1 / (2 * dt)
        
        # TODO: change to t0 since this is orbit related
        if T > self.orbits.t_base.max():
            raise ValueError(
                f"Observation time ({T}) is larger than length of time in orbital information ({self.orbits.t_base.max()})"
            )

        # adjust inputs for run wave
        N_obs = int(T / dt)
        T = N_obs * dt
        kwargs["T"] = T
        kwargs["dt"] = dt
        self.df = df = 1 / T

        # create frequencies
        f = np.arange(0.0, fmax + df, df)
        num = len(f)

        # NumPy arrays for data streams of injections
        A_out = np.zeros(num, dtype=np.complex128)
        E_out = np.zeros(num, dtype=np.complex128)

        # build the templates
        self.run_wave(*args, **kwargs)

        # add each mode to the templates
        start = self.start_inds[0]

        # if using GPU, will return to CPU
        if self.backend.name == "gpu":
            A_temp = self.A_out.squeeze().get()
            E_temp = self.E_out.squeeze().get()

        else:
            A_temp = self.A_out.squeeze()
            E_temp = self.E_out.squeeze()

        # fill the data streams at the4 proper frqeuencies
        A_out[start.item() : start.item() + self.N] = A_temp
        E_out[start.item() : start.item() + self.N] = E_temp

        return A_out, E_out

    def _apply_parameter_transforms(self, params, parameter_transforms):
        """Apply ``parameter_transforms`` in place to a parameter array (information matrix).

        Args:
            params (xp.ndarray): ``(num_params, num_bin)`` parameters (rows = parameters).
            parameter_transforms (dict): ``{index: fn}`` sets ``params[index] =
                fn(params[index])``; ``{(i, j, ...): fn}`` sets ``params[[i, j, ...]] =
                fn(params[i], params[j], ...)``.

        Returns:
            xp.ndarray: ``params`` (the same array, modified).
        """
        for ind_trans, trans in parameter_transforms.items():
            if isinstance(ind_trans, int):
                params[ind_trans] = trans(params[ind_trans])
            else:
                params[np.asarray(ind_trans)] = trans(*params[np.asarray(ind_trans)])
        return params

    def information_matrix(
        self,
        params,
        psd,
        inds=None,
        eps: float = 1e-9,
        parameter_transforms: Dict = {},
        easy_central_difference: bool = False,
        noise_index=None,
        use_c_implementation: bool = False,
        N = None,
        T: float = 4 * YRSID_SI,
        dt: float = 10.0,
        data_length=None,
        data_splits=None,
        tdi_channel_setup: str = "AE",
        num_per_gpu: Optional[int] = None,
        oversample: int = 1,
        return_cupy: bool = False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        batch_size: int = 50000,
        **kwargs
    ):
        r"""Fisher information matrices of a batch of binaries by finite differences.

        The waveform derivatives are 4th-order central differences by default,

        .. math::

            \frac{\partial h}{\partial \lambda_i} \approx
            \frac{-h(\lambda_i + 2\epsilon) + h(\lambda_i - 2\epsilon)
            + 8 [h(\lambda_i + \epsilon) - h(\lambda_i - \epsilon)]}{12 \epsilon},

        or 2nd-order ``[h(+eps) - h(-eps)] / (2 eps)`` with ``easy_central_difference``. The
        matrix is ``Gamma_ij = 4 df Re sum_k conj(dh_i[k]) psd[k] dh_j[k]`` (per channel for
        ``"AE"`` / ``"AET"``, full channel matrix for ``"XYZ"``), ``df = 1 / T``. Steps are
        taken in the coordinates of ``params``; ``parameter_transforms`` then maps each
        stepped row to the physical basis before generating. The Python path (default)
        calls :meth:`run_wave` four times per parameter (two with the 2nd-order rule), in
        source batches of ``batch_size``; ``use_c_implementation=True`` runs
        ``SharedMemoryInfoMatComp_wrap`` instead (GPU only).

        Args:
            params (xp.ndarray): ``(num_bin, num_params)`` parameters (rows = binaries; 1D
                is promoted). With no transforms these are the 9 physical parameters.
            psd (xp.ndarray or list): Inverse noise weights on the FULL ``df`` grid starting
                at bin 0: flat, reshaped to ``(num_psd, nchannels, data_length)`` (``"AE"`` /
                ``"AET"``) or ``(num_psd, 3, 3, data_length)`` (``"XYZ"``) on the Python path;
                one array or a per-device list on the C path.
            inds (xp.ndarray, optional): Parameter columns to differentiate, in output order.
                Default ``None`` (all).
            eps: Absolute step in every differentiated coordinate. Default ``1e-9``.
            parameter_transforms: ``{index: fn}`` / ``{(i, j): fn}`` map from ``params`` to the
                physical basis (see :meth:`_apply_parameter_transforms`). Default ``{}``.
            easy_central_difference: Use the 2nd-order rule. Default ``False``.
            noise_index (xp.ndarray, optional): ``(num_bin,)`` psd row of each binary.
                Default ``None`` (row 0).
            use_c_implementation: Run the shared-memory kernel (needs ``gpus`` and
                ``data_length``). Default ``False``.
            N (int or xp.ndarray, optional): Template length. Default ``None`` uses
                :func:`gbgpu.utils.utility.get_N` on the physical ``(amp, f0)``; the Python
                path then uses the batch maximum.
            T: Observation time [s]. Default ``4 * YRSID_SI``.
            dt: Sampling cadence [s]. Default ``10.0``.
            data_length (int): Bins per psd row; required.
            data_splits (xp.ndarray, optional): C path: device id per psd row.
            tdi_channel_setup: ``"AE"``, ``"AET"`` or ``"XYZ"``. Default ``"AE"``.
            num_per_gpu: C path: modulo block for the noise index. Default ``None``.
            oversample: Multiplier on the automatic ``N``. Default ``1``.
            return_cupy: Intended to keep the result on the GPU; see the Warning.
            tdi2: Second-generation TDI factor. Default ``False``.
            window: ``"tukey"``, ``"planck"`` or ``None``. Default ``None``.
            window_alpha: Taper parameter; ``0.0`` when ``window`` is ``None``.
            batch_size: Sources per batch (bounds the ``dh`` workspace). Default ``50000``.
            **kwargs: Ignored.

        Returns:
            xp.ndarray: ``(num_bin, len(inds), len(inds))`` float64 matrices, entry
            ``(a, b)`` in ``1 / (unit_a * unit_b)``.

        Raises:
            KeyError: For an unknown ``window``.
            AssertionError: On the C path without ``gpus`` or ``data_length``, or if
                ``window`` is ``None`` and ``window_alpha != 0``.

        Warning:
            The host copy tests ``backend.name == "gpu"``, which no GBGPU backend matches, so
            the result always stays on the backend (cupy on GPU) whatever ``return_cupy``.
            The psd rows are sliced from bin ``int(f0 * T) - N / 2`` (truncation) while
            :meth:`run_wave` starts the template at ``rint(f0 * T) - N / 2``. With
            ``flip_ref_phase=True`` the caller's ``phi0`` column is negated in place.
        """
        params = self.xp.atleast_2d(params)
        self.num_bin = num_bin = params.shape[0]
        num_params = params.shape[1]

        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params[:, 4] = -params[:, 4]
        
        if parameter_transforms is not None:
            phys_base = self._apply_parameter_transforms(params.T.copy(), parameter_transforms).T
        else:
            phys_base = params.copy()
        
        if N is None:
            N = get_N(self.xp.asarray(phys_base[:, 0]), self.xp.asarray(phys_base[:, 1]), T, oversample=oversample)
            if self.xp.any(N == 0):
                breakpoint()
        else:
            if isinstance(N, self.xp.ndarray):
                assert phys_base.shape[0] == N.shape[0]
            elif isinstance(N, (int, self.xp.integer)):
                N = self.xp.full(phys_base.shape[0], N)
                            
        q_check = (phys_base[:, 1] * T).astype(self.xp.int32) 
        start_freq_inds = (q_check - N / 2).astype(self.xp.int32) 

        if inds is None:
            inds = self.xp.arange(num_params)
        else:
            inds = self.xp.asarray(inds, dtype=self.xp.int32)
            
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        
        num_derivs = len(inds)
        self.df = 1 / T

        # Setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except KeyError:
                raise KeyError(f"The window '{window}' is not supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"

        nchannels = 3 if tdi_channel_setup != "AE" else 2

        # cuda/c++ backend
        if use_c_implementation:
            assert self.gpus is not None and data_length is not None, f"GPU setup is {self.gpus}, {data_length=}"
            info_matrix = self.xp.zeros((num_bin, num_derivs, num_derivs), dtype=self.xp.float64)

            eps_scaled = self.xp.zeros((num_bin, num_derivs), dtype=self.xp.float64)
            if parameter_transforms:
                for i, ind in enumerate(inds):
                    p_up = params.copy()
                    p_up[:, ind] += eps
                    p_down = params.copy()
                    p_down[:, ind] -= eps
                    
                    phys_up = self._apply_parameter_transforms(p_up.T.copy(), parameter_transforms).T
                    phys_down = self._apply_parameter_transforms(p_down.T.copy(), parameter_transforms).T
                    
                    phys_up[:, 8] = np.pi / 2.0 - phys_up[:, 8]
                    phys_down[:, 8] = np.pi / 2.0 - phys_down[:, 8]
                        
                    eps_scaled[:, i] = (phys_up[:, ind] - phys_down[:, ind]) / (2.0 * eps)
            else:
                eps_scaled = self.xp.full((num_bin, num_derivs), eps, dtype=self.xp.float64)
                if 8 in inds:
                    idx_8 = list(inds).index(8)
                    eps_scaled[:, idx_8] = -eps  # Theta = pi/2 - Beta logic

            phys_base[:, 8] = np.pi / 2.0 - phys_base[:, 8]
            
            if isinstance(psd, self.xp.ndarray):
                psd_in = [psd.astype(self.xp.complex128)]
            else:
                psd_in = [p.astype(self.xp.complex128) for p in psd]

            psd_sub_shape = (nchannels,) if tdi_channel_setup in ["AE", "AET"] else (nchannels, nchannels)
            num_psd = []
            for t_i, t in enumerate(psd_in):
                if t.ndim == 1:
                    num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                else:
                    if t.ndim == 2: num_psd.append(1); data_length = t.shape[1]
                    elif t.ndim == 3: num_psd.append(t.shape[0] if tdi_channel_setup in ["AE", "AET"] else 1); data_length = t.shape[2]
                    elif t.ndim == 4: num_psd.append(t.shape[0]); data_length = t.shape[3]
                    psd_in[t_i] = t.flatten()

            unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
            N_groups = self.xp.arange(len(unique_N))[inverse]
            
            do_synchronize = False
            main_device = self.xp.cuda.runtime.getDevice()
            
            if data_splits is None:
                data_splits = self.xp.full(num_bin, self.gpus[0])
            if num_per_gpu is None:
                num_per_gpu = int(1e9)

            inputs_in = []
            for nnn, N_here in enumerate(unique_N):
                N_here = N_here.item()
                tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
                tm_abs = tm_rel + self.t0_abs
                Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()  
                
                for gpu_i, gpu in enumerate(self.gpus):
                    self.xp.cuda.runtime.setDevice(main_device)
                    keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[noise_index] == gpu)
                    num_split_total = keep_bool.sum().item()
                    
                    if num_split_total == 0: continue 
                    
                    batches = self.xp.arange(0, num_split_total, batch_size, dtype=self.xp.int32)
                    if batches[-1] < num_split_total:
                        batches = self.xp.concatenate([batches, self.xp.array([num_split_total], dtype=self.xp.int32)])

                    full_inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                    psd_here = psd_in[gpu_i]

                    for start, end in zip(batches[:-1], batches[1:]):
                        num_split_here = int(end - start)
                        self.xp.cuda.runtime.setDevice(gpu)
                        
                        inds_here = full_inds_here[start:end]
                        params_here = phys_base[inds_here]
                        params_tuple = tuple([pars_tmp.copy() for pars_tmp in params_here.T])
                        
                        start_freq_ind_tmp = start_freq_inds[inds_here]
                        noise_index_in = self.xp.asarray(noise_index[inds_here] % num_per_gpu).astype(np.int32)
                        eps_scaled_here = eps_scaled[inds_here].flatten()
                        
                        info_mat_temp = self.xp.zeros(num_split_here * num_derivs * num_derivs, dtype=self.xp.float64)
                        d_dh_workspace = self.xp.zeros(num_split_here * num_derivs * 3 * N_here, dtype=self.xp.complex128)
                
                        tuple_in = (
                            (info_mat_temp, d_dh_workspace, psd_here, noise_index_in, inds)
                            + params_tuple 
                            + (eps_scaled_here, eps, T, dt, N_here, num_split_here, num_derivs, 
                               start_freq_ind_tmp, data_length, tdi_channel_setup_map[tdi_channel_setup], 
                               gpu, do_synchronize, num_psd[gpu_i], Ps_arr, self.orbits.armlength, 
                               tdi2, easy_central_difference, window_type, window_alpha)
                        )    
                        
                        self.xp.cuda.runtime.deviceSynchronize()
                        self.shared_mem_backend.SharedMemoryInfoMatComp_wrap(*tuple_in)
                        inputs_in.append([gpu, inds_here, tuple_in])
                        self.xp.cuda.runtime.deviceSynchronize()

            for gpu, inds_gpu, inputs_tmp in inputs_in:
                with self.xp.cuda.Device(main_device):
                    self.xp.cuda.runtime.deviceSynchronize()
                    info_matrix[inds_gpu] = inputs_tmp[0][:].reshape(-1, num_derivs, num_derivs)
                    
            self.xp.cuda.runtime.setDevice(main_device)
            self.xp.cuda.runtime.deviceSynchronize()

            if self.backend.name == "gpu" and not return_cupy:
                info_matrix = info_matrix.get()

            return info_matrix
        
        # python implementation
        else:
            info_matrix = self.xp.zeros((num_bin, num_derivs, num_derivs), dtype=self.xp.float64)

            if tdi_channel_setup == "XYZ":
                reshaped_psd = psd.reshape(-1, 3, 3, data_length)
            else:
                reshaped_psd = psd.reshape(-1, len(tdi_channel_setup), data_length)

            # I think this is fine for now, cannot have multiple N when setting up dh later
            N_val = int(N.max().item()) if isinstance(N, self.xp.ndarray) else int(N)

            waveform_kwargs: Dict[str, Any] = dict(
                N=N_val, T=T, dt=dt, oversample=oversample, tdi2=tdi2,
                tdi_channel_setup=tdi_channel_setup, use_c_implementation=False,
                window=window, window_alpha=window_alpha
            )
            
            # batching here to avoid memory allocation issues (dh is of size 10000x9x3x1024=4.4 GBytes)
            batches = self.xp.arange(0, num_bin, batch_size, dtype=self.xp.int32)
            if batches[-1] < num_bin:
                batches = self.xp.concatenate([batches, self.xp.array([num_bin], dtype=self.xp.int32)])
            
            for start, end in zip(batches[:-1], batches[1:]):
                batched_params = params[start:end]
                noise_idx_batch = noise_index[start:end]
                start_idx_batch = start_freq_inds[start:end]
                num_bin_batch = int(end - start)
                
                # Get correct psd slice for each binary out of psd_linear_arr
                f_idx = start_idx_batch[:, None] + self.xp.arange(N_val)
                w_idx = self.xp.asarray(noise_idx_batch)
                
                if tdi_channel_setup == "XYZ":
                    w_idx_exp = w_idx[:, None, None, None]               
                    c1_idx = self.xp.arange(3)[None, :, None, None]       
                    c2_idx = self.xp.arange(3)[None, None, :, None]      
                    f_idx_exp = f_idx[:, None, None, :]                  
                    inv_psd = reshaped_psd[w_idx_exp, c1_idx, c2_idx, f_idx_exp]
                else: 
                    w_idx_exp = w_idx[:, None, None]
                    c_idx = self.xp.arange(nchannels)[None, :, None]
                    f_idx_exp = f_idx[:, None, :]
                    inv_psd = reshaped_psd[w_idx_exp, c_idx, f_idx_exp]
                
                dh = self.xp.zeros(
                    (num_bin_batch, num_derivs, len(tdi_channel_setup), N_val), 
                    dtype=self.xp.complex128
                )

                def get_wave(params_to_run): # wrapper function for wave_gen with parameter transforms
                    transformed_params = params_to_run.T.copy()
                    if parameter_transforms:
                        transformed_params = self._apply_parameter_transforms(
                            transformed_params, 
                            parameter_transforms
                        )
                    self.run_wave(*transformed_params, **waveform_kwargs)
                    if tdi_channel_setup == "XYZ":
                        return self.XYZf.copy()
                    elif tdi_channel_setup == "AE":
                        return self.AETf[:, :2].copy()
                    else: # "AET"
                        return self.AETf.copy()

                for i, ind in enumerate(inds):
                    # 1 eps up derivative
                    params_up_1 = batched_params.copy()
                    params_up_1[:, ind] += eps # to match sharedmem backend * batched_params[:, ind]
                    h_I_up_eps = get_wave(params_up_1)

                    # 1 eps down derivative
                    params_down_1 = batched_params.copy()
                    params_down_1[:, ind] -= eps # to match sharedmem backend * batched_params[:, ind]
                    h_I_down_eps = get_wave(params_down_1)

                    if easy_central_difference: # compute derivative and store
                        dh[:, i] = (h_I_up_eps - h_I_down_eps) / (2.0 * eps)
                        
                    else: # higher degree derivative computation
                        # 2 eps up derivative
                        params_up_2 = batched_params.copy()
                        params_up_2[:, ind] += 2.0 * eps # to match sharedmem backend * batched_params[:, ind]
                        h_I_up_2eps = get_wave(params_up_2)

                        # 2 eps down derivative
                        params_down_2 = batched_params.copy()
                        params_down_2[:, ind] -= 2.0 * eps # to match sharedmem backend * batched_params[:, ind]
                        h_I_down_2eps = get_wave(params_down_2)

                        dh[:, i] = (-h_I_up_2eps + h_I_down_2eps + 8.0 * (h_I_up_eps - h_I_down_eps)) / (12.0 * eps)

                # TODO: Possibly add a check for scenario where \pm 2\epsilon is outside of freq band?
                
                # compute Information matrix via inner products
                if tdi_channel_setup == "XYZ":
                    # dh shape = [num_bins, num_derivs, 3, N]
                    # inv_csd shape = [num_bins, 3, 3, N]
                    # b=num_bins, i=deriv1, c=chan1, k=freq_bin, d=chan2, j=deriv2
                    info_matrix[start:end] = 4.0 * self.df * self.xp.einsum(
                        "bick,bcdk,bjdk->bij", dh.conj(), inv_psd, dh
                    ).real
                    # output shape = [num_bin, num_derivs, num_derivs]
                else:
                    # similar indices as above, but inv_psd is diagonal
                    info_matrix[start:end] = 4.0 * self.df * self.xp.einsum(
                        "bick,bjck,bck->bij", dh.conj(), dh, inv_psd
                    ).real

            if self.backend.name == "gpu" and not return_cupy:
                info_matrix = info_matrix.get()

            return info_matrix
    #* Old infomatrix setup
    # def information_matrix(
    #     self,
    #     params,
    #     psd,
    #     inds=None,
    #     eps: float = 1e-9,
    #     parameter_transforms: Dict = {},
    #     easy_central_difference: bool = False,
    #     noise_index=None,
    #     adjust_inplace=False,
    #     use_c_implementation: bool = False,
    #     N = None,
    #     T: float = 4 * YRSID_SI,
    #     dt: float = 10.0,
    #     data_length=None,
    #     data_splits=None,
    #     tdi_channel_setup: str = "AE",
    #     num_per_gpu: Optional[int] = None,
    #     oversample: int = 1,
    #     return_cupy: bool = False,
    #     tdi2: bool = False,
    #     window: Optional[str] = None,
    #     window_alpha: float = 0.0,
    #     batch_size: int = 1000,
    #     **kwargs
    # ):
    #     """Get the information matrix for a batch.

    #     This function computes the Information matrix for a batch of Galactic binaries.
    #     It uses a 2nd order calculation for the derivative if ``easy_central_difference`` is ``False``:

    #     ..math:: \\frac{dh}{d\\lambda_i} = \\frac{-h(\\lambda_i + 2\\epsilon) + h(\\lambda_i - 2\\epsilon) + 8(h(\\lambda_i + \\epsilon) - h(\\lambda_i - \\epsilon))}{12\\epsilson}

    #     Otherwise, it will just calculate the derivate with a first-order central difference.
        
    #     This function calculates the Information Matrix dynamically. It internally
    #     handles memory-safe batching if running in Python, or dispatches the entire
    #     array to the C++/CUDA backend if `use_c_implementation=True`.
        
    #     Args:
    #         params (2D double np.ndarrays): Parameters of all binaries to be calculated.
    #             Shape should be (num_sources, num_params).
    #         psd (xp.ndarray): The 1D flattened linear_psd_arr from AnalysisContainerArray.
    #         noise_index (1D int np.ndarray): Specific walker indices mapped to the binaries.
    #         data_length (int): Length of the original time-domain data stream.
    #         inds (1D int np.ndarray, optional): Indices of the parameters to test.
    #         eps (double, optional): Step to take when calculating the derivative. Default is ``1e-9``.
    #         parameter_transforms (dict, optional): Dictionary containing the parameter transform functions.
    #         easy_central_difference (bool, optional): If ``True``, compute derivatives with a first-order difference.
    #         use_c_implementation (bool): If True, invokes the C++/CUDA backend.
    #         batch_size (int): Chunk size to use to prevent VRAM overflow when running the Python backend.
            
    #     Returns:
    #         3D xp.ndarray: Information Matrices for all binaries with shape: ``(num_sources, num_derivs, num_derivs)``.        
    #     """
    #     # param shape now follows the other methods of gbgpu
    #     params = self.xp.atleast_2d(params)
    #     self.num_bin = num_bin = params.shape[0]
    #     num_params = params.shape[1]

    #     if N is None:
    #         N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample)
    #         if self.xp.any(N == 0):
    #             breakpoint()
    #     else:
    #         if isinstance(N, self.xp.ndarray):
    #             assert params.shape[0] == N.shape[0]
    #         elif isinstance(N, (int, self.xp.integer)):
    #             N = self.xp.full(params.shape[0], N)

    #     if inds is None:
    #         inds = self.xp.arange(num_params)
    #     else:
    #         inds = self.xp.asarray(inds, dtype=self.xp.int32)
            
    #     if noise_index is None:
    #         noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        
    #     num_derivs = len(inds)
        
    #     q_check = (params[:, 1] / 1000.0 * T).astype(self.xp.int32) 
    #     start_freq_inds = start_freq_inds = (q_check - N / 2).astype(self.xp.int32) # needed for psd slicing
    #     self.df = 1 / T
        
    #     info_matrix = self.xp.zeros((num_bin, num_derivs, num_derivs), dtype=self.xp.float64)
        
    #     if True:
    #         assert self.gpus is not None and len(self.gpus) == 1, "GPU setup for infomatrix is incorrect"
    #         psd = psd[self.gpus[0]] # assume that we are working on one gpu for now, for multi gpu we will likely shift to c++/cuda
    #         if tdi_channel_setup == "XYZ":
    #             reshaped_psd = psd.reshape(-1, 3, 3, data_length)
    #         else:
    #             reshaped_psd = psd.reshape(-1, len(tdi_channel_setup), data_length)
            
    #         # I think this is fine for now, cannot have multiple N when setting up dh later   
    #         N_val = int(N.max().item()) if isinstance(N, self.xp.ndarray) else int(N)
    #         info_matrix = self.xp.zeros((num_bin, num_derivs, num_derivs), dtype=self.xp.float64)

    #         waveform_kwargs: Dict[str, Any] = dict(
    #                 N=N_val,T=T, dt=dt, oversample=oversample, tdi2=tdi2,
    #                 tdi_channel_setup=tdi_channel_setup, use_c_implementation=use_c_implementation,
    #                 window=window, window_alpha=window_alpha
    #             )
            
    #         # batching here to avoid memory allocation issues (dh is of size 10000x9x3x1024=4.4 GBytes)
    #         batches = self.xp.arange(0, num_bin, batch_size, dtype=self.xp.int32)
    #         if batches[-1] < num_bin:
    #             batches = self.xp.concatenate([batches, self.xp.array([num_bin], dtype=self.xp.int32)])
            
    #         for start, end in zip(batches[:-1], batches[1:]):
    #             batched_params = params[start:end]
    #             noise_idx_batch = noise_index[start:end]
    #             start_idx_batch = start_freq_inds[start:end]
    #             num_bin_batch = int(end - start)
                
    #             # Get correct psd slice for each binary out of psd_linear_arr
    #             f_idx = start_idx_batch[:, None] + self.xp.arange(N_val)
    #             w_idx = self.xp.asarray(noise_idx_batch)
                
    #             if tdi_channel_setup == "XYZ":
    #                 w_idx_exp = w_idx[:, None, None, None]               
    #                 c1_idx = self.xp.arange(3)[None, :, None, None]       
    #                 c2_idx = self.xp.arange(3)[None, None, :, None]      
    #                 f_idx_exp = f_idx[:, None, None, :]                  
    #                 inv_csd = reshaped_psd[w_idx_exp, c1_idx, c2_idx, f_idx_exp]
    #             else: # csd is assumed to be diagonal
    #                 w_idx_exp = w_idx[:, None, None]
    #                 c_idx = self.xp.arange(psd.shape[1])[None, :, None]
    #                 f_idx_exp = f_idx[:, None, :]
    #                 inv_psd = reshaped_psd[w_idx_exp, c_idx, f_idx_exp]
                
    #             dh = self.xp.zeros(
    #                 (num_bin_batch, num_derivs, len(tdi_channel_setup), N_val), 
    #                 dtype=self.xp.complex128
    #             )

    #             def get_wave(params_to_run): # seperate function for wave_gen since it is used often
    #                 transformed_params = params_to_run.T.copy()
    #                 if parameter_transforms:
    #                     transformed_params = self._apply_parameter_transforms(
    #                         transformed_params, 
    #                         parameter_transforms
    #                     )
    #                 self.run_wave(*transformed_params, **waveform_kwargs)
    #                 wave_out = self.XYZf.copy() if tdi_channel_setup == "XYZ" else self.AETf.copy()
    #                 return wave_out

    #             for i, ind in enumerate(inds):
    #                 # 1 eps up derivative
    #                 params_up_1 = batched_params.copy()
    #                 params_up_1[:, ind] += eps * batched_params[:, ind]
    #                 h_I_up_eps = get_wave(params_up_1)

    #                 # 1 eps down derivative
    #                 params_down_1 = batched_params.copy()
    #                 params_down_1[:, ind] -= eps * batched_params[:, ind]
    #                 h_I_down_eps = get_wave(params_down_1)

    #                 if easy_central_difference: # compute derivative and store
    #                     dh[:, i] = (h_I_up_eps - h_I_down_eps) / (2.0 * eps)
                        
    #                 else: # higher degree derivative computation
    #                     # 2 eps up derivative
    #                     params_up_2 = batched_params.copy()
    #                     params_up_2[:, ind] += 2.0 * eps * batched_params[:, ind]
    #                     h_I_up_2eps = get_wave(params_up_2)

    #                     # 2 eps down derivative
    #                     params_down_2 = batched_params.copy()
    #                     params_down_2[:, ind] -= 2.0 * eps * batched_params[:, ind]
    #                     h_I_down_2eps = get_wave(params_down_2)

    #                     dh[:, i] = (-h_I_up_2eps + h_I_down_2eps + 8.0 * (h_I_up_eps - h_I_down_eps)) / (12.0 * eps)

    #             # TODO: Possibly add a check for scenario where \pm 2\epsilon is outside of freq band?
                
    #             # compute Information matrix via inner products
    #             if tdi_channel_setup == "XYZ":
    #                 # dh shape = [num_bins, num_derivs, 3, N]
    #                 # inv_csd shape = [num_bins, 3, 3, N]
    #                 # b=num_bins, i=deriv1, c=chan1, k=freq_bin, d=chan2, j=deriv2
    #                 info_matrix[start:end] = 4.0 * self.df * self.xp.einsum(
    #                     "bick,bcdk,bjdk->bij", dh.conj(), inv_csd, dh
    #                 ).real
    #                 # output shape = [num_bin, num_derivs, num_derivs]
    #             else:
    #                 # similar indices as above, but inv_psd is diagonal
    #                 info_matrix[start:end] = 4.0 * self.df * self.xp.einsum(
    #                     "bick,bjck,bck->bij", dh.conj(), dh, inv_psd
    #                 ).real

    #         if self.backend.name == "gpu" and not return_cupy:
    #             info_matrix = info_matrix.get()

    #         return info_matrix
        
    
 


class GBGPU(GBGPUBase):
    """FastGB waveforms and legacy FD likelihoods for circular galactic binaries.

    The concrete class to instantiate: :class:`GBGPUBase` with the four model hooks as
    identities (no extra parameters, ``N`` from :func:`gbgpu.utils.utility.get_N`, no
    frequency shift, no extra phase). Methods, attributes and parameter conventions are
    those of :class:`GBGPUBase` and the module docstring. To add astrophysics, subclass
    :class:`GBGPUBase` and implement the hooks (as :class:`gbgpu.thirdbody.GBGPUThirdBody`
    does).

    In the global fit, ``lisatools.globalfit.recipe`` builds one per GB / VGB branch with
    ``GBGPU(force_backend=..., orbits=general_info.gpu_orbits)`` and sets ``gpus``; on the
    WDM path it is a container read by the GB moves. The warm-start match referee
    (``match_referee.build_wave_gen``) builds one on the CPU backend with the run's ICRS
    :class:`~lisatools.detector.L1Orbits` and ``t0`` = the data start, and calls
    :meth:`run_wave`.

    Args:
        orbits: :class:`~lisatools.detector.Orbits` instance (required; ``None`` raises
            ``ValueError``).
        force_backend (str | Backend | tuple, optional): Backend flavor (``"cpu"``,
            ``"cuda13x"``, ``"cuda"``, ...), prefixed with ``gbgpu_``. See
            :class:`GBGPUBase` for the ``None`` default caveat.
        t0 (float, optional): Absolute start time [s] of the observation window. Default
            ``None`` uses ``orbits.sc_t0`` if present, else ``0.0``.
        flip_ref_phase (bool, optional): Negate ``phi0`` before generating. Default
            ``False``.

    Example:
        .. code-block:: python

            import numpy as np
            from lisatools.detector import EqualArmlengthOrbits
            from lisatools.utils.constants import YRSID_SI
            from gbgpu.gbgpu import GBGPU

            gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                       force_backend="cpu")
            amp, f0, fdot, fddot = 2e-23, 2e-3, 7.5e-18, 0.0
            phi0, iota, psi, lam, beta = 0.1, 0.2, 0.3, 0.4, 0.5
            gb.run_wave(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta,
                        T=1.0 * YRSID_SI, dt=10.0)
            X, freqs = gb.X[0], gb.freqs[0]  # (N,) X channel and frequencies [Hz]
    """

    def prepare_additional_args(self, *args):
        """Hook: convert the extra :meth:`run_wave` arguments into ``add_args``.

        A derived class takes the extra ``*args`` given to :meth:`run_wave` (after the 9
        base parameters), converts units, copies them to the backend and returns the tuple
        that :meth:`_construct_slow_part`, :meth:`shift_frequency` and
        :meth:`add_to_phasing` receive as ``*add_args``. :meth:`run_wave` only calls it
        when ``args`` is non-empty.

        Args:
            *args: The extra per-binary parameters.

        Returns:
            ValueError: :class:`GBGPU` has no extra parameters and RETURNS (does not raise)
            a ``ValueError`` instance; unpacking it later in :meth:`run_wave` raises
            ``TypeError``.
        """
        return ValueError(
            "If providing more args than the base args, must be a class derived from GBGPUBase with an adjusted 'prepare_additional_args' method."
        )

    def special_get_N(
        self,
        amp,
        f0,
        T,
        *args,
        oversample=1,
    ):
        """Hook: number of slow-part samples ``N`` per binary when :meth:`run_wave` gets none.

        Args:
            amp (float or np.ndarray): Amplitudes.
            f0 (float or np.ndarray): Frequencies [Hz].
            T (float): Observation time [s].
            *args: The extra :meth:`run_wave` arguments (unused here).
            oversample (int, optional): Multiplier on ``N``. Default ``1``.

        Returns:
            np.ndarray: ``(num_bin,)`` int32 ``N`` from :func:`gbgpu.utils.utility.get_N`
            (frequency and observation-time bands, raised for loud sources when the LAT
            sensitivity is available, times ``oversample``, capped at 2048).
        """
        return get_N(amp, f0, T, oversample=oversample)
    
    def shift_frequency(self, fi, xi, *args):
        """Hook: modify the instantaneous frequency at each spacecraft (identity here).

        Args:
            fi (xp.ndarray): Instantaneous GW frequencies [Hz] at each spacecraft,
                ``(num_bin, 3, N)``.
            xi (xp.ndarray): Delayed times at each spacecraft [s], ``(num_bin, 3, N)``.
            *args: The ``add_args`` from :meth:`prepare_additional_args`.

        Returns:
            xp.ndarray: ``fi`` unchanged.
        """
        return fi

    def add_to_phasing(self, arg_phase, f0, fdot, fddot, xi, *args):
        """Hook: add an extra phase to the link-response phasing (identity here).

        :meth:`_construct_slow_part` forms ``arg_phasing = om * kdotP - argS`` (``om = 2 pi
        f0``, ``kdotP`` the propagation delay to each spacecraft, ``argS`` the source phase)
        and passes it through this hook before building the transfer functions. (Original
        author note: the new C code still needs checking against this.)

        Args:
            arg_phase (xp.ndarray): Phasing ``(num_bin, 3, N)`` [rad].
            f0 (xp.ndarray): ``(num_bin,)`` frequencies [Hz].
            fdot (xp.ndarray): ``(num_bin,)`` frequency derivatives [Hz/s].
            fddot (xp.ndarray): ``(num_bin,)`` second derivatives [Hz/s^2].
            xi (xp.ndarray): Delayed times at each spacecraft [s], ``(num_bin, 3, N)``.
            *args: The ``add_args`` from :meth:`prepare_additional_args`.

        Returns:
            xp.ndarray: ``arg_phase`` unchanged.
        """
        return arg_phase
