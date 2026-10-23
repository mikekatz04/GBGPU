"""GB astrophysics helpers and constants for GBGPU.

Re-exports every name of :mod:`lisatools.utils.constants` (``MSUN_SI``, ``G_SI``,
``C_SI``, ``PC_SI``, ``YRSID_SI``, ``fstar``, ...; values from ``lisaconstants``), and
provides:

* :func:`get_N` -- waveform length ``N`` of the FastGB slow part (used by GBGPU's
  ``run_wave`` and by the LAT global fit to size GB bands, e.g.
  ``gbbands.band_support_halfwidths`` and ``stock.erebor.gb.get_n_based_band_edges``);
* :func:`get_fdot`, :func:`get_chirp_mass_from_f_fdot`, :func:`get_amplitude`,
  :func:`get_chirp_mass`, :func:`get_eta`, :func:`get_fGW` -- quadrupole GB relations
  (used by the LAT GB transforms and F-stat tools);
* :func:`AET` and :func:`cuda_set_device`.

``tdi_available`` is True when :mod:`lisatools.sensitivity` imports (then :func:`get_N`
includes its SNR criterion); ``setDevice`` is cupy's ``setDevice`` or ``None``.
"""
from ast import Mod
import numpy as np
import warnings

from lisatools.utils.constants import *

try:
    from lisatools import sensitivity as tdi

    tdi_available = True

except (ModuleNotFoundError, ImportError) as e:
    tdi_available = False
    warnings.warn("tdi module not found. No sensitivity information will be included.")

try:
    from cupy.cuda.runtime import setDevice


except (ModuleNotFoundError, ImportError):
    setDevice = None


def AET(X, Y, Z):
    """Return the A, E, T channels from X, Y, Z.

    ``A = (Z - X) / sqrt(2)``, ``E = (X - 2Y + Z) / sqrt(6)``, ``T = (X + Y + Z) / sqrt(3)``
    (the equal-noise orthogonal combination; exact noise orthogonality needs equal arms).

    Args:
        X (xp.ndarray): ``X`` TDI channel, any domain.
        Y (xp.ndarray): ``Y`` channel, same shape.
        Z (xp.ndarray): ``Z`` channel, same shape.

    Returns:
        tuple: ``(A, E, T)`` with array shapes the same as the input ``XYZ``.

    Example:
        >>> import numpy as np
        >>> from gbgpu.utils.utility import AET
        >>> A, E, T = AET(np.ones(2), np.ones(2), np.ones(2))
        >>> A.tolist(), E.tolist()
        ([0.0, 0.0], [0.0, 0.0])
    """
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


def get_fGW(f0, fdot, fddot, t, xp=None):
    r"""Get instantaneous frequency of gravitational wave.

    Computes :math:`f(t)\approx f_0 + \dot{f}_0 t + \frac{1}{2}\ddot{f}_0 t^2.`
    This assumes the initial time is 0.

    Args:
        f0 (xp.ndarray): Initial frequency of source in Hz. Shape is ``(num_bin_all,)``.
        fdot (xp.ndarray): Initial frequency derivative of source in Hz/s. Shape is
            ``(num_bin_all,)``.
        fddot (xp.ndarray): Initial second derivative of the frequency of source in
            Hz/s^2. Shape is ``(num_bin_all,)``.
        t (xp.ndarray): Time values at which to evaluate the frequencies.
            If ``t.ndim > 1``, it will cast the frequency information properly.
            In this case, ``t.shape[0]`` must be equal to ``num_bin_all``.
        xp (module, optional): Array module (``numpy`` or ``cupy``) used for the
            broadcasting. Default ``None`` = ``numpy``.

    Returns:
        xp.ndarray: All frequencies evaluated at the given times.

    Raises:
        AssertionError: ``t.shape[0] != num_bin_all and t.shape[0] != 1``.

    Example:
        >>> import numpy as np
        >>> from gbgpu.utils.utility import get_fGW
        >>> get_fGW(np.array([0.5]), np.array([0.25]), np.array([0.0]),
        ...         np.array([0.0, 2.0])).tolist()
        [0.5, 1.0]
    """
    if xp is None:
        xp = np

    # get dimensionality of t
    tdim = t.ndim

    # adjust dimensions if needed
    dim_diff = tdim - 1
    if dim_diff > 0:
        assert (t.shape[0] == len(f0)) or (t.shape[0] == 1)

        dims_to_expand = tuple(list(np.arange(1, dim_diff + 1)))
        f0 = xp.expand_dims(f0, dims_to_expand)
        fdot = xp.expand_dims(fdot, dims_to_expand)
        fddot = xp.expand_dims(fddot, dims_to_expand)

    # calculate
    f = f0 + fdot * t + 0.5 * fddot * t * t
    return f


def get_chirp_mass(m1, m2):
    """Get chirp mass ``(m1 m2)^(3/5) / (m1 + m2)^(1/5)``.

    Args:
        m1 (xp.ndarray): Mass 1 (any mass unit).
        m2 (xp.ndarray): Mass 2 (same unit).

    Returns:
        xp.ndarray: Chirp mass, in the unit of the inputs.
    """
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def get_eta(m1, m2):
    """Get symmetric mass ratio ``eta = m1 m2 / (m1 + m2)^2``.

    Args:
        m1 (xp.ndarray): Mass 1.
        m2 (xp.ndarray): Mass 2 (same unit).

    Returns:
        xp.ndarray: Symmetric mass ratio (dimensionless, at most 0.25).
    """
    return (m1 * m2) / (m1 + m2) ** 2


def get_amplitude(m1, m2, f, d):
    r"""Get the strain amplitude of a GB.

    .. math::

        A = \frac{2 (G \mathcal{M})^{5/3}}{c^4 d} (\pi f)^{2/3},

    with :math:`\mathcal{M}` the chirp mass (:func:`get_chirp_mass`) in kg and ``d`` in
    meters.

    Args:
        m1 (xp.ndarray): Mass 1 in solar masses.
        m2 (xp.ndarray): Mass 2 in solar masses.
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        d (xp.ndarray): Luminosity distance in kpc.

    Returns:
        xp.ndarray: Amplitude (dimensionless strain).
    """
    Mc = get_chirp_mass(m1, m2) * MSUN_SI
    d = d * 1e3 * PC_SI  # kpc to meters
    A = 2 * (G_SI * Mc) ** (5.0 / 3.0) / (C_SI**4 * d) * (np.pi * f) ** (2.0 / 3.0)
    return A


def get_fdot(f, m1=None, m2=None, Mc=None):
    r"""Get the GW frequency derivative of a quadrupole-driven inspiral.

    .. math::

        \dot f = \frac{96}{5} \pi^{8/3} \left(\frac{G \mathcal{M}}{c^3}\right)^{5/3}
        f^{11/3}.

    Must provide either ``m1`` and ``m2`` or ``Mc`` (``m1`` / ``m2`` win when given).
    The inputs are not modified in place.

    Args:
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        m1 (xp.ndarray, optional): Mass 1 in solar masses. Default ``None``.
        m2 (xp.ndarray, optional): Mass 2 in solar masses. Default ``None``.
        Mc (xp.ndarray, optional): Chirp mass in solar masses. Default ``None``.

    Returns:
        xp.ndarray: fdot in Hz/s.

    Raises:
        ValueError: None of ``m1``, ``m2``, ``Mc`` is given.
        AssertionError: Only one of ``m1`` / ``m2`` is given.
    """

    # prepare inputs and convert to chirp mass if needed
    if m1 is None and m2 is None and Mc is None:
        raise ValueError("Must provide either m1 & m2 or Mc.")
    elif m1 is not None or m2 is not None:
        assert m1 is not None and m2 is not None
        Mc = get_chirp_mass(m1, m2) * MSUN_SI
    elif Mc is not None:
        # no in-place op: callers pass views of live parameter arrays
        Mc = Mc * MSUN_SI

    # calculate fdot
    fdot = (
        (96.0 / 5.0)
        * np.pi ** (8 / 3)
        * (G_SI * Mc / C_SI**3) ** (5 / 3)
        * f ** (11 / 3)
    )
    return fdot


def get_chirp_mass_from_f_fdot(f, fdot):
    """Get chirp mass from f and fdot of GW (the inverse of :func:`get_fdot`).

    A negative ``fdot`` gives ``nan`` (fractional power of a negative number).

    Args:
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        fdot (xp.ndarray): Frequency derivative of gravitational wave in Hz/s.

    Returns:
        xp.ndarray: Chirp mass in solar masses.
    """

    # backout
    Mc_SI = (
        5.0
        / 96.0
        * np.pi ** (-8 / 3)
        * (G_SI / C_SI**3) ** (-5 / 3)
        * f ** (-11 / 3)
        * fdot
    ) ** (3 / 5)
    Mc = Mc_SI / MSUN_SI
    return Mc


def get_N(amp, f0, Tobs, oversample=1):
    """Determine sampling rate for slow part of FastGB waveform.

    The base length is ``32 * mult`` with ``mult`` = 1 / 2 / 4 / 8 for ``Tobs`` up to
    1 / 2 / 4 / 8 years of ``YRSID_SI`` (8 beyond), raised to ``64 / 256 / 512 / 1024 * mult``
    for ``f0`` in ``[1, 10)`` mHz / ``[10, 30)`` mHz / ``[30, 100)`` mHz / ``>= 100`` mHz.
    When :mod:`lisatools.sensitivity` is available it is further raised to
    ``2^ceil(log2(A_cut) + 1)`` with ``A_cut = amp * sqrt(Tobs / S_m)`` and
    ``S_m = sqrt(S_X(f0)) / (4 sin^2(f0 / fstar))``, ``S_X`` the X-channel PSD of
    :class:`lisatools.sensitivity.X1TDISens` (default model; note that the code
    normalizes the square root of the PSD) -- louder sources get more points. The result
    is multiplied by ``oversample`` and capped at 2048 (the CUDA kernels' maximum).

    Called by GBGPU's ``run_wave`` when ``N`` is not given and by the LAT global fit to
    size GB bands (``gbbands.band_support_halfwidths``, ``get_n_based_band_edges``).

    Args:
        amp (xp.ndarray): Amplitude of gravitational wave (dimensionless strain).
        f0 (xp.ndarray): Frequency of gravitational wave in Hz.
        Tobs (float): Observation time in seconds.
        oversample (int, optional): Oversampling factor. This function will return
            ``oversample * N``, if N is the determined sample number.
            (Default: ``1``).

    Returns:
        np.ndarray: int32 N values for each binary entered, shape ``(num_bin,)``
        (scalars are promoted to 1-D). NumPy only: inputs are converted with
        ``np.atleast_1d``.

    Warns:
        UserWarning: When the sensitivity module is unavailable (no SNR criterion).
    """

    # make sure they are arrays
    amp = np.atleast_1d(amp)
    f0 = np.atleast_1d(f0)

    # default mult
    mult = 8

    # adjust mult based on observation time
    if (Tobs / YRSID_SI) <= 1.0:
        mult = 1

    elif (Tobs / YRSID_SI) <= 2.0:
        mult = 2

    elif (Tobs / YRSID_SI) <= 4.0:
        mult = 4

    elif (Tobs / YRSID_SI) <= 8.0:
        mult = 8

    # cast for all binaries
    mult = np.full_like(f0, mult, dtype=np.int32)

    N = 32 * mult

    # adjust based on the frequency of the source
    N[f0 >= 0.1] = 1024 * mult[f0 >= 0.1]
    N[(f0 >= 0.03) & (f0 < 0.1)] = 512 * mult[(f0 >= 0.03) & (f0 < 0.1)]
    N[(f0 >= 0.01) & (f0 < 0.03)] = 256 * mult[(f0 >= 0.01) & (f0 < 0.03)]
    N[(f0 >= 0.001) & (f0 < 0.01)] = 64 * mult[(f0 >= 0.001) & (f0 < 0.01)]

    # if a sensitivity curve is available, verify the SNR is not too high
    # if it is, needs more points
    if tdi_available:
        fonfs = f0 / fstar

        SnX = np.sqrt(tdi.X1TDISens.get_Sn(f0))

        #  calculate michelson noise
        Sm = SnX / (4.0 * np.sin(fonfs) * np.sin(fonfs))

        Acut = amp * np.sqrt(Tobs / Sm)

        exponent = np.ceil(np.log(Acut) / np.log(2.0) + 1)
        M = np.round(2.0 ** exponent).astype(int)

        N = np.maximum(M, N)
    else:
        warnings.warn(
            "Sensitivity information not available. The number of points in the waveform "
            "will not be determined by the signal strength."
        )
        
    # adjust with oversample
    N_out = (N * oversample).astype(np.int32)

    # cuda implementation relies on Nmax=2048
    Nmax = 2048
    N_out = np.minimum(N_out, Nmax)

    if np.any(N_out) == 0:
        breakpoint()

    return N_out    


def cuda_set_device(dev):
    """Globally sets CUDA device (cupy's ``setDevice``).

    Warns instead when cupy is not available.

    Args:
        dev (int): CUDA device number.
    """
    if setDevice is not None:
        setDevice(dev)
    else:
        warnings.warn("Setting cuda device, but cupy/cuda not detected.")
