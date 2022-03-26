from ast import Mod
import numpy as np
import warnings

from gbgpu.utils.constants import *

try:
    from lisatools import sensitivity as tdi

    tdi_available = True

except (ModuleNotFoundError, ImportError) as e:
    tdi_available = False
    warnings.warn("tdi module not found. No sensitivity information will be included.")

try:
    from cupy.cuda.runtime import setDevice
except ModuleNotFoundError:
    setDevice = None


def AET(X, Y, Z):
    """Return the A,E,T channels from X,Y,Z

    Args:
        X,Y,Z (xp.ndarray): Arrays holding ``XYZ`` TDI information. The signals can be in any domain.

    Returns:
        Tuple: ``(A,E,T)`` with array shapes the same as the input ``XYZ``.

    """
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


def get_fGW(f0, fdot, fddot, t):
    # assuming t0 = 0.
    return (
        f0[:, None, None] + fdot[:, None, None] * t + 0.5 * fddot[:, None, None] * t * t
    )


def get_chirp_mass(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def get_eta(m1, m2):
    return (m1 * m2) / (m1 + m2) ** 2


def get_amplitude(m1, m2, f, d):
    Mc = get_chirp_mass(m1, m2) * MSUN
    d = d * 1e3 * PC  # kpc to meters
    A = 2 * (G * Mc) ** (5.0 / 3.0) / (Clight**4 * d) * (np.pi * f) ** (2.0 / 3.0)
    return A


def get_fdot(f, m1=None, m2=None, Mc=None):

    if m1 is None and m2 is None and Mc is None:
        raise ValueError("Must provide either m1 & m2 or Mc.")
    elif m1 is not None or m2 is not None:
        assert m1 is not None and m2 is not None
        Mc = get_chirp_mass(m1, m2) * MSUN
    elif Mc is not None:
        Mc *= MSUN

    fdot = (
        (96.0 / 5.0)
        * np.pi ** (8 / 3)
        * (G * Mc / Clight**3) ** (5 / 3)
        * f ** (11 / 3)
    )
    return fdot


def get_chirp_mass_from_f_fdot(f, fdot):
    Mc_SI = (
        5.0
        / 96.0
        * np.pi ** (-8 / 3)
        * (G / Clight**3) ** (-5 / 3)
        * f ** (-11 / 3)
        * fdot
    ) ** (3 / 5)
    Mc = Mc_SI / MSUN
    return Mc


def get_N(amp, f0, Tobs, oversample=1, P2=None):
    """Determine proper sampling in time domain."""

    amp = np.atleast_1d(amp)
    f0 = np.atleast_1d(f0)

    mult = 8

    if (Tobs / YEAR) <= 1.0:
        mult = 1

    elif (Tobs / YEAR) <= 2.0:
        mult = 2

    elif (Tobs / YEAR) <= 4.0:
        mult = 4

    elif (Tobs / YEAR) <= 2.0:
        mult = 8

    mult = np.full_like(f0, mult, dtype=np.int32)

    N = 32 * mult

    N[f0 >= 0.1] = 1024 * mult[f0 >= 0.1]
    N[(f0 >= 0.03) & (f0 < 0.1)] = 512 * mult[(f0 >= 0.03) & (f0 < 0.1)]
    N[(f0 >= 0.01) & (f0 < 0.3)] = 256 * mult[(f0 >= 0.01) & (f0 < 0.3)]
    N[(f0 >= 0.001) & (f0 < 0.01)] = 64 * mult[(f0 >= 0.001) & (f0 < 0.01)]

    # TODO: add amplitude into N calculation
    if tdi_available:
        fonfs = f0 / fstar

        SnX = np.sqrt(tdi.noisepsd_X(f0))

        #  calculate michelson noise
        Sm = SnX / (4.0 * np.sin(fonfs) * np.sin(fonfs))

        Acut = amp * np.sqrt(Tobs / Sm)

        M = (2.0 ** (np.log(Acut) / np.log(2.0) + 1.0)).astype(int)

        M = M * (M > N) + N * (M < N)
        N = M * (M > N) + N * (M < N)
    else:
        warnings.warn(
            "Sensitivity information not available. The number of points in the waveform will not be determined byt the signal strength without the availability of the Sensitivity."
        )
        M = N

    M[M > 8192] = 8192

    N = M

    # check against exoplanet sampling
    if P2 is not None:
        P2 = np.atleast_1d(P2)

        freq_N = 1 / ((Tobs / YEAR) / N)
        while np.any(freq_N < (2.0 / P2)):
            inds_fix = freq_N < (2.0 / P2)
            N = 2 * N * (inds_fix) + N * (~inds_fix)
            freq_N = 1 / ((Tobs / YEAR) / N)

    # for j = 1 mode, so N/2 not N
    N_out = (N * oversample).astype(int)

    return N_out


def omp_set_num_threads(num_threads=1):
    """Globally sets OMP_NUM_THREADS
    Args:
        num_threads (int, optional): Number of parallel threads to use in OpenMP.
            Default is 1.
    """
    set_threads_wrap(num_threads)


def omp_get_num_threads():
    """Get global variable OMP_NUM_THREADS


    Returns:
        int: Number of OMP threads.

    """
    num_threads = get_threads_wrap()
    return num_threads


def cuda_set_device(dev):
    """Globally sets CUDA device

    Args:
        dev (int): CUDA device number.
    """
    if setDevice is not None:
        setDevice(dev)
    else:
        warnings.warn("Setting cuda device, but cupy/cuda not detected.")
