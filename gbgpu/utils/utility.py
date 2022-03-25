import numpy as np
import warnings

from gbgpu.utils.constants import *

try:
    from lisatools import sensitivity as tdi

    tdi_available = True

except (ModuleNotFoundError, ImportError) as e:
    tdi_available = False
    warnings.warn("tdi module not found. No sensitivity information will be included.")


def AET(X, Y, Z):
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


def get_u(l, e):

    ######################/
    ##
    ## Invert Kepler's equation l = u - e sin(u)
    ## Using Mikkola's method (1987)
    ## referenced Tessmer & Gopakumar 2007
    ##
    ######################/

    # double u0                            ## initial guess at eccentric anomaly
    # double z, alpha, beta, s, w        ## auxiliary variables
    # double mult                        ## multiple number of 2pi

    # int neg         = 0                    // check if l is negative
    # int over2pi  = 0                    // check if over 2pi
    # int overpi     = 0                    // check if over pi but not 2pi

    # double f, f1, f2, f3, f4            // pieces of root finder
    # double u, u1, u2, u3, u4

    # enforce the mean anomaly to be in the domain -pi < l < pi
    neg = l < 0
    l = l * np.sign(l)

    over2pi = l > 2 * np.pi
    mult = np.floor(l[over2pi] / (2 * np.pi))
    l[over2pi] -= mult * 2.0 * np.pi

    overpi = l > np.pi
    l[overpi] = 2.0 * np.pi - l[overpi]

    # if (l < 0)
    # {
    #    neg = 1
    #    l   = -l
    # }
    # if (l > 2.*M_PI)
    # {
    #    over2pi = 1
    #    mult    = floor(l/(2.*M_PI))
    #    l       -= mult*2.*M_PI
    # }
    # if (l > M_PI)
    ##    overpi = 1
    #    l       = 2.*M_PI - l
    # }

    alpha = (1.0 - e) / (4.0 * e + 0.5)
    beta = 0.5 * l / (4.0 * e + 0.5)

    z = np.sqrt(beta * beta + alpha * alpha * alpha)
    z[:] = (beta - z) * (neg) + (beta + z) * (~neg)

    # if (neg == 1) z = beta - z
    # else          z = beta + z

    # to handle nan's from negative arguments
    # if (z < 0.) z = -pow(-z, 0.3333333333333333)
    # else         z =  pow( z, 0.3333333333333333)
    z[z < 0] = -((-z[z < 0]) ** (1 / 3))
    z[z >= 0] = z[z >= 0] ** (1 / 3)
    s = z - alpha / z
    w = s - 0.078 * s * s * s * s * s / (1.0 + e)

    u0 = l + e * (3.0 * w - 4.0 * w * w * w)

    # now this initial guess must be iterated once with a 4th order Newton root finder
    f = u0 - e * np.sin(u0) - l
    f1 = 1.0 - e * np.cos(u0)
    f2 = u0 - f - l
    f3 = 1.0 - f1
    f4 = -f2

    f2 *= 0.5
    f3 *= 0.166666666666667
    f4 *= 0.0416666666666667

    u1 = -f / f1
    u2 = -f / (f1 + f2 * u1)
    u3 = -f / (f1 + f2 * u2 + f3 * u2 * u2)
    u4 = -f / (f1 + f2 * u3 + f3 * u3 * u3 + f4 * u3 * u3 * u3)

    u = u0 + u4

    u[overpi] = 2.0 * np.pi - u[overpi]
    u[over2pi] = 2.0 * np.pi * mult + u[over2pi]
    u[neg] *= -1

    # if (overpi  == 1) u = 2.*M_PI - u
    # if (over2pi == 1) u = 2.*M_PI*mult + u
    # if (neg        == 1) u = -u

    return u


# get phi value for Line-of-sight velocity. See arXiv:1806.00500
def get_phi(t, T, e, n):
    u = get_u(n[:, None, None] * (t - T[:, None, None]), e[:, None, None])

    adjust = e > 1e-6  # return u

    # adjust if not circular
    beta = (1.0 - np.sqrt(1.0 - e[adjust] * e[adjust])) / e[adjust]
    u[adjust] += 2.0 * np.arctan2(
        beta[:, None, None] * np.sin(u[adjust]),
        1.0 - beta[:, None, None] * np.cos(u[adjust]),
    )
    return u


# calculate the line-of-site velocity
# see equation 13 in arXiv:1806.00500
def get_vLOS(t, A2, omegabar, e2, n2, T2):
    phi2 = get_phi(t, T2, e2, n2)
    return A2[:, None, None] * (
        np.sin(phi2 + omegabar[:, None, None])
        + e2[:, None, None] * np.sin(omegabar[:, None, None])
    )


def get_fGW(f0, fdot, fddot, T, t):
    # assuming t0 = 0.
    return (
        f0[:, None, None] + fdot[:, None, None] * t + 0.5 * fddot[:, None, None] * t * t
    )


def parab_step_ET(f0, fdot, fddot, A2, omegabar, e2, n2, T2, t0, t0_old, T):
    dtt = t0 - t0_old
    get_fGW(f0, fdot, fddot, T, t0_old)
    g1 = get_vLOS(t0_old, A2, omegabar, e2, n2, T2) * get_fGW(
        f0, fdot, fddot, T, t0_old
    )
    # g2 = get_vLOS(A2, omegabar, e2, n2, T2, (t0 + t0_old)/2.)*get_fGW(f0,  fdot,  fddot, T, (t0 + t0_old)/2.)
    g3 = get_vLOS(t0, A2, omegabar, e2, n2, T2) * get_fGW(f0, fdot, fddot, T, t0)

    # return area from trapezoidal rule
    return dtt * (g1 + g3) / 2.0 * PI2 / Clight


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


def third_body_factors(
    total_mass_inner,
    mc,
    orbit_period,
    orbit_eccentricity,
    orbit_inclination,
    orbit_Omega2,
    orbit_omega2,
    orbit_phi2,
    orbit_lambda,
    orbit_beta,
    third_mass_unit="Mjup",
    third_period_unit="years",
):

    total_mass_inner *= MSUN

    if third_mass_unit == "Mjup":
        factor = Mjup
    elif third_mass_unit == "MSUN":
        factor = MSUN

    else:
        raise NotImplementedError

    mc *= factor

    if third_period_unit == "years":
        orbit_period *= YEAR

    elif third_period_unit == "seconds":
        pass
    else:
        raise NotImplementedError

    P = orbit_period
    M = total_mass_inner
    m2 = M + mc
    iota = orbit_inclination
    Omega2 = orbit_Omega2
    omega2 = orbit_omega2
    phi2 = orbit_phi2
    theta = np.pi / 2 - orbit_beta
    phi = orbit_lambda

    a2 = (G * M * P**2 / (4 * np.pi**2)) ** (1 / 3)
    e2 = orbit_eccentricity
    p2 = a2 * (1 - e2**2)  # semilatus rectum

    # get C and S
    C = np.cos(theta) * np.sin(iota) + np.sin(theta) * np.cos(iota) * np.sin(
        phi - Omega2
    )
    S = np.sin(theta) * np.cos(phi - Omega2)

    # bar quantities
    A_bar = np.sqrt(C**2 + S**2)
    phi_bar = np.arctan(C / (-S))
    omega_bar = (omega2 + phi_bar) % (2 * np.pi)
    # check factor of 0.77
    amp2 = (mc / m2) * np.sqrt(G * m2 / p2) * A_bar

    T2 = get_T2(P, e2, phi2, third_period_unit="seconds")

    return amp2, omega_bar, T2


def get_T2(P2, e2, phi2, third_period_unit="years"):

    if third_period_unit == "years":
        P2 *= YEAR

    elif third_period_unit == "seconds":
        pass
    else:
        raise NotImplementedError

    # compute T2
    u2 = 2.0 * np.arctan(np.sqrt((1 - e2) / (1 + e2)) * np.tan(phi2 / 2.0))

    n2 = 2 * np.pi / P2

    temp_T2 = (u2 - e2 * np.sin(u2)) / n2
    T2 = (temp_T2 / YEAR) * (temp_T2 >= 0.0) + ((P2 - np.abs(temp_T2)) / YEAR) * (
        temp_T2 < 0.0
    )
    return T2


def get_aLOS(A2, omegabar, e2, P2, T2, t, eps=1e-9):

    # central differencing for derivative of velocity
    up = get_vLOS(A2, omegabar, e2, P2, T2, t + eps)
    down = get_vLOS(A2, omegabar, e2, P2, T2, t - eps)

    aLOS = (up - down) / (2 * eps)

    return aLOS


def get_f_derivatives(f0, fdot, A2, omegabar, e2, P2, T2, eps=5e4, t=None):

    if t is not None and not isinstance(t, list) and not isinstance(t, np.ndarray):
        raise ValueError("t must be 1d list or 1d np.ndarray")

    elif t is None:
        t = np.array([-eps, 0.0, eps])

    else:
        t = np.asarray(t)

    fddot = 11 / 3 * fdot**2 / f0

    A2_in = np.full_like(t, A2)
    omegabar_in = np.full_like(t, omegabar)
    e2_in = np.full_like(t, e2)
    P2_in = np.full_like(t, P2)
    T2_in = np.full_like(t, T2)
    f0_in = np.full_like(t, f0)
    fdot_in = np.full_like(t, fdot)

    f_temp = f0 + fdot * t + 0.5 * fddot * t * t
    f_temp *= 1.0 + get_vLOS(A2_in, omegabar_in, e2_in, P2_in, T2_in, t) / Clight

    fdot_new = (f_temp[2] - f_temp[0]) / (2 * eps)

    fddot_new = (f_temp[2] - 2 * f_temp[1] + f_temp[0]) / (2 * eps) ** 2

    return (f_temp[1], fdot_new, fddot_new)


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
        num_threads (int, optional):
        Number of parallel threads to use in OpenMP.
            Default is 1.
    """
    set_threads_wrap(num_threads)


def omp_get_num_threads():
    """Get global variable OMP_NUM_THREADS"""
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
