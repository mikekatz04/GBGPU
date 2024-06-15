from multiprocessing.sharedctypes import Value
import time
import warnings

import numpy as np

# import constants
from gbgpu.utils.constants import *
from gbgpu.utils.citation import *

# import for GPU if available
try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from gbgpu.gbgpu import InheritGBGPU
from gbgpu.utils.utility import *


class GBGPUThirdBody(InheritGBGPU):
    """Build the effect of a third body into Galactic binary waveforms.

    The third-body components are originally
    by Travis Robson for the paper `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_.

    Args:
        use_gpu (bool, optional): If True, run on GPUs. Default is ``False``.

    Attributes:

    """

    @property
    def citation(self):
        """Get citations for this class"""
        return zenodo + cornish_fastb + robson_triple

    def prepare_additional_args(self, A2, varpi, e2, P2, T2):
        """Prepare the arguments special to this class

        Args:
            A2 (double or 1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above.
            varpi (double or 1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (double or 1D double np.ndarray): Eccentricity of the third body orbit.
            P2 (double or 1D double np.ndarray): Period of the third body orbit in Years.
            T2 (double or 1D double np.ndarray): Time of pericenter passage of the third body in Years.
                This parameter is effectively a constant of integration.

        Returns:
            Tuple: (A2, varpi, e2, n2, T2) adjusted for GPU usage if necessary.
                (:math:`n_2=\\frac{2\\pi}{P_2}` is the angular frequency of the orbit.)

        """
        # cast to 1D if scalar
        A2 = np.atleast_1d(A2).copy()
        varpi = np.atleast_1d(varpi).copy()
        e2 = np.atleast_1d(e2).copy()
        P2 = np.atleast_1d(P2).copy()
        T2 = np.atleast_1d(T2).copy()

        self.P2 = P2

        n2 = 2 * np.pi / (P2 * YEAR)

        T2 *= YEAR

        # copy to GPU if needed
        A2 = self.xp.asarray(A2)
        varpi = self.xp.asarray(varpi)
        e2 = self.xp.asarray(e2)
        n2 = self.xp.asarray(n2)
        T2 = self.xp.asarray(T2)

        args_third = (A2, varpi, e2, n2, T2)
        return args_third

    def special_get_N(
        self,
        amp,
        f0,
        T,
        A2,
        varpi,
        e2,
        P2,
        T2,
        oversample=1,
    ):
        """Determine proper sampling rate in time domain for slow-part.

        Args:
            amp (double or 1D double np.ndarray): Amplitude parameter.
            f0 (double or 1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            T (double): Observation time in seconds.
            A2 (double or 1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above. (Not needed in this function)
            varpi (double or 1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above. (Not needed in this function)
            e2 (double or 1D double np.ndarray): Eccentricity of the third body orbit. (Not needed in this function)
            P2 (double or 1D double np.ndarray): Period of the third body orbit in Years.
            T2 (double or 1D double np.ndarray): Time of pericenter passage of the third body in Years.
                This parameter is effectively a constant of integration. (Not needed in this function)

        Returns:
            1D int32 xp.ndarray: Number of time-domain points recommended for each binary.

        Raises:
            AssertionError: Shapes of inputs are wrong.


        """

        # make sure everything is the same shape
        amp = np.atleast_1d(amp)
        f0 = np.atleast_1d(f0)
        P2 = np.atleast_1d(P2)

        assert len(amp) == len(f0)
        assert len(P2) == len(f0)

        # get level from base get_N calculator
        N = get_N(amp, f0, T, oversample=oversample)

        # check against exoplanet sampling
        P2 = np.atleast_1d(P2)

        # frequency of the sampling in /yr
        freq_N = 1 / ((T / YEAR) / N)

        # while freq_N does not reach the sampling frequency
        # necessary to resolve the third-body orbit
        if np.any(freq_N < (2.0 / P2)):
            while np.any(freq_N < (2.0 / P2)):
                inds_fix = freq_N < (2.0 / P2)

                # double sources that need fixing
                # while keeping N for sources that are fine
                N = 2 * N * (inds_fix) + N * (~inds_fix)
                freq_N = 1 / ((T / YEAR) / N)

            # reapply oversampling if needed
            N = N * oversample

        N_out = N.astype(int)
        return N_out

    def shift_frequency(self, fi, xi, A2, varpi, e2, n2, T2):
        """Shift the evolution of the frequency in the slow part

        Args:
            fi (3D double xp.ndarray): Instantaneous frequencies of the
                wave before applying third-body effect at each spacecraft as a function of time.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            xi (3D double xp.ndarray): Time at each spacecraft.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            A2 (1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above.
            varpi (1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (1D double np.ndarray): Eccentricity of the third body orbit.
            n2 (1D double np.ndarray): Angular frequency of the third body orbit in per seconds.
            T2 (1D double np.ndarray): Time of pericenter passage of the third body in seconds.
                This parameter is effectively a constant of integration.

        Returns:
            3D double xp.ndarray: Updated frequencies with third-body effect.

        """
        fi *= 1 + self.get_vLOS(xi, A2, varpi, e2, n2, T2) / Clight
        return fi

    def add_to_argS(self, argS, f0, fdot, fddot, xi, A2, varpi, e2, n2, T2):
        """Update ``argS`` in FastGB formalism for third-body effect

        ``argS`` is an effective phase that goes into ``kdotP`` in the construction
        of the slow part of the waveform. ``kdotP`` is then included directly
        in the transfer function. See :meth:`gbgpu.gbgpu.GBGPU._construct_slow_part`
        for the use of argS in the larger code.

        Args:
            argS (3D double xp.ndarray): Special phase evaluation that goes into ``kdotP``.
                Shape is ``(num binaries, 3 spacecraft, N)``.
            f0 (1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            xi (3D double xp.ndarray): Time at each spacecraft.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            T (double): Observation time in seconds.
            A2 (1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above.
            varpi (1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (1D double np.ndarray): Eccentricity of the third body orbit.
            n2 (1D double np.ndarray): Angular frequency of the third body orbit in per seconds.
            T2 (1D double np.ndarray): Time of pericenter passage of the third body in seconds.
                This parameter is effectively a constant of integration.

        Returns:
            3D double xp.ndarray: Updated ``argS`` with third-body effect

        """
        # prepare inputs to parabolic integration
        input_tuple = (
            (f0, fdot, fddot) + (A2, varpi, e2, n2, T2) + (xi[:, :, 1:], xi[:, :, :-1])
        )
        third_body_term = self.xp.zeros_like(xi)
        third_body_term[:, :, 1:] = self.xp.cumsum(
            self.parab_step_ET(*input_tuple), axis=-1
        )
        argS += third_body_term
        return argS

    def get_u(self, l, e):
        """Invert Kepler's Equation to get eccentric anomaly

        Invert Kepler's equation (:math:`l = u - e \\sin{u}`)
        using Mikkola's method (1987) referenced in Tessmer & Gopakumar 2007.

        Args:
            l (1D double xp.ndarray): Mean anomaly in radians.
            e (1D double xp.ndarray): Eccentricity.

        Returns:
            3D double xp.ndarray: Eccentric anomaly

        """

        # enforce the mean anomaly to be in the domain -pi < l < pi
        neg = l < 0
        l = l * self.xp.sign(l)

        over2pi = l > 2 * np.pi

        # multiple number of 2pi
        mult = self.xp.floor(l[over2pi] / (2 * np.pi))
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

        # auxillary variables
        alpha = (1.0 - e) / (4.0 * e + 0.5)
        beta = 0.5 * l / (4.0 * e + 0.5)

        z = self.xp.sqrt(beta * beta + alpha * alpha * alpha)
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

        # initial guess at eccentric anomaly
        u0 = l + e * (3.0 * w - 4.0 * w * w * w)

        # f,f1,f2,f3,f4,u1,u2,u3,u4 are part of root solver
        # now this initial guess must be iterated once with a 4th order Newton root finder
        f = u0 - e * self.xp.sin(u0) - l
        f1 = 1.0 - e * self.xp.cos(u0)
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

    def get_phi(self, t, T, e, n):
        """Get phi value for Line-of-sight velocity

        See `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_.

        Args:
            t (3D double xp.ndarray): Time values to evaluate :math:`\\bar{\\phi}`.
            T (1D double xp.ndarray): Time of periastron passage (``T2``) in seconds.
            e (1D double xp.ndarray): Eccentricity.
            n (1D double xp.ndarray): Angular frequency of third-body orbit (``n2``) in per seconds.

        Returns:
            3D double xp.ndarray: Phi values for line-of-sight velocities.

        """

        # get eccentric anomaly
        u = self.get_u(n[:, None, None] * (t - T[:, None, None]), e[:, None, None])

        # adjust if eccentricity is not (close to) zero
        adjust = e > 1e-6  # return u

        # adjust if not circular
        beta = (1.0 - np.sqrt(1.0 - e[adjust] * e[adjust])) / e[adjust]
        u[adjust] += 2.0 * np.arctan2(
            beta[:, None, None] * np.sin(u[adjust]),
            1.0 - beta[:, None, None] * np.cos(u[adjust]),
        )
        return u

    def get_vLOS(self, t, A2, varpi, e2, n2, T2):
        """Calculate the line-of-site velocity

        See equation 13 in `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_.

        Args:
            t (3D double xp.ndarray): Time values to evaluate :math:`\\bar{\\phi}`.
            A2 (1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above.
            varpi (1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (1D double np.ndarray): Eccentricity of the third body orbit.
            n2 (1D double np.ndarray): Angular frequency of the third body orbit in per seconds.
            T2 (1D double np.ndarray): Time of pericenter passage of the third body in seconds.
                This parameter is effectively a constant of integration.

        Returns:
            3D double xp.ndarray: LOS velocity.

        """
        # get special phi value
        phi2 = self.get_phi(t, T2, e2, n2)

        return A2[:, None, None] * (
            np.sin(phi2 + varpi[:, None, None])
            + e2[:, None, None] * np.sin(varpi[:, None, None])
        )

    def parab_step_ET(self, f0, fdot, fddot, A2, varpi, e2, n2, T2, t0, t0_old):
        """Determine phase difference caused by third-body

        Takes a step in the integration of the orbit. In this setup,
        the calculations can all be done in parallel because we are just
        inverted Kepler's equation rather than integrating an ODE where
        a serial operation is required. TODO: Check this again. Was checked in past.

        Args:
            f0 (1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            A2 (1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above.
            varpi (1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (1D double np.ndarray): Eccentricity of the third body orbit.
            n2 (1D double np.ndarray): Angular frequency of the third body orbit in per seconds.
            T2 (1D double np.ndarray): Time of pericenter passage of the third body in seconds.
                This parameter is effectively a constant of integration.
            t0 (3D double xp.ndarray): Time values at end of step.
                Shape is ``(num binaries, 3 spacecraft, N - 1)``.
            t0_old (3D double xp.ndarray): Time values at start of step.
                Shape is ``(num binaries, 3 spacecraft, N - 1)``.


        Returns:
            3D double xp.ndarray: Phase shifts due to third-body effect.


        """
        dtt = t0 - t0_old
        g1 = self.get_vLOS(t0_old, A2, varpi, e2, n2, T2) * get_fGW(
            f0, fdot, fddot, t0_old
        )
        # g2 is kept in case we go back to parabolic integration
        # g2 = get_vLOS(A2, varpi, e2, n2, T2, (t0 + t0_old)/2.)*get_fGW(f0,  fdot,  fddot, (t0 + t0_old)/2.)

        g3 = self.get_vLOS(t0, A2, varpi, e2, n2, T2) * get_fGW(f0, fdot, fddot, t0)

        # return area from trapezoidal rule
        return dtt * (g1 + g3) / 2.0 * PI2 / Clight

    def get_aLOS(self, A2, varpi, e2, P2, T2, t, eps=1e-9):
        """Get line-of-sight acceleration

        Use central difference with LOS velocity to get
        LOS acceleration.

        Args:
            A2 (1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above.
            varpi (1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (1D double np.ndarray): Eccentricity of the third body orbit.
            n2 (1D double np.ndarray): Angular frequency of the third body orbit in per seconds.
            T2 (1D double np.ndarray): Time of pericenter passage of the third body in seconds.
                This parameter is effectively a constant of integration.
            t (3D double xp.ndarray): Time values to evaluate. Shape is
                ``(num binaries, 3 spacecraft, N)``.

        Returns:
            3D double xp.ndarray: LOS acceleration

        """
        n2 = 2 * np.pi / (P2 * YEAR)
        # central differencing for derivative of velocity
        up = self.get_vLOS(A2, varpi, e2, n2, T2, t + eps)
        down = self.get_vLOS(A2, varpi, e2, n2, T2, t - eps)

        aLOS = (up - down) / (2 * eps)

        return aLOS

    def get_f_derivatives(
        self, f0, fdot, fddot, A2, varpi, e2, P2, T2, t=None, eps=5e4
    ):
        """Get instantaneous frequency derivatives in third-body waveform

        Computes the instantaneous frequency and frequency derivatives by
        calculating the effect of the third-body over the course of the orbit.

        Central difference is used for both first and second derivatives.

        Args:
            f0 (1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            A2 (1D double np.ndarray): Special amplitude parameter related to the
                    line-of-site velocity for the third body orbit as defined in the paper
                    given in the description above.
            varpi (1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (1D double np.ndarray): Eccentricity of the third body orbit.
            n2 (1D double np.ndarray): Angular frequency of the third body orbit in per seconds.
            T2 (1D double np.ndarray): Time of pericenter passage of the third body in seconds.
                This parameter is effectively a constant of integration.
            t (3D double xp.ndarray, optional): Time values for derivative calculation.
                Shape is ``(num binaries, anything, 3)``. The 3 here represents
                the times for each derivative computation. The derivatives are to be
                calucated at index 1. The step before (after) the time of the derivative
                (for central differencing) is at index 0 (1). In other words, the final dimension
                of ``t`` should be ``[t_deriv - eps, t_deriv, t_deriv + eps]``. Default is ``None``.
                If ``None``, will fill ``[t_deriv - eps, t_deriv, t_deriv + eps]`` with
                ``t_deriv = 0.0`` and ``eps`` will be the kwarg value.
            eps (double): Step size for central differencing. Only used if ``t`` is not provided.

        Returns:
            3D double xp.ndarray: Phase shifts due to third-body effect.


        """
        A2 = np.atleast_1d(A2).copy()
        varpi = np.atleast_1d(varpi).copy()
        e2 = np.atleast_1d(e2).copy()
        P2 = np.atleast_1d(P2).copy()
        n2 = 2 * np.pi / (P2 * YEAR)
        T2 = np.atleast_1d(T2).copy()
        f0 = np.atleast_1d(f0).copy()
        fdot = np.atleast_1d(fdot).copy()
        fddot = np.atleast_1d(fddot).copy()

        if t is not None and not isinstance(t, list) and not isinstance(t, np.ndarray):
            raise ValueError("t must be 1d list or 1d np.ndarray")

        elif t is None:
            t = np.tile(np.array([-eps, 0.0, eps]), (len(A2), 1, 1))

        else:
            t = np.asarray(t)

        assert t.ndim == 3
        assert t.shape[-1] == 3
        assert t.shape[0] == len(A2)

        f_temp = f0 + fdot * t + 0.5 * fddot * t * t
        f_temp *= 1.0 + self.get_vLOS(A2, varpi, e2, n2, T2, t) / Clight

        fdot_new = (f_temp[2] - f_temp[0]) / (2 * eps)

        fddot_new = (f_temp[2] - 2 * f_temp[1] + f_temp[0]) / (2 * eps) ** 2

        return (f_temp[1], fdot_new, fddot_new)


def third_body_factors(
    M,
    mc,
    P2,
    e2,
    iota,
    Omega2,
    omega2,
    phi2,
    lam,
    beta,
    third_mass_unit="Mjup",
    third_period_unit="yrs",
):
    """Get ``A2,varpi,T2`` from third-body parameters

    Get all the third-body factors that go into the waveform computation:
    ``A2``, ``varpi``, and ``T2``.

    Args:
        M (double or double np.ndarray): Total mass of inner
            Galactic binary in Solar Masses.
        mc (double or double np.ndarray): Mass of third body in units of
            ``third_mass_unit`` kwarg.
        P2 (double or double np.ndarray): Orbital period of third body
            in units of ``third_period_unit`` kwarg.
        e2 (double or double np.ndarray): Orbital eccentricity of third body.
        iota (double or double np.ndarray): Orbital inclination of third body
            in radians. This orbital inclination is one of the three euler angles describing
            the rotation of the third-body orbital frame to ecliptic frame.
        Omega2, omega2 (double or double np.ndarray): The other two of three
            euler angles describing the third-body orbital frame rotation from
            the ecliptic frame. See the Figure 1 in `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_.
        phi2 (double or double np.ndarray): Orbital phase in radians.
        lam (double or double np.ndarray): Ecliptic longitude in radians.
        beta (double or double np.ndarray): Ecliptic latitude in radians.
        third_mass_unit (str, optional): Mass unit for third-body mass. Options are
            ``"Mjup"`` for Jupiter Masses or ``"MSUN``" for Solar Masses. Default is ``"Mjup"``.
        third_period_unit (str, optional): Time unit for third-body period. Options are
            ``"sec"`` for seconds or ``"yrs``" for years. Default is ``"yrs"``.

    Returns:
        Tuple: ``(A2, varpi, T2)`` associated with the input parameters.

    """

    # adjust inner binary mass to kg
    M *= MSUN

    # adjust perturber mass to kg
    if third_mass_unit == "Mjup":
        factor = Mjup
    elif third_mass_unit == "MSUN":
        factor = MSUN

    else:
        raise NotImplementedError

    mc *= factor

    # adjust perturber period to seconds
    if third_period_unit == "yrs":
        P2 *= YEAR

    elif third_period_unit == "sec":
        pass
    else:
        raise NotImplementedError

    # total system mass
    m2 = M + mc

    # polar sky
    theta = np.pi / 2 - beta
    # aximuthal sky
    phi = lam

    # make sure shapes are the same
    assert P2.shape == M.shape
    assert P2.shape == m2.shape
    assert P2.shape == iota.shape
    assert P2.shape == Omega2.shape
    assert P2.shape == omega2.shape
    assert P2.shape == phi2.shape
    assert P2.shape == M.shape
    assert P2.shape == theta.shape
    assert P2.shape == phi.shape

    # semimajor axis
    a2 = (G * M * P2**2 / (4 * np.pi**2)) ** (1 / 3)
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

    T2 = get_T2(P2, e2, phi2, third_period_unit="sec")

    return amp2, omega_bar, T2


def get_T2(P2, e2, phi2, third_period_unit="yrs"):
    """Get ``T2`` from third-body parameters

    Args:
        P2 (double or double np.ndarray): Orbital period of third body
            in units of ``third_period_unit`` kwarg.
        e2 (double or double np.ndarray): Orbital eccentricity of third body.
        phi2 (double or double np.ndarray): Orbital phase in radians.
        third_period_unit (str, optional): Time unit for third-body period. Options are
            ``"sec"`` for seconds or ``"yrs``" for years. Default is ``"yrs"``.

    Returns:
        double xp.ndarray: ``T2`` associated with the input parameters.

    """
    # adjust perturber period to seconds
    if third_period_unit == "yrs":
        P2 *= YEAR

    elif third_period_unit == "sec":
        pass
    else:
        raise NotImplementedError

    # compute T2
    u2 = 2.0 * np.arctan(np.sqrt((1 - e2) / (1 + e2)) * np.tan(phi2 / 2.0))

    # angular frequency of orbit
    n2 = 2 * np.pi / P2

    temp_T2 = (u2 - e2 * np.sin(u2)) / n2

    # adjust for values less than zero since it is periodic of P2
    T2 = (temp_T2 / YEAR) * (temp_T2 >= 0.0) + ((P2 - np.abs(temp_T2)) / YEAR) * (
        temp_T2 < 0.0
    )
    return T2
