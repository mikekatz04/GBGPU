"""FastGB waveforms of a galactic binary with a third body on a long outer orbit.

:class:`GBGPUThirdBody` extends :class:`gbgpu.gbgpu.GBGPUBase` with the line-of-sight
motion of the inner binary around the centre of mass of a hierarchical triple (Robson et al.
2018, arXiv:1806.00500): the outer orbit Doppler-shifts the instantaneous frequency
(``shift_frequency``) and adds the integrated phase (``add_to_phasing``). Five extra
per-binary parameters follow the nine GB ones in :meth:`GBGPUThirdBody.run_wave`:
``A2`` [m/s], ``varpi`` [rad], ``e2``, ``P2`` [yr], ``T2`` [yr].
:func:`third_body_factors` and :func:`get_T2` convert physical triple parameters to them.

Public GBGPU API (docs ``user/derivedwaves.rst``, ``GBGPU_tutorial``); not used by the
global fit.

Example:
    .. code-block:: python

        import numpy as np
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.utils.constants import YRSID_SI
        from gbgpu.thirdbody import GBGPUThirdBody

        gb = GBGPUThirdBody(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                            force_backend="cpu")
        base = [2e-23, 2e-3, 7.5e-18, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        third = [400.0, 0.0, 0.3, 1.2, 0.6]  # A2, varpi, e2, P2 [yr], T2 [yr]
        params = np.tile(base + third, (10, 1))
        gb.run_wave(*params.T, T=1.0 * YRSID_SI, dt=10.0, oversample=2)
"""

from multiprocessing.sharedctypes import Value
import time
import warnings

import numpy as np

# import constants
from lisatools.utils.constants import *
from .utils.citation import *

# import for GPU if available
try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from .gbgpu import GBGPUBase
from .utils.utility import *


class GBGPUThirdBody(GBGPUBase):
    """Build the effect of a third body into Galactic binary waveforms.

    The third-body components are originally
    by Travis Robson for the paper `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_.
    The four :class:`gbgpu.gbgpu.GBGPUBase` hooks implement it: the extra
    :meth:`run_wave` arguments are ``(A2, varpi, e2, P2, T2)``
    (:meth:`prepare_additional_args`), ``N`` is doubled until the outer orbit is resolved
    (:meth:`special_get_N`), the frequency at each spacecraft is Doppler-shifted by the
    line-of-sight velocity (:meth:`shift_frequency`) and the phase gains its time integral
    (:meth:`add_to_phasing`). Only the Python path of :meth:`run_wave` includes the third
    body; the shared-memory likelihood kernels do not take the extra parameters.

    Args:
        orbits: :class:`~lisatools.detector.Orbits` instance (required), as for
            :class:`gbgpu.gbgpu.GBGPUBase`.
        force_backend (str, optional): Backend flavor (``"cpu"``, ``"cuda13x"``, ...),
            prefixed with ``gbgpu_``; see :class:`gbgpu.gbgpu.GBGPUBase`.
        t0 (float, optional): Absolute start time [s] of the observation window. Default
            ``None`` uses ``orbits.sc_t0`` if present, else ``0.0``.
        flip_ref_phase (bool, optional): Negate ``phi0`` before generating. Default
            ``False``.

    Example:
        .. code-block:: python

            from lisatools.detector import EqualArmlengthOrbits
            from gbgpu.thirdbody import GBGPUThirdBody

            gb_third = GBGPUThirdBody(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                                      force_backend="cpu")
            # 9 GB parameters, then A2 [m/s], varpi, e2, P2 [yr], T2 [yr]
            gb_third.run_wave(2e-23, 2e-3, 7.5e-18, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                              400.0, 0.0, 0.3, 1.2, 0.6, T=3.15e7, dt=10.0)
    """

    @property
    def citation(self):
        """str: BibTeX entries to cite: GBGPU (Zenodo), FastGB and Robson et al. 2018."""
        return zenodo + cornish_fastb + robson_triple

    def prepare_additional_args(self, A2, varpi, e2, P2, T2):
        r"""Convert the five third-body parameters for the slow-part hooks.

        Stores ``P2`` (years, host) on ``self.P2``.

        Args:
            A2 (float or np.ndarray): Line-of-sight velocity amplitude of the inner binary's
                orbit about the triple's centre of mass [m/s] (``A2`` of the paper).
            varpi (float or np.ndarray): Phase offset of the line-of-sight velocity [rad]
                (``omega_bar`` from :func:`third_body_factors`).
            e2 (float or np.ndarray): Eccentricity of the outer orbit.
            P2 (float or np.ndarray): Period of the outer orbit [yr].
            T2 (float or np.ndarray): Time of pericenter passage of the outer orbit [yr],
                effectively a constant of integration.

        Returns:
            tuple: ``(A2, varpi, e2, n2, T2)`` as backend arrays, with
            :math:`n_2 = 2\pi / (P_2 \, \mathrm{YRSID\_SI})` [rad/s] and ``T2`` in seconds.
        """
        # cast to 1D if scalar
        A2 = np.atleast_1d(A2).copy()
        varpi = np.atleast_1d(varpi).copy()
        e2 = np.atleast_1d(e2).copy()
        P2 = np.atleast_1d(P2).copy()
        T2 = np.atleast_1d(T2).copy()

        self.P2 = P2

        n2 = 2 * np.pi / (P2 * YRSID_SI)

        T2 *= YRSID_SI

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
        """Slow-part sample count per binary, doubled until the outer orbit is resolved.

        Starts from :func:`gbgpu.utils.utility.get_N` (which already includes
        ``oversample``) and doubles ``N`` for every binary whose sampling rate
        ``N / (T / yr)`` is below ``2 / P2`` [1/yr]. If any doubling happened, the whole
        array is then multiplied by ``oversample`` once more. The 2048 cap of ``get_N`` is
        not re-applied.

        Args:
            amp (float or np.ndarray): Amplitudes.
            f0 (float or np.ndarray): Frequencies [Hz].
            T (float): Observation time [s].
            A2 (float or np.ndarray): Unused here.
            varpi (float or np.ndarray): Unused here.
            e2 (float or np.ndarray): Unused here.
            P2 (float or np.ndarray): Outer-orbit period [yr].
            T2 (float or np.ndarray): Unused here.
            oversample (int, optional): Multiplier on ``N``. Default ``1``.

        Returns:
            np.ndarray: ``(num_bin,)`` int ``N`` per binary.

        Raises:
            AssertionError: If ``amp``, ``f0`` and ``P2`` lengths differ.
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
        freq_N = 1 / ((T / YRSID_SI) / N)

        # while freq_N does not reach the sampling frequency
        # necessary to resolve the third-body orbit
        if np.any(freq_N < (2.0 / P2)):
            while np.any(freq_N < (2.0 / P2)):
                inds_fix = freq_N < (2.0 / P2)

                # double sources that need fixing
                # while keeping N for sources that are fine
                N = 2 * N * (inds_fix) + N * (~inds_fix)
                freq_N = 1 / ((T / YRSID_SI) / N)

            # reapply oversampling if needed
            N = N * oversample

        N_out = N.astype(int)
        return N_out

    def shift_frequency(self, fi, xi, A2, varpi, e2, n2, T2):
        """Doppler-shift the instantaneous frequency: ``fi * (1 + v_LOS(xi) / c)``.

        Args:
            fi (xp.ndarray): Instantaneous GW frequencies [Hz] at each spacecraft before the
                third-body effect, ``(num_bin, 3, N)``; modified in place.
            xi (xp.ndarray): Delayed times at each spacecraft [s], ``(num_bin, 3, N)``.
            A2 (xp.ndarray): ``(num_bin,)`` line-of-sight velocity amplitude [m/s].
            varpi (xp.ndarray): ``(num_bin,)`` velocity phase offset [rad].
            e2 (xp.ndarray): ``(num_bin,)`` outer-orbit eccentricity.
            n2 (xp.ndarray): ``(num_bin,)`` outer-orbit angular frequency [rad/s].
            T2 (xp.ndarray): ``(num_bin,)`` time of pericenter passage [s].

        Returns:
            xp.ndarray: The shifted ``fi``.
        """
        fi *= 1 + self.get_vLOS(xi, A2, varpi, e2, n2, T2) / C_SI
        return fi

    def add_to_phasing(self, arg_phasing, f0, fdot, fddot, xi, A2, varpi, e2, n2, T2):
        """Add the third-body phase to the link-response phasing.

        The phase is the running trapezoid-rule integral over the delayed times ``xi`` of
        ``2 pi f_GW(t) v_LOS(t) / c`` (:meth:`parab_step_ET` per step, cumulated along the
        time axis, zero at the first sample). See
        :meth:`gbgpu.gbgpu.GBGPUBase._construct_slow_part` for where the phasing enters.

        Args:
            arg_phasing (xp.ndarray): Phasing of the base FastGB waveform,
                ``(num_bin, 3, N)`` [rad]; modified in place.
            f0, fdot, fddot (xp.ndarray): ``(num_bin,)`` frequency parameters [Hz, Hz/s,
                Hz/s^2].
            xi (xp.ndarray): Delayed times at each spacecraft [s], ``(num_bin, 3, N)``.
            A2, varpi, e2, n2, T2 (xp.ndarray): ``(num_bin,)`` outputs of
                :meth:`prepare_additional_args` (m/s, rad, -, rad/s, s).

        Returns:
            xp.ndarray: The updated ``arg_phasing``.
        """
        # prepare inputs to parabolic integration
        input_tuple = (
            (f0, fdot, fddot) + (A2, varpi, e2, n2, T2) + (xi[:, :, 1:], xi[:, :, :-1])
        )
        third_body_term = self.xp.zeros_like(xi)
        third_body_term[:, :, 1:] = self.xp.cumsum(
            self.parab_step_ET(*input_tuple), axis=-1
        )

        arg_phasing += third_body_term
        return arg_phasing

    def get_u(self, l, e):
        """Invert Kepler's Equation to get eccentric anomaly.

        Invert Kepler's equation (:math:`l = u - e \\sin{u}`)
        using Mikkola's method (1987) referenced in Tessmer & Gopakumar 2007: a cubic
        initial guess refined by one 4th-order Newton step, after folding ``l`` into
        ``[0, pi]`` (the fold is undone on ``u``).

        Args:
            l (xp.ndarray): Mean anomaly [rad]; :meth:`get_phi` passes ``(num_bin, 3, N)``.
            e (xp.ndarray): Eccentricity, broadcastable to ``l`` (e.g. ``(num_bin, 1, 1)``).

        Returns:
            xp.ndarray: Eccentric anomaly [rad], shape of ``l``.
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
        """Get phi value for Line-of-sight velocity.

        See `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_. Solves Kepler's
        equation for the mean anomaly ``n (t - T)`` (:meth:`get_u`) and, for ``e > 1e-6``,
        converts the eccentric anomaly to the true anomaly
        ``u + 2 arctan(beta sin u / (1 - beta cos u))``, ``beta = (1 - sqrt(1 - e^2)) / e``.

        Args:
            t (xp.ndarray): Times [s], ``(num_bin, 3, N)``.
            T (xp.ndarray): ``(num_bin,)`` time of periastron passage (``T2``) [s].
            e (xp.ndarray): ``(num_bin,)`` eccentricity.
            n (xp.ndarray): ``(num_bin,)`` angular frequency of the outer orbit (``n2``)
                [rad/s].

        Returns:
            xp.ndarray: Orbital phase (true anomaly) [rad], shape of ``t``.
        """

        # get eccentric anomaly
        u = self.get_u(n[:, None, None] * (t - T[:, None, None]), e[:, None, None])

        # adjust if eccentricity is not (close to) zero
        adjust = e > 1e-6  # return u

        # adjust if not circular
        if np.any(adjust):
            beta = (1.0 - np.sqrt(1.0 - e[adjust] * e[adjust])) / e[adjust]
            u[adjust] += 2.0 * np.arctan2(
                beta[:, None, None] * np.sin(u[adjust]),
                1.0 - beta[:, None, None] * np.cos(u[adjust]),
            )
            
        return u

    def get_vLOS(self, t, A2, varpi, e2, n2, T2):
        """Calculate the line-of-sight velocity ``A2 (sin(phi2 + varpi) + e2 sin(varpi))``.

        See equation 13 in `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_;
        ``phi2`` is the outer-orbit phase from :meth:`get_phi`.

        Args:
            t (xp.ndarray): Times [s], ``(num_bin, 3, N)`` (any 3D shape with ``num_bin``
                leading).
            A2 (xp.ndarray): ``(num_bin,)`` velocity amplitude [m/s].
            varpi (xp.ndarray): ``(num_bin,)`` velocity phase offset [rad].
            e2 (xp.ndarray): ``(num_bin,)`` eccentricity of the outer orbit.
            n2 (xp.ndarray): ``(num_bin,)`` angular frequency of the outer orbit [rad/s].
            T2 (xp.ndarray): ``(num_bin,)`` time of pericenter passage [s], effectively a
                constant of integration.

        Returns:
            xp.ndarray: Line-of-sight velocity [m/s], shape of ``t``.
        """
        # get special phi value
        phi2 = self.get_phi(t, T2, e2, n2)

        return A2[:, None, None] * (
            np.sin(phi2 + varpi[:, None, None])
            + e2[:, None, None] * np.sin(varpi[:, None, None])
        )
        
    def parab_step_ET(self, f0, fdot, fddot, A2, varpi, e2, n2, T2, t0, t0_old):
        """Determine phase difference caused by third-body.

        Takes a step in the integration of the orbit. In this setup,
        the calculations can all be done in parallel because we are just
        inverted Kepler's equation rather than integrating an ODE where
        a serial operation is required. TODO: Check this again. Was checked in past.
        Each step is the trapezoid rule on ``2 pi f_GW(t) v_LOS(t) / c`` between
        ``t0_old`` and ``t0``.

        Args:
            f0 (xp.ndarray): ``(num_bin,)`` frequency [Hz].
            fdot (xp.ndarray): ``(num_bin,)`` frequency derivative [Hz/s].
            fddot (xp.ndarray): ``(num_bin,)`` second frequency derivative [Hz/s^2].
            A2 (xp.ndarray): ``(num_bin,)`` line-of-sight velocity amplitude [m/s].
            varpi (xp.ndarray): ``(num_bin,)`` velocity phase offset [rad].
            e2 (xp.ndarray): ``(num_bin,)`` eccentricity of the outer orbit.
            n2 (xp.ndarray): ``(num_bin,)`` angular frequency of the outer orbit [rad/s].
            T2 (xp.ndarray): ``(num_bin,)`` time of pericenter passage [s], effectively a
                constant of integration.
            t0 (xp.ndarray): Time values at end of step [s],
                shape ``(num_bin, 3, N - 1)``.
            t0_old (xp.ndarray): Time values at start of step [s],
                shape ``(num_bin, 3, N - 1)``.

        Returns:
            xp.ndarray: Phase increments [rad] due to the third body, shape of ``t0``.
        """
        dtt = t0 - t0_old
        g1 = self.get_vLOS(t0_old, A2, varpi, e2, n2, T2) * get_fGW(
            f0, fdot, fddot, t0_old
        )
        # g2 is kept in case we go back to parabolic integration
        # g2 = get_vLOS(A2, varpi, e2, n2, T2, (t0 + t0_old)/2.)*get_fGW(f0,  fdot,  fddot, (t0 + t0_old)/2.)

        g3 = self.get_vLOS(t0, A2, varpi, e2, n2, T2) * get_fGW(f0, fdot, fddot, t0)

        # return area from trapezoidal rule
        return dtt * (g1 + g3) / 2.0 * PI * 2 / C_SI

    def get_aLOS(self, A2, varpi, e2, P2, T2, t, eps=1e-9):
        """Get line-of-sight acceleration.

        Use central difference with LOS velocity to get
        LOS acceleration: ``(v_LOS(t + eps) - v_LOS(t - eps)) / (2 eps)``.

        Args:
            A2 (xp.ndarray): ``(num_bin,)`` line-of-sight velocity amplitude [m/s].
            varpi (xp.ndarray): ``(num_bin,)`` velocity phase offset [rad].
            e2 (xp.ndarray): ``(num_bin,)`` eccentricity of the outer orbit.
            P2 (xp.ndarray): ``(num_bin,)`` outer-orbit period [yr] (converted to
                ``n2 = 2 pi / (P2 * YRSID_SI)``).
            T2 (xp.ndarray): ``(num_bin,)`` time of pericenter passage [s] (passed to
                :meth:`get_vLOS` unchanged).
            t (xp.ndarray): Times [s], ``(num_bin, 3, N)`` (any 3D shape with ``num_bin``
                leading).
            eps (float, optional): Time step [s]. Default ``1e-9``.

        Returns:
            xp.ndarray: LOS acceleration [m/s^2], shape of ``t``.
        """
        n2 = 2 * np.pi / (P2 * YRSID_SI)
        # central differencing for derivative of velocity
        up = self.get_vLOS(t + eps, A2, varpi, e2, n2, T2)
        down = self.get_vLOS(t - eps, A2, varpi, e2, n2, T2)

        aLOS = (up - down) / (2 * eps)

        return aLOS

    def get_f_derivatives(
        self, f0, fdot, fddot, A2, varpi, e2, P2, T2, t=None, eps=5e4
    ):
        """Get instantaneous frequency derivatives in third-body waveform.

        Computes the instantaneous frequency and frequency derivatives by
        calculating the effect of the third-body over the course of the orbit:
        ``f(t) = (f0 + fdot t + fddot t^2 / 2) (1 + v_LOS(t) / c)`` on three times per
        row, then central differences for the first and second derivatives.

        Args:
            f0 (float or np.ndarray): ``(num_bin,)`` frequency [Hz].
            fdot (float or np.ndarray): ``(num_bin,)`` frequency derivative [Hz/s].
            fddot (float or np.ndarray): ``(num_bin,)`` second derivative [Hz/s^2].
            A2 (float or np.ndarray): ``(num_bin,)`` line-of-sight velocity amplitude [m/s].
            varpi (float or np.ndarray): ``(num_bin,)`` velocity phase offset [rad].
            e2 (float or np.ndarray): ``(num_bin,)`` eccentricity of the outer orbit.
            P2 (float or np.ndarray): ``(num_bin,)`` outer-orbit period [yr] (converted to
                ``n2 = 2 pi / (P2 * YRSID_SI)``).
            T2 (float or np.ndarray): ``(num_bin,)`` time of pericenter passage, passed on
                unconverted.
            t (np.ndarray or list, optional): Time values for derivative calculation.
                Shape is ``(num_bin, anything, 3)``: the last axis is
                ``[t_deriv - eps, t_deriv, t_deriv + eps]`` and the derivatives are taken at
                index 1. Default is ``None``: ``[-eps, 0.0, eps]`` for every binary
                (``(num_bin, 1, 3)``).
            eps (float, optional): Step size for central differencing [s]; also used as the
                step when ``t`` is given (it must match ``t``). Default ``5e4``.

        Returns:
            tuple: ``(f, fdot_new, fddot_new)``, each ``(num_bin, anything)``: the
            frequency, first and second derivative at ``t_deriv``.

        Raises:
            ValueError: If ``t`` is not ``None``, a list or an ``np.ndarray``.
            AssertionError: If ``t`` is not 3D with a last axis of 3 and ``num_bin`` rows.

        Warning:
            As written it calls ``get_vLOS(A2, varpi, e2, n2, T2, t)`` while the signature
            is ``get_vLOS(t, A2, varpi, e2, n2, T2)``, so the result is wrong (or the call
            fails on shapes). The second derivative also divides by ``(2 eps)**2`` instead
            of ``eps**2``.
        """
        A2 = np.atleast_1d(A2).copy()
        varpi = np.atleast_1d(varpi).copy()
        e2 = np.atleast_1d(e2).copy()
        P2 = np.atleast_1d(P2).copy()
        n2 = 2 * np.pi / (P2 * YRSID_SI)
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
        f_temp *= 1.0 + self.get_vLOS(A2, varpi, e2, n2, T2, t) / C_SI

        fdot_new = (f_temp[:, :, 2] - f_temp[:, :, 0]) / (2 * eps)

        fddot_new = (f_temp[:, :, 2] - 2 * f_temp[:, :, 1] + f_temp[:, :, 0]) / (2 * eps) ** 2

        return (f_temp[:, :, 1], fdot_new, fddot_new)


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
    """Get ``A2``, ``varpi``, ``T2`` from third-body parameters.

    Get all the third-body factors that go into the waveform computation:
    ``A2``, ``varpi``, and ``T2`` (the extra :meth:`GBGPUThirdBody.run_wave` arguments,
    with ``e2`` and ``P2`` [yr]). Following arXiv:1806.00500: the sky projection
    ``C = cos(theta) sin(iota) + sin(theta) cos(iota) sin(lam - Omega2)``,
    ``S = sin(theta) cos(lam - Omega2)`` (``theta = pi / 2 - beta``) gives
    ``A2 = (mc / (M + mc)) sqrt(G (M + mc) / p2) sqrt(C^2 + S^2)`` and
    ``varpi = (omega2 + arctan(C / -S)) mod 2 pi``, with the semilatus rectum
    ``p2 = a2 (1 - e2^2)``, ``a2 = (G M P2^2 / (4 pi^2))^(1/3)``. The projection only needs
    ``lam`` / ``beta`` and the Euler angles in one common frame (the ecliptic below; use the
    orbits frame, ICRS for L1 orbits, to match :class:`GBGPUThirdBody`'s sky angles).

    Args:
        M (np.ndarray): Total mass of inner
            Galactic binary in Solar Masses.
        mc (np.ndarray): Mass of third body in units of
            ``third_mass_unit`` kwarg.
        P2 (np.ndarray): Orbital period of third body
            in units of ``third_period_unit`` kwarg.
        e2 (np.ndarray): Orbital eccentricity of third body.
        iota (np.ndarray): Orbital inclination of third body
            in radians. This orbital inclination is one of the three euler angles describing
            the rotation of the third-body orbital frame to ecliptic frame.
        Omega2, omega2 (np.ndarray): The other two of three
            euler angles describing the third-body orbital frame rotation from
            the ecliptic frame. See Figure 1 in
            `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_.
        phi2 (np.ndarray): Orbital phase in radians.
        lam (np.ndarray): Ecliptic longitude in radians.
        beta (np.ndarray): Ecliptic latitude in radians.
        third_mass_unit (str, optional): Mass unit for third-body mass. Options are
            ``"Mjup"`` for Jupiter Masses or ``"MSUN"`` for Solar Masses. Default is
            ``"Mjup"``.
        third_period_unit (str, optional): Time unit for third-body period. Options are
            ``"sec"`` for seconds or ``"yrs"`` for years. Default is ``"yrs"``.

    Returns:
        tuple: ``(A2, varpi, T2)``: velocity amplitude [m/s], phase offset [rad] and time
        of pericenter passage [yr] (:func:`get_T2`).

    Raises:
        NotImplementedError: For an unknown unit string.
        AssertionError: If the array shapes differ (all inputs must be arrays of one
            shape; Python floats fail on ``.shape``).

    Warning:
        ``M``, ``mc`` and ``P2`` are converted to SI IN PLACE (``*=``), so the caller's
        arrays change. As written the function also uses a constant ``G`` that is not
        defined in this module (``lisatools.utils.constants`` provides ``G_SI``), so it
        raises ``NameError`` before returning.
    """

    # adjust inner binary mass to kg
    M *= MSUN_SI

    # adjust perturber mass to kg
    if third_mass_unit == "Mjup":
        factor = Mjup
    elif third_mass_unit == "MSUN":
        factor = MSUN_SI

    else:
        raise NotImplementedError

    mc *= factor

    # adjust perturber period to seconds
    if third_period_unit == "yrs":
        P2 *= YRSID_SI

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
    """Get ``T2`` from third-body parameters.

    Converts the orbital phase ``phi2`` (true anomaly) to the eccentric anomaly
    ``u2 = 2 arctan(sqrt((1 - e2) / (1 + e2)) tan(phi2 / 2))`` and the time since pericenter
    ``(u2 - e2 sin u2) / n2``, ``n2 = 2 pi / P2``; negative values are wrapped by one period.

    Args:
        P2 (float or np.ndarray): Orbital period of third body
            in units of ``third_period_unit`` kwarg.
        e2 (float or np.ndarray): Orbital eccentricity of third body.
        phi2 (float or np.ndarray): Orbital phase in radians.
        third_period_unit (str, optional): Time unit for third-body period. Options are
            ``"sec"`` for seconds or ``"yrs"`` for years. Default is ``"yrs"``.

    Returns:
        float or np.ndarray: ``T2`` in YEARS (``YRSID_SI``), whatever the input unit.

    Raises:
        NotImplementedError: For an unknown ``third_period_unit``.

    Warning:
        With ``"yrs"`` an array ``P2`` is converted to seconds IN PLACE.

    Example:
        >>> from gbgpu.thirdbody import get_T2
        >>> float(get_T2(1.0, 0.0, 0.0))
        0.0
    """
    # adjust perturber period to seconds
    if third_period_unit == "yrs":
        P2 *= YRSID_SI

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
    T2 = (temp_T2 / YRSID_SI) * (temp_T2 >= 0.0) + ((P2 - np.abs(temp_T2)) / YRSID_SI) * (
        temp_T2 < 0.0
    )
    return T2
